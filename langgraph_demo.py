from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Literal
from langchain_core.messages.base import BaseMessage
import tools
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage

# read local .env file
load_dotenv()


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = os.getenv(var)


# get the environment variable
_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph_demo"
_model = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools.toolList)


def call_openai(messages: list):
    return _model.invoke(messages)


def simple_graph():
    graph = MessageGraph()

    graph.add_node("oracle", call_openai)
    graph.add_edge("oracle", END)

    graph.set_entry_point("oracle")

    return graph.compile()


def agent_chain(messages: list):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant who always speaks in pirate dialect"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | _model
    return chain.invoke(messages)


def graph_with_agent_chain():
    graph = MessageGraph()

    graph.add_node("oracle", agent_chain)
    graph.add_edge("oracle", END)

    graph.set_entry_point("oracle")

    return graph.compile()


def router(state: list[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    tool_names = [call['function']['name'] for call in tool_calls]
    return tool_names[0]


def graph_with_tools():
    model_with_tools = _model

    builder = MessageGraph()

    builder.add_node("oracle", model_with_tools)

    tool_node = ToolNode(tools.toolList)
    builder.add_node("multiply", tool_node)

    builder.add_edge("multiply", END)

    builder.set_entry_point("oracle")

    builder.add_conditional_edges("oracle", router)
    return builder.compile()


def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right


class AgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = _model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def graph_cycle():
    tool_node = ToolNode(tools.toolList)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge('tools', 'agent')
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )
    return workflow.compile()


if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="what is 123*456?")]}
    print(graph_cycle().invoke(inputs))
    # print(graph_with_tools().invoke(HumanMessage("What is 123*456?")))
