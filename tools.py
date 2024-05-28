from langchain_core.tools import tool
import os


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


toolList = [multiply]
