# Utility tools for the customer service agent.
# This module defines general-purpose tools that can be used by the agent for various tasks that do not fit into the structured data analysis or summarization categories.
# The calculate_expression tool is implemented to allow the agent to perform basic mathematical calculations when the user asks for totals, sums, or other arithmetic operations based on counts or distributions provided by the structured tools.
# This tool is designed to safely evaluate simple mathematical expressions while preventing the execution of arbitrary code, ensuring that the agent can provide accurate calculations without security risks.
# It can be invoked whenever the agent's reasoning process determines that a mathematical operation is needed to answer the user's query effectively.

from __future__ import annotations
import ast
import operator as op
from langchain_core.tools import tool

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _evaluate_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body)
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)
        operator = type(node.op)
        if operator not in _ALLOWED_BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator.__name__}")
        return _ALLOWED_BINARY_OPERATORS[operator](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand)
        operator = type(node.op)
        if operator not in _ALLOWED_UNARY_OPERATORS:
            raise ValueError(
                f"Unsupported unary operator: {operator.__name__}")
        return _ALLOWED_UNARY_OPERATORS[operator](operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError(
        "Only numeric literals and basic arithmetic operators are allowed.")


@tool
def calculate_expression(expression: str) -> str:
    """
    Evaluate a basic mathematical expression string.
    Supports basic operations: +, -, *, /, (, ).
    Example input: "250 + 50" or "(100 * 5) / 2"
    Use this tool whenever the user asks to sum, add, subtract, or total previous counts.
    """
    sanitized = "".join(c for c in expression if c in "0123456789+-*/(). ")
    try:
        parsed = ast.parse(sanitized, mode="eval")
        result = _evaluate_ast(parsed)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except (ValueError, SyntaxError) as error:
        return f"Error evaluating mathematical expression: {error}"
