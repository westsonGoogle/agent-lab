"""计算器:用 AST 白名单求值,绝不使用 eval()。

为什么不能直接 eval?因为模型可能被提示注入,让它传
`__import__("os").system("rm -rf ~")`。这里只允许数字、四则运算、
乘方和几个数学函数。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

MAX_EXPR_LENGTH = 200
MAX_POW_EXPONENT = 1000

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def calculate(expression: str) -> int | float:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    if len(expression) > MAX_EXPR_LENGTH:
        raise ValueError(f"表达式太长(上限 {MAX_EXPR_LENGTH} 字符)")
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval(tree.body)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("结果是无穷大或 NaN")
    if abs(result) > 1e100:
        raise ValueError("结果过大")
    return result


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError(f"指数过大(上限 {MAX_POW_EXPONENT})")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
        return op(_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"未知变量: {node.id}")
        return _CONSTANTS[node.id]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("只允许调用内置数学函数")
        args = [_eval(arg) for arg in node.args]
        kwargs = {kw.arg: _eval(kw.value) for kw in node.keywords if kw.arg}
        return _FUNCTIONS[node.func.id](*args, **kwargs)

    raise ValueError(f"不支持的语法: {type(node).__name__}")


CALCULATOR_TOOL = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "要计算的数学表达式,例如 (123 * 456) / 7 或 sqrt(2)",
        }
    },
    "required": ["expression"],
}
