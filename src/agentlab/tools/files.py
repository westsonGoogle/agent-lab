"""文件工具:能读本地文件,但被关在 ALLOWED_ROOT 沙箱里。

安全要点:
1. 先拼路径再 resolve(),这样 ../ 和符号链接都会被还原成真实路径
2. 用 is_relative_to 判断是否越界,而不是字符串 startswith
3. 拒绝 .env / .git / 私钥这类敏感文件,避免模型被注入后把密钥读走
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Tool

MAX_READ_BYTES = 200_000
MAX_READ_LINES = 2_000
BLOCKED_NAMES = {".env", ".git", ".ssh", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12"}


def _resolve(root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path 不能为空")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    if not resolved.is_relative_to(root):
        raise PermissionError(f"路径越界: {raw_path} 不在沙箱 {root} 内")

    for part in resolved.relative_to(root).parts:
        if part in BLOCKED_NAMES or part.startswith(".env"):
            raise PermissionError(f"出于安全原因,禁止读取 {part}")
    if resolved.suffix.lower() in BLOCKED_SUFFIXES:
        raise PermissionError(f"出于安全原因,禁止读取 {resolved.suffix} 文件")
    return resolved


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_file_tools(root: Path) -> list[Tool]:
    root = root.expanduser().resolve()

    def read_file(
        path: str,
        start_line: int = 1,
        max_lines: int = 500,
    ) -> str:
        target = _resolve(root, path)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"{path} 是目录,请用 list_dir")

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"文件太大({size} 字节),上限 {MAX_READ_BYTES}")

        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = max(1, int(start_line))
        count = max(1, min(int(max_lines), MAX_READ_LINES))
        chunk = lines[start - 1 : start - 1 + count]
        end = start + len(chunk) - 1 if chunk else start - 1

        return json.dumps(
            {
                "path": _display(root, target),
                "start_line": start,
                "end_line": end,
                "total_lines": len(lines),
                "truncated": end < len(lines),
                "content": "\n".join(chunk),
            },
            ensure_ascii=False,
        )

    def list_dir(path: str = ".", max_entries: int = 100) -> str:
        target = _resolve(root, path)
        if not target.exists():
            raise FileNotFoundError(f"目录不存在: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"{path} 不是目录")

        entries = []
        for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))[:max_entries]:
            if item.name in BLOCKED_NAMES or item.name.startswith(".env"):
                continue
            entries.append(
                {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        return json.dumps({"path": _display(root, target), "entries": entries}, ensure_ascii=False)

    return [
        Tool(
            name="read_file",
            description=(
                "读取沙箱目录内的文本文件。返回按行切片的 JSON,"
                "包含 content、total_lines 和 truncated。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对沙箱根目录的文件路径"},
                    "start_line": {
                        "type": "integer",
                        "description": "起始行号,从 1 开始,默认 1",
                        "minimum": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "最多读多少行,默认 500",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
            },
            func=read_file,
            example={"path": "README.md", "start_line": 1, "max_lines": 50},
        ),
        Tool(
            name="list_dir",
            description="列出沙箱目录内的文件和子目录。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径,默认当前目录"},
                    "max_entries": {
                        "type": "integer",
                        "description": "最多返回多少项,默认 100",
                        "minimum": 1,
                    },
                },
                "required": [],
            },
            func=list_dir,
            example={"path": "."},
        ),
    ]
