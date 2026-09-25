"""去掉 Pine 源文件里的注释，生成直接粘贴进 TradingView 的版本.

    python scripts/pine_strip_comments.py pine/impulse_wave_12345.pine pine/impulse_wave.pine

带注释的文件是源文件，改代码改那个；这个脚本只负责生成干净版。

规则：
  - 保留 //@version= 那一行（编译器指令，删了会编译失败）
  - 整行注释删掉；代码后面的行尾注释删掉
  - 字符串里的 // 不算注释（按双引号 / 单引号配对跳过，处理 \\" 转义）
  - 连续空行压成一行，首尾空行去掉
  - input 的 tooltip 不是注释，是设置页里的说明文字，保留

生成后自检：去掉注释前后，每一行代码的内容逐行一致、顺序一致。
"""

from __future__ import annotations

import sys
from pathlib import Path


def comment_start(line: str) -> int:
    """行内第一个不在字符串里的 // 的位置；没有返回 -1。"""
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in ("\"", "'"):
            quote = ch
        elif line.startswith("//", i):
            return i
        i += 1
    return -1


def strip(src: str) -> str:
    out: list[str] = []
    for line in src.split("\n"):
        if line.strip().startswith("//@version"):
            out.append(line.rstrip())
            continue
        k = comment_start(line)
        if k >= 0:
            code = line[:k].rstrip()
            if code.strip():
                out.append(code)
            continue
        out.append(line.rstrip())
    # 连续空行压成一行
    squashed: list[str] = []
    for line in out:
        if not line.strip() and squashed and not squashed[-1].strip():
            continue
        squashed.append(line)
    while squashed and not squashed[-1].strip():
        squashed.pop()
    # 版本行后面紧跟一个空行
    if len(squashed) > 1 and squashed[1].strip():
        squashed.insert(1, "")
    return "\n".join(squashed) + "\n"


def code_lines(src: str) -> list[str]:
    """用来自检：每一行去掉注释后的代码部分（空的不要）。"""
    res = []
    for line in src.split("\n"):
        if line.strip().startswith("//@version"):
            res.append(line.strip())
            continue
        k = comment_start(line)
        code = (line[:k] if k >= 0 else line).rstrip()
        if code.strip():
            res.append(code)
    return res


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1]).read_text()
    out = strip(src)
    # 自检：干净版里的每一行，必须和源文件去掉注释后的代码逐行相同
    a, b = code_lines(src), [l for l in out.split("\n") if l.strip()]
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise SystemExit(f"自检失败：第 {i + 1} 行代码不一致\n  源：{x}\n  出：{y}")
        raise SystemExit(f"自检失败：代码行数不一致 {len(a)} vs {len(b)}")
    left = [l for l in out.split("\n") if comment_start(l) >= 0 and not l.strip().startswith("//@version")]
    if left:
        raise SystemExit(f"自检失败：还有注释残留\n  {left[0]}")
    Path(sys.argv[2]).write_text(out)
    n_src, n_out = len(src.split("\n")), len(out.split("\n"))
    print(f"{sys.argv[1]} → {sys.argv[2]}   {n_src} 行 → {n_out} 行，代码 {len(a)} 行逐行一致，无注释残留")


if __name__ == "__main__":
    main()
