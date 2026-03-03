#!/usr/bin/env python3
"""
文本截取工具
功能：在输入文本中找到指定关键词，保留其前/后n个字符，其余内容删除。
      可选择是否在结果中保留关键词本身，结果自动去重，片段间以换行分隔。
支持：多处匹配、重叠合并、命令行参数、交互式输入
"""

import argparse
import sys


def extract_context(text: str, keyword: str, before: int = 0, after: int = 0,
                    include_keyword: bool = True) -> list[dict]:
    matches = []
    search_start = 0
    kw_len = len(keyword)

    while True:
        pos = text.find(keyword, search_start)
        if pos == -1:
            break

        extract_start = max(0, pos - before)
        extract_end   = min(len(text), pos + kw_len + after)

        if include_keyword:
            fragment = text[extract_start:extract_end]
        else:
            part_before = text[extract_start:pos]
            part_after  = text[pos + kw_len: extract_end]
            fragment    = part_before + part_after

        matches.append({
            "pos":      pos,
            "start":    extract_start,
            "end":      extract_end,
            "fragment": fragment,
        })
        search_start = pos + 1

    return matches


def merge_overlapping(matches: list[dict], text: str,
                      include_keyword: bool, keyword: str) -> list[dict]:
    if not matches:
        return []

    kw_len = len(keyword)
    merged_ranges = []
    cur_start    = matches[0]["start"]
    cur_end      = matches[0]["end"]
    kw_positions = [matches[0]["pos"]]

    for m in matches[1:]:
        if m["start"] <= cur_end:
            cur_end = max(cur_end, m["end"])
            kw_positions.append(m["pos"])
        else:
            merged_ranges.append((cur_start, cur_end, kw_positions))
            cur_start    = m["start"]
            cur_end      = m["end"]
            kw_positions = [m["pos"]]
    merged_ranges.append((cur_start, cur_end, kw_positions))

    result = []
    for start, end, positions in merged_ranges:
        if include_keyword:
            fragment = text[start:end]
        else:
            segment = text[start:end]
            for pos in sorted(positions, reverse=True):
                rel = pos - start
                segment = segment[:rel] + segment[rel + kw_len:]
            fragment = segment

        result.append({
            "pos":      "+".join(str(p) for p in positions),
            "start":    start,
            "end":      end,
            "fragment": fragment,
        })
    return result


def run(text: str, keyword: str, before: int, after: int,
        merge: bool = True, include_keyword: bool = True) -> str:
    matches = extract_context(text, keyword, before, after, include_keyword)
    if not matches:
        return ""

    if merge:
        matches = merge_overlapping(matches, text, include_keyword, keyword)

    # 去重，保留顺序
    seen = set()
    unique = []
    for m in matches:
        f = m["fragment"]
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return "\n".join(unique)


# ──────────────────────────────────────────────
# 命令行 / 交互式入口
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="保留文本中指定关键词前/后n个字符，其余删除"
    )
    parser.add_argument("-t", "--text",        help="输入文本（不提供则从 stdin 读取）")
    parser.add_argument("-f", "--file",        help="从文件读取输入文本")
    parser.add_argument("-k", "--keyword",     help="要查找的关键词")
    parser.add_argument("-b", "--before",      type=int, default=0, help="关键词前保留字符数（默认0）")
    parser.add_argument("-a", "--after",       type=int, default=0, help="关键词后保留字符数（默认0）")
    parser.add_argument("--no-merge",          action="store_true", help="不合并重叠片段")
    parser.add_argument("--exclude-keyword",   action="store_true", help="结果中不包含关键词本身")
    parser.add_argument("-i", "--interactive", action="store_true", help="强制进入交互模式")
    return parser.parse_args()


def interactive_mode():
    print("=" * 55)
    print("  文本截取工具 — 交互模式")
    print("=" * 55)

    print("\n① 输入文本（输入完成后连续按两次回车）：")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    text = "\n".join(lines[:-1])

    if not text.strip():
        text = (
            "Python是一种高级编程语言，由Guido van Rossum于1991年创建。\n"
            "Python的设计哲学强调代码的可读性和简洁性。\n"
            "Python支持多种编程范式，包括面向对象、命令式和函数式编程。\n"
            "Python拥有庞大的标准库，涵盖字符串处理到网络编程等各种功能。\n"
            "许多知名公司如Google、Facebook都在使用Python进行开发。"
        )
        print(f"\n（使用内置示例文本）\n{text}")

    keyword = input("\n② 关键词：").strip()
    if not keyword:
        keyword = "Python"
        print(f"（使用默认关键词：{keyword}）")

    try:
        before = int(input("③ 关键词【前】保留字符数（默认10）：") or "10")
    except ValueError:
        before = 10
    try:
        after = int(input("④ 关键词【后】保留字符数（默认20）：") or "20")
    except ValueError:
        after = 20

    inc = input("⑤ 结果中是否包含关键词本身？[Y/n]：").strip().lower()
    include_keyword = (inc != "n")

    no_merge = input("⑥ 合并重叠片段？[Y/n]：").strip().lower() == "n"

    return text, keyword, before, after, include_keyword, not no_merge


def main():
    args = parse_args()

    if args.interactive or (not args.keyword):
        text, keyword, before, after, include_keyword, merge = interactive_mode()
    else:
        keyword         = args.keyword
        before          = args.before
        after           = args.after
        merge           = not args.no_merge
        include_keyword = not args.exclude_keyword

        if args.file:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            text = sys.stdin.read()

    # ── 执行 ──
    matches = extract_context(text, keyword, before, after, include_keyword)

    kw_label = f"'{keyword}'" + ("" if include_keyword else "（不含关键词）")
    print("\n" + "=" * 55)
    print(f"  关键词：{kw_label}  |  前：{before}字符  |  后：{after}字符")
    print(f"  共找到 {len(matches)} 处匹配")
    print("=" * 55)

    if not matches:
        print(f"\n⚠ 未在文本中找到关键词 '{keyword}'")
        return

    if merge:
        matches = merge_overlapping(matches, text, include_keyword, keyword)

    # 去重，保留顺序
    seen = set()
    unique = []
    for m in matches:
        f = m["fragment"]
        if f not in seen:
            seen.add(f)
            unique.append(f)

    print(f"  去重后片段数：{len(unique)}\n")
    print("=" * 55)
    print("最终保留文本：")
    print("=" * 55)
    print("\n".join(unique))


if __name__ == "__main__":
    main()