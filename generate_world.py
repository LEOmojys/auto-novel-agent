#!/usr/bin/env python3
"""
世界观生成工具 —— 帮助新用户从零开始搭建小说世界观。

用法:
    python generate_world.py              # 交互式流程
    python generate_world.py --auto       # 全自动，随机生成一个世界观
    python generate_world.py --idea       # 只做头脑风暴，列出 10 个创意
"""

import os
import sys
import argparse
from config import DEFAULT_PROJECT, ensure_project_dirs, get_project_paths
from tools_call_deepseek import access_deepseek


BRAINSTORM_PROMPT = """你是一位资深小说编辑和创意顾问。请生成 10 个新颖的小说世界观设定。

要求：
- 每个设定不超过 50 字
- 避免市面上烂大街的组合（穿越+系统、重生+复仇等）
- 每个设定要包含：时代/技术背景 + 独特规则 + 核心冲突
- 编号从 1 到 10
- 风格多样：可以有奇幻、科幻、悬疑、都市、历史等不同方向

直接输出 10 条设定，格式：
1. 【设定名称】一句话描述
2. 【设定名称】一句话描述
..."""


EXPAND_PROMPT = """你是一位资深世界观架构师。请根据下面的设定方向，写一份完整的世界观文档。

设定方向：{idea}

请用 800-1500 字详细描述，包含以下内容：

1. **时代与背景**：这个世界处于什么时代？技术水平如何？关键历史事件是什么？
2. **地理与势力**：世界的地理格局是怎样的？有哪些主要势力/国家/组织？它们之间的关系是什么？
3. **社会结构**：谁掌权？谁被压迫？阶层流动可能吗？
4. **核心冲突**：这个世界的根本矛盾是什么？不同群体在争夺什么？
5. **独特法则**：这个世界有什么不同于现实世界的规则？（魔法体系、科技设定、物理规律等）
6. **叙事调性**：适合写什么风格的故事？光明/黑暗？史诗/日常？热血/压抑？

注意：
- 不要写成一个"列表"，而是一个连贯的、有画面感的描述性文档
- 给创作者留白——不要把所有细节都定死，保留发挥空间
- 语言风格：平实有力，像一位资深设定师在向作者交代世界观"""


def brainstorm():
    print("\n🧠 正在头脑风暴，生成 10 个世界观创意...\n")
    result = access_deepseek(BRAINSTORM_PROMPT, model_name="deepseek-v4-pro")
    print(result)
    return result


def expand_world(idea: str) -> str:
    print(f"\n📝 正在基于创意扩展世界观...")
    prompt = EXPAND_PROMPT.format(idea=idea)
    result = access_deepseek(prompt, model_name="deepseek-v4-pro")
    return result


def save_world(name: str, content: str, project: str = DEFAULT_PROJECT) -> str:
    worlds_folder = get_project_paths(project)["worlds"]
    ensure_project_dirs(project)
    filepath = os.path.join(worlds_folder, f"{name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def interactive(project: str = DEFAULT_PROJECT):
    print("=" * 60)
    print("  小说世界观生成工具")
    print("=" * 60)

    print("\n你想要什么类型的故事？用几个关键词描述即可，比如：")
    print('  "赛博朋克 + 东方修仙"\n  "蒸汽朋克 + 克苏鲁"\n  "古代武侠 + 宇宙科幻"\n')

    user_input = input("> 你的想法（回车跳过，我会帮你头脑风暴）: ").strip()

    if not user_input:
        brainstorm()
        print("\n" + "=" * 60)
        choice = input("请输入你感兴趣的编号（1-10），或直接输入你的自定义方向: ").strip()
    else:
        choice = user_input

    if choice.isdigit():
        print(f"\n⚠️ 直接输入你选中的那条创意的一句话描述（从上面复制），或者输入你自己的方向:")
        choice = input("> ").strip()

    if not choice:
        print("没有输入，退出。")
        return

    content = expand_world(choice)
    print("\n" + "=" * 60)
    print(content)
    print("=" * 60)

    save_name = input("\n保存为文件名（不含扩展名，如 赛博修仙世界）: ").strip()
    if not save_name:
        save_name = "未命名世界观"

    filepath = save_world(save_name, content, project)
    print(f"\n✅ 世界观已保存: {filepath}")
    print(f"   可在 Gradio 界面中通过「世界观」下拉框选中「{save_name}」使用。")


def auto_generate(project: str = DEFAULT_PROJECT):
    print("🤖 自动模式：随机生成一个世界观\n")
    result = access_deepseek(BRAINSTORM_PROMPT, model_name="deepseek-v4-pro")
    print(result)

    first_line = result.strip().split("\n")[0] if result else "一个充满神秘的世界"
    idea = first_line.split(".", 1)[-1].strip() if "." in first_line else first_line

    content = expand_world(idea)
    print("\n" + "=" * 60)
    print(content)
    print("=" * 60)

    filepath = save_world("自动生成世界观", content, project)
    print(f"\n✅ 世界观已保存: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小说世界观生成工具")
    parser.add_argument("--auto", action="store_true", help="全自动生成")
    parser.add_argument("--idea", action="store_true", help="只做头脑风暴")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="小说项目名称")
    args = parser.parse_args()

    if args.auto:
        auto_generate(args.project)
    elif args.idea:
        brainstorm()
    else:
        interactive(args.project)
