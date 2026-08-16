import gradio as gr
import os
import json
from typing import List, Optional
from tools_history_save import novel_save
from tools_writting import writting_novel
import re
from config import (
    NOVEL_STYLES_FOLDER,
    DEFAULT_PROJECT,
    get_project_list,
    get_project_paths,
    ensure_project_dirs,
)
from evaluate import rule_based_metrics, llm_evaluate_chapter, load_characters, EVALUATION_DIMS


def extract_chapter_number(filename: str) -> Optional[int]:
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def get_characters_list(project: str = DEFAULT_PROJECT) -> List[str]:
    folder = get_project_paths(project)["characters"]
    characters = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.json'):
                with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if 'name' in data:
                            characters.append(data['name'])
                    except (json.JSONDecodeError, IOError):
                        continue
    return characters


def get_world_list(project: str = DEFAULT_PROJECT) -> List[str]:
    folder = get_project_paths(project)["worlds"]
    worlds = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                worlds.append(file.replace('.txt', ''))
    return worlds


def get_novel_history_list(project: str = DEFAULT_PROJECT) -> List[str]:
    folder = get_project_paths(project)["novel"]
    novels = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.json'):
                novels.append(file.replace('.json', ''))
    return novels


def load_previous_chapter(chapter_number: str, project: str = DEFAULT_PROJECT) -> str:
    folder = get_project_paths(project)["novel"]
    if not chapter_number or not chapter_number.isdigit():
        return ""
    current_chapter = int(chapter_number)
    prev_chapter = current_chapter - 1
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.json'):
                file_chapter = extract_chapter_number(file)
                if file_chapter == prev_chapter:
                    file_path = os.path.join(folder, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            return data.get('passage', '')
                        except (json.JSONDecodeError, IOError):
                            return ""
    return ""


def load_history_novel(history_file: str, project: str = DEFAULT_PROJECT) -> str:
    folder = get_project_paths(project)["novel"]
    if not history_file:
        return ""
    file_path = os.path.join(folder, f"{history_file}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data.get('passage', '')
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading history novel: {e}")
                return f"加载失败: {e}"
    return "未找到文件"


def find_character_filepath(character_name: str, project: str = DEFAULT_PROJECT) -> Optional[str]:
    folder = get_project_paths(project)["characters"]
    if not character_name or not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        if file.endswith('.json'):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if data.get('name') == character_name:
                        return file_path
                except (json.JSONDecodeError, IOError):
                    continue
    return None


def update_project(project: str):
    return (
        gr.update(choices=get_characters_list(project)),
        gr.update(choices=get_world_list(project), value=[]),
        gr.update(choices=get_novel_history_list(project), value=None),
        "",
        "",
    )

# --- 修改后的函数 ---

def save_novel_with_confirm(passage, article_name, chapter_number, chapter_name, confirm_save):
    """带确认的保存函数, 并在成功后更新历史列表，返回详细状态信息"""
    if not confirm_save:
        return "请先勾选确认保存选项", gr.update()
    
    # 增加对必要字段的检查
    if not all([passage, article_name, chapter_number, chapter_name]):
        return "保存失败: 篇章名、章节序号、章节名称和小说内容不能为空。", gr.update()

    try:
        # novel_save 应该在成功时返回保存路径，便于日志记录
        file_path = novel_save(passage, article_name, chapter_number, chapter_name)
        message = f"保存成功！\n文件已保存至: {file_path}"
        print(message) # 后台日志
    except Exception as e:
        message = f"保存失败！\n错误类型: {type(e).__name__}\n错误信息: {e}"
        print(message) # 后台日志

    updated_choices = get_novel_history_list()
    return message, gr.update(choices=updated_choices, value=f"{chapter_number} {article_name} {chapter_name}")


def update_last_passage(chapter_number: str, project: str = DEFAULT_PROJECT) -> str:
    return load_previous_chapter(chapter_number, project)


def save_novel_with_confirm(passage, article_name, chapter_number, chapter_name, this_passage_request, confirm_save, project):
    if not confirm_save:
        return "请先勾选确认保存选项", gr.update()
    if not all([passage, article_name, chapter_number, chapter_name]):
        return "保存失败: 篇章名、章节序号、章节名称和小说内容不能为空。", gr.update()
    try:
        file_path = novel_save(
            passage,
            article_name,
            chapter_number,
            chapter_name,
            this_passage_request=this_passage_request,
            project=project,
        )
        message = f"保存成功！\n文件已保存至: {file_path}"
        print(message)
    except Exception as e:
        message = f"保存失败！\n错误类型: {type(e).__name__}\n错误信息: {e}"
        print(message)
    updated_choices = get_novel_history_list(project)
    saved_value = f"{chapter_number} {article_name.strip()} {chapter_name.strip()}"
    return message, gr.update(choices=updated_choices, value=saved_value)


def save_character_details(character_name: str, character_json_str: str, project: str = DEFAULT_PROJECT) -> str:
    if not character_name:
        return "保存失败: 没有指定要保存的角色。"
    filepath = find_character_filepath(character_name, project)
    if not filepath:
        return f"保存失败: 找不到角色 '{character_name}' 的原始文件。"
    try:
        new_data = json.loads(character_json_str)
        if not isinstance(new_data, dict) or new_data.get('name') != character_name:
            return f"保存失败: JSON 中 'name' 字段必须为 '{character_name}'"
    except json.JSONDecodeError as e:
        return f"保存失败: 内容不是有效的 JSON 格式\n错误信息: {e}"
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        message = f"角色 '{character_name}' 的信息已成功更新！"
        print(message)
        return message
    except IOError as e:
        message = f"保存角色时写入文件失败\n错误: {e}"
        print(message)
        return message


def format_characters_summary(selected_characters: List[str], project: str = DEFAULT_PROJECT) -> str:
    if not selected_characters:
        return ""
    lines = ["### 📋 已选角色概览", ""]
    lines.append("| 角色 | 简介 | 性格 | 近期剧情 |")
    lines.append("|------|------|------|------|")
    for name in selected_characters:
        filepath = find_character_filepath(name, project)
        if filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    intro = data.get("role_introduction", "") or "-"
                    personality = data.get("role_personality", "") or "-"
                    history = data.get("plot_history", [])
                    if history:
                        recent = history[-3:] if len(history) > 3 else history
                        entries = []
                        for item in recent:
                            for key, val in item.items():
                                summary = val[:40] + ("..." if len(val) > 40 else "")
                                entries.append(f"{key}: {summary}")
                        history_text = "<br>".join(entries)
                    else:
                        history_text = "-"
                    lines.append(f"| {name} | {intro} | {personality} | {history_text} |")
                except (json.JSONDecodeError, IOError):
                    lines.append(f"| {name} | ⚠️ 加载失败 | - | - |")
        else:
            lines.append(f"| {name} | ⚠️ 未找到 | - | - |")
    return "\n".join(lines)


def update_character_editor(selected_characters: List[str], project: str):
    if not selected_characters:
        return gr.update(visible=False), "", "", None
    summary_md = format_characters_summary(selected_characters, project)
    filepath = find_character_filepath(selected_characters[0], project)
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                char_data = json.load(f)
                char_json_str = json.dumps(char_data, indent=4, ensure_ascii=False)
                return gr.update(visible=True), summary_md, char_json_str, selected_characters[0]
            except (json.JSONDecodeError, IOError) as e:
                return gr.update(visible=True), summary_md, f"加载失败: {e}", None
    return gr.update(visible=False), summary_md, f"未找到角色", None


def evaluate_for_ui(passage: str, target_words: int, project: str = DEFAULT_PROJECT, chapter_requirement: str = "") -> str:
    if not passage.strip():
        return "### ⚠️ 请先生成或加载小说内容后再进行评估"
    characters_folder = get_project_paths(project)["characters"]
    characters = load_characters(characters_folder)
    rules = rule_based_metrics(passage, characters, target_words)
    llm = llm_evaluate_chapter(
        passage,
        characters,
        "当前章节",
        chapter_requirement=chapter_requirement,
    )

    sensory_items = "  ".join(
        f"{k}: {v}" for k, v in rules["sensory"].items() if v > 0
    )
    dev_sign = "+" if rules["deviation_pct"] >= 0 else ""
    chars_display = ", ".join(rules["characters_appeared"]) if rules["characters_appeared"] else "未检测到"

    lines = [
        "## 📊 评估结果",
        "",
        f"**综合均分: {llm['avg_score']} / 5.0**",
        "",
        "### 规则指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 字数 | {rules['chinese_chars']} (目标 {target_words}, {dev_sign}{rules['deviation_pct']}%) |",
        f"| 感官密度 | {rules['sensory_density']} / 千字 |",
        f"| 感官分布 | {sensory_items} |",
        f"| 对话占比 | {rules['dialogue_ratio']}% |",
        f"| 段落数 | {rules['paragraph_count']} |",
        f"| 出场角色 | {chars_display} |",
        "",
        "### LLM 评分",
        "",
        "| 维度 | 评分 | 理由 |",
        "|------|------|------|",
    ]

    for dim in EVALUATION_DIMS:
        score = llm["scores"].get(dim, "?")
        reason = llm["reasons"].get(dim, "")
        star = "⭐" * score + "☆" * (5 - score)
        lines.append(f"| {dim} | {star} ({score}/5) | {reason} |")

    return "\n".join(lines)


def generate_novel_wrapper(last_passage, this_passage_request, article_name,
                          chapter_number, chapter_name, characters,
                          world, styles, words, is_NPC, project):
    return writting_novel(
        last_passage, this_passage_request, article_name,
        chapter_number, chapter_name, characters,
        ", ".join(world) if isinstance(world, list) else world,
        ", ".join(styles) if isinstance(styles, list) else styles,
        words, is_NPC, project=project,
    )


def get_styles_list() -> List[str]:
    styles = []
    if os.path.exists(NOVEL_STYLES_FOLDER):
        for file in os.listdir(NOVEL_STYLES_FOLDER):
            if file.endswith('.txt'):
                styles.append(file.replace('.txt', ''))
    return styles


def create_novel_app():
    with gr.Blocks(title="小说设计与生成工具") as demo:
        editing_character_state = gr.State(value=None)
        project_state = gr.State(value=DEFAULT_PROJECT)

        gr.Markdown("# 小说设计与生成工具")

        with gr.Row():
            project_dropdown = gr.Dropdown(
                choices=get_project_list(),
                value=DEFAULT_PROJECT,
                label="📁 小说项目",
                interactive=True,
                scale=1,
            )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 历史查询")
                history_dropdown = gr.Dropdown(
                    choices=get_novel_history_list(),
                    label="选择历史小说章节",
                    interactive=True,
                    allow_custom_value=True,
                )
                history_btn = gr.Button("查询历史")
                
                gr.Markdown("## 小说内容")
                passage_textbox = gr.Textbox(
                    label="小说文本",
                    lines=20,
                    placeholder="这里将显示生成的小说内容，您可以在此编辑..."
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        confirm_save_checkbox = gr.Checkbox(label="确认保存", value=False)
                    with gr.Column(scale=10):
                        save_btn = gr.Button("保存内容")
                save_output = gr.Textbox(label="保存状态", interactive=False, lines=2)

                evaluate_btn = gr.Button("📊 评估内容", variant="secondary")
                eval_result_md = gr.Markdown()
            
            with gr.Column(scale=1):
                gr.Markdown("## 小说元信息")
                with gr.Row():
                    article_name_textbox = gr.Textbox(label="篇章名")
                    chapter_number_textbox = gr.Textbox(label="章节序号", placeholder="数字，如：1、2...")
                    chapter_name_textbox = gr.Textbox(label="章节名称")
                
                last_passage_textbox = gr.Textbox(
                    label="上一章内容", lines=3, max_lines=3, placeholder="自动从上一章填充或手动输入..."
                )
                this_passage_request_textbox = gr.Textbox(
                    label="本章要求", lines=5, max_lines=5, placeholder="输入本章的写作要求..."
                )
                
                characters_dropdown = gr.Dropdown(
                    choices=get_characters_list(), label="出场角色", interactive=True, multiselect=True
                )
                
                # 修改后的角色详情区域
                with gr.Accordion("出场角色详情", open=False, visible=False) as character_details_accordion:
                    character_summary_md = gr.Markdown()
                    character_details_editor = gr.Textbox(
                        label="角色卡内容 (JSON格式)", 
                        lines=10,
                    )
                    save_character_btn = gr.Button("保存角色修改")
                    character_save_status = gr.Textbox(label="角色保存状态", interactive=False)
                
                with gr.Row():
                    world_dropdown = gr.Dropdown(
                        choices=get_world_list(), label="世界观", interactive=True, multiselect=True
                    )
                    styles_dropdown = gr.Dropdown(
                        choices=get_styles_list(), label="写作风格", interactive=True, multiselect=True
                    )
                
                words_slider = gr.Slider(
                    minimum=100, maximum=10000, value=3000, step=100, label="写作字数"
                )
                is_npc_checkbox = gr.Checkbox(label="是否引入NPC")
                generate_btn = gr.Button("生成内容", variant="primary")
        
        
        chapter_number_textbox.change(
            fn=lambda chapter, proj: update_last_passage(chapter, proj),
            inputs=[chapter_number_textbox, project_state],
            outputs=last_passage_textbox,
        )
        history_btn.click(
            fn=lambda hist, proj: load_history_novel(hist, proj),
            inputs=[history_dropdown, project_state],
            outputs=passage_textbox,
        )
        generate_btn.click(
            fn=generate_novel_wrapper,
            inputs=[
                last_passage_textbox, this_passage_request_textbox, article_name_textbox,
                chapter_number_textbox, chapter_name_textbox, characters_dropdown,
                world_dropdown, styles_dropdown, words_slider, is_npc_checkbox, project_state,
            ],
            outputs=passage_textbox,
        )
        save_btn.click(
            fn=save_novel_with_confirm,
            inputs=[
                passage_textbox, article_name_textbox, chapter_number_textbox,
                chapter_name_textbox, this_passage_request_textbox, confirm_save_checkbox, project_state,
            ],
            outputs=[save_output, history_dropdown],
        )

        evaluate_btn.click(
            fn=lambda passage, words, proj, req: evaluate_for_ui(passage, words, proj, req),
            inputs=[passage_textbox, words_slider, project_state, this_passage_request_textbox],
            outputs=eval_result_md,
        )

        characters_dropdown.change(
            fn=lambda chars, proj: update_character_editor(chars, proj),
            inputs=[characters_dropdown, project_state],
            outputs=[character_details_accordion, character_summary_md, character_details_editor, editing_character_state],
        )

        save_character_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=None,
            js="function() { return confirm('当前操作涉及修改角色卡底层数据，是否确认？'); }",
        ).then(
            fn=lambda name, json_str, proj: save_character_details(name, json_str, proj),
            inputs=[editing_character_state, character_details_editor, project_state],
            outputs=character_save_status,
        )

        project_dropdown.change(
            fn=lambda proj: update_project(proj),
            inputs=project_dropdown,
            outputs=[characters_dropdown, world_dropdown, history_dropdown, passage_textbox, eval_result_md],
        ).then(
            fn=lambda p: p,
            inputs=project_dropdown,
            outputs=project_state,
        )
        
        return demo

if __name__ == "__main__":
    app = create_novel_app()
    app.launch()
