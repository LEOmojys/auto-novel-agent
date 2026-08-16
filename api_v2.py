import gradio as gr
import os
import json
from typing import List, Tuple, Optional
from tools_history_save import novel_save
from tools_writting import writting_novel
import re
from config import CHARACTERS_FOLDER, NOVEL_FOLDER, NOVEL_STYLES_FOLDER, NOVEL_WORLD_FOLDER


def extract_chapter_number(filename: str) -> Optional[int]:
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def get_characters_list() -> List[str]:
    characters = []
    if os.path.exists(CHARACTERS_FOLDER):
        for file in os.listdir(CHARACTERS_FOLDER):
            if file.endswith('.json'):
                with open(os.path.join(CHARACTERS_FOLDER, file), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if 'name' in data:
                            characters.append(data['name'])
                    except (json.JSONDecodeError, IOError):
                        continue
    return characters


def get_world_list() -> List[str]:
    worlds = []
    if os.path.exists(NOVEL_WORLD_FOLDER):
        for file in os.listdir(NOVEL_WORLD_FOLDER):
            if file.endswith('.txt'):
                worlds.append(file.replace('.txt', ''))
    return worlds


def get_styles_list() -> List[str]:
    styles = []
    if os.path.exists(NOVEL_STYLES_FOLDER):
        for file in os.listdir(NOVEL_STYLES_FOLDER):
            if file.endswith('.txt'):
                styles.append(file.replace('.txt', ''))
    return styles


def get_novel_history_list() -> List[str]:
    novels = []
    if os.path.exists(NOVEL_FOLDER):
        for file in os.listdir(NOVEL_FOLDER):
            if file.endswith('.json'):
                novels.append(file.replace('.json', ''))
    return novels


def load_previous_chapter(chapter_number: str) -> str:
    if not chapter_number or not chapter_number.isdigit():
        return ""
    current_chapter = int(chapter_number)
    prev_chapter = current_chapter - 1
    if os.path.exists(NOVEL_FOLDER):
        for file in os.listdir(NOVEL_FOLDER):
            if file.endswith('.json'):
                file_chapter = extract_chapter_number(file)
                if file_chapter == prev_chapter:
                    file_path = os.path.join(NOVEL_FOLDER, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            return data.get('passage', '')
                        except (json.JSONDecodeError, IOError):
                            return ""
    return ""


def update_last_passage(chapter_number: str) -> str:
    return load_previous_chapter(chapter_number)


def load_history_novel(history_file: str) -> str:
    if not history_file:
        return ""
    file_path = os.path.join(NOVEL_FOLDER, f"{history_file}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data.get('passage', '')
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading history novel: {e}")
                return f"加载失败: {e}"
    return "未找到文件"


def save_novel_with_confirm(passage, article_name, chapter_number, chapter_name, confirm_save):
    if not confirm_save:
        return "请先勾选确认保存选项", gr.update()

    if not all([passage, article_name, chapter_number, chapter_name]):
        return "保存失败: 篇章名、章节序号、章节名称和小说内容不能为空。", gr.update()

    try:
        file_path = novel_save(passage, article_name, chapter_number, chapter_name)
        message = f"保存成功！\n文件已保存至: {file_path}"
        print(message)
    except Exception as e:
        message = f"保存失败！\n错误类型: {type(e).__name__}\n错误信息: {e}"
        print(message)

    updated_choices = get_novel_history_list()
    return message, gr.update(choices=updated_choices, value=f"{chapter_number} {article_name} {chapter_name}")


def find_character_filepath(character_name: str) -> Optional[str]:
    if not character_name or not os.path.exists(CHARACTERS_FOLDER):
        return None

    for file in os.listdir(CHARACTERS_FOLDER):
        if file.endswith('.json'):
            file_path = os.path.join(CHARACTERS_FOLDER, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if data.get('name') == character_name:
                        return file_path
                except (json.JSONDecodeError, IOError):
                    continue
    return None


def update_character_selector(selected_characters: List[str]) -> Tuple:
    if not selected_characters:
        return (
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            "",
            "",
            None,
        )

    return (
        gr.update(visible=True),
        gr.update(choices=selected_characters, value=selected_characters[0]),
        *load_character_data(selected_characters[0]),
    )


def load_character_data(character_name: str) -> Tuple:
    if not character_name:
        return "", "", None

    filepath = find_character_filepath(character_name)
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                char_data = json.load(f)
                char_json_str = json.dumps(char_data, indent=4, ensure_ascii=False)
                return char_json_str, "", character_name
            except (json.JSONDecodeError, IOError) as e:
                return f"加载失败: {e}", "", None

    return f"未找到角色 '{character_name}' 的数据文件", "", None


def save_character_details(character_name: str, character_json_str: str) -> Tuple[str, str]:
    if not character_name:
        return "保存失败: 未选择角色", ""

    filepath = find_character_filepath(character_name)
    if not filepath:
        return f"保存失败: 找不到角色 '{character_name}' 的原始文件", ""

    try:
        new_data = json.loads(character_json_str)
        if not isinstance(new_data, dict) or new_data.get('name') != character_name:
            return f"保存失败: JSON 中 'name' 字段必须为 '{character_name}'", ""
    except json.JSONDecodeError as e:
        return f"保存失败: 内容不是有效的 JSON 格式\n错误信息: {e}", ""

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        message = f"角色 '{character_name}' 的信息已成功更新！"
        print(message)
        return message, character_name
    except IOError as e:
        message = f"保存角色 '{character_name}' 时写入文件失败\n错误: {e}"
        print(message)
        return message, ""


def create_novel_app():
    with gr.Blocks(title="小说设计与生成工具") as demo:

        editing_character_state = gr.State(value=None)

        gr.Markdown("# 小说设计与生成工具")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 历史查询")
                history_dropdown = gr.Dropdown(
                    choices=get_novel_history_list(),
                    label="选择历史小说章节",
                    interactive=True,
                )
                history_btn = gr.Button("查询历史")

                gr.Markdown("## 小说内容")
                passage_textbox = gr.Textbox(
                    label="小说文本",
                    lines=20,
                    placeholder="这里将显示生成的小说内容，您可以在此编辑...",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        confirm_save_checkbox = gr.Checkbox(label="确认保存", value=False)
                    with gr.Column(scale=10):
                        save_btn = gr.Button("保存内容")
                save_output = gr.Textbox(label="保存状态", interactive=False, lines=2)

            with gr.Column(scale=1):
                gr.Markdown("## 小说元信息")
                with gr.Row():
                    article_name_textbox = gr.Textbox(label="篇章名")
                    chapter_number_textbox = gr.Textbox(label="章节序号", placeholder="数字，如：1、2...")
                    chapter_name_textbox = gr.Textbox(label="章节名称")

                last_passage_textbox = gr.Textbox(
                    label="上一章内容", lines=3, max_lines=3,
                    placeholder="自动从上一章填充或手动输入...",
                )
                this_passage_request_textbox = gr.Textbox(
                    label="本章要求", lines=5, max_lines=5,
                    placeholder="输入本章的写作要求...",
                )

                characters_dropdown = gr.Dropdown(
                    choices=get_characters_list(),
                    label="出场角色",
                    interactive=True,
                    multiselect=True,
                )

                with gr.Row():
                    world_dropdown = gr.Dropdown(
                        choices=get_world_list(),
                        label="世界观",
                        interactive=True,
                        multiselect=True,
                    )
                    styles_dropdown = gr.Dropdown(
                        choices=get_styles_list(),
                        label="写作风格",
                        interactive=True,
                        multiselect=True,
                    )

                words_slider = gr.Slider(
                    minimum=100, maximum=10000,
                    value=3000, step=100,
                    label="写作字数",
                )
                is_npc_checkbox = gr.Checkbox(label="是否引入NPC")
                generate_btn = gr.Button("生成内容", variant="primary")

        with gr.Column(visible=False) as character_editor_column:
            gr.Markdown("## 角色卡编辑")
            character_edit_dropdown = gr.Dropdown(
                choices=[],
                label="选择要编辑的角色",
                interactive=True,
            )
            character_editor_textbox = gr.Textbox(
                label="角色卡内容 (JSON 格式)",
                lines=12,
            )
            character_save_status = gr.Textbox(label="保存状态", interactive=False, lines=2)
            character_save_btn = gr.Button("保存角色修改")

        chapter_number_textbox.change(
            fn=update_last_passage,
            inputs=chapter_number_textbox,
            outputs=last_passage_textbox,
        )
        history_btn.click(
            fn=load_history_novel,
            inputs=history_dropdown,
            outputs=passage_textbox,
        )
        generate_btn.click(
            fn=writting_novel,
            inputs=[
                last_passage_textbox, this_passage_request_textbox, article_name_textbox,
                chapter_number_textbox, chapter_name_textbox, characters_dropdown,
                world_dropdown, styles_dropdown, words_slider, is_npc_checkbox,
            ],
            outputs=passage_textbox,
        )
        save_btn.click(
            fn=save_novel_with_confirm,
            inputs=[
                passage_textbox, article_name_textbox, chapter_number_textbox,
                chapter_name_textbox, confirm_save_checkbox,
            ],
            outputs=[save_output, history_dropdown],
        )

        characters_dropdown.change(
            fn=update_character_selector,
            inputs=characters_dropdown,
            outputs=[
                character_editor_column,
                character_edit_dropdown,
                character_editor_textbox,
                character_save_status,
                editing_character_state,
            ],
        )

        character_edit_dropdown.change(
            fn=load_character_data,
            inputs=character_edit_dropdown,
            outputs=[
                character_editor_textbox,
                character_save_status,
                editing_character_state,
            ],
        )

        character_save_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=None,
            js="function() { return confirm('当前操作涉及修改角色卡底层数据，是否确认？'); }",
        ).then(
            fn=save_character_details,
            inputs=[editing_character_state, character_editor_textbox],
            outputs=[character_save_status, editing_character_state],
        )

        return demo


if __name__ == "__main__":
    app = create_novel_app()
    app.launch()
