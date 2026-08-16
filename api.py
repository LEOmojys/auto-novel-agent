import gradio as gr
import os
import json
from typing import List, Dict
# 假设这些是你本地的工具函数和配置文件
from tools_history_save import novel_save
from tools_writting import writting_novel
import re
from config import CHARACTERS_FOLDER, NOVEL_FOLDER, NOVEL_STYLES_FOLDER, NOVEL_WORLD_FOLDER

# --- 原有的辅助函数无需改动 ---

def extract_chapter_number(filename):
    """从文件名中提取章节号"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def get_characters_list() -> List[str]:
    """从 CHARACTERS_FOLDER 获取所有角色名称"""
    characters = []
    if os.path.exists(CHARACTERS_FOLDER):
        for file in os.listdir(CHARACTERS_FOLDER):
            if file.endswith('.json'):
                with open(os.path.join(CHARACTERS_FOLDER, file), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if 'name' in data:
                            characters.append(data['name'])
                    except:
                        continue
    return characters

def get_world_list() -> List[str]:
    """从 NOVEL_WORLD_FOLDER 获取所有世界观选项"""
    worlds = []
    if os.path.exists(NOVEL_WORLD_FOLDER):
        for file in os.listdir(NOVEL_WORLD_FOLDER):
            if file.endswith('.txt'):
                worlds.append(file.replace('.txt', ''))
    return worlds

def get_styles_list() -> List[str]:
    """从 NOVEL_STYLES_FOLDER 获取所有写作风格选项"""
    styles = []
    if os.path.exists(NOVEL_STYLES_FOLDER):
        for file in os.listdir(NOVEL_STYLES_FOLDER):
            if file.endswith('.txt'):
                styles.append(file.replace('.txt', ''))
    return styles

def get_novel_history_list() -> List[str]:
    """从 NOVEL_FOLDER 获取所有历史小说文件"""
    novels = []
    if os.path.exists(NOVEL_FOLDER):
        for file in os.listdir(NOVEL_FOLDER):
            if file.endswith('.json'):
                novels.append(file.replace('.json', ''))
    return novels

def load_previous_chapter(chapter_number: str) -> str:
    """根据当前章节号加载上一章内容"""
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
                        except:
                            return ""
    return ""

def update_last_passage(chapter_number: str) -> str:
    """当章节号改变时更新上一章内容"""
    return load_previous_chapter(chapter_number)

def load_history_novel(history_file: str):
    """加载选中的历史小说文件内容"""
    if not history_file:
        return ""
    file_path = os.path.join(NOVEL_FOLDER, f"{history_file}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data.get('passage', '')
            except Exception as e:
                print(f"Error loading history novel: {e}")
                return ""
    return ""

def save_novel_with_confirm(passage, article_name, chapter_number, chapter_name, confirm_save):
    """带确认的保存函数, 并在成功后更新历史列表"""
    if not confirm_save:
        return "请先勾选确认保存选项", gr.update()
    try:
        novel_save(passage, article_name, chapter_number, chapter_name)
        message = "保存成功！"
    except Exception as e:
        message = f"保存失败: {e}"
    updated_choices = get_novel_history_list()
    return message, gr.update(choices=updated_choices)


def find_character_file(character_name: str) -> Dict:
    """根据角色名查找并加载对应的JSON文件内容"""
    if not character_name or not os.path.exists(CHARACTERS_FOLDER):
        return None
    
    for file in os.listdir(CHARACTERS_FOLDER):
        if file.endswith('.json'):
            file_path = os.path.join(CHARACTERS_FOLDER, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if data.get('name') == character_name:
                        return data
                except:
                    continue
    return None

def format_character_details_to_markdown(character_data: Dict) -> str:
    """将角色信息的字典格式化为Markdown字符串"""
    if not character_data:
        return ""
    
    md_string = f"### 角色卡: {character_data.get('name', '未知')}\n"
    for key, value in character_data.items():
        if key == 'name' or key == 'plot_history': # 跳过已在标题中显示的姓名和具体情节
            continue
        # 将键值对格式化为无序列表项
        md_string += f"- **{key}**: {value}\n"
    return md_string

def show_character_details(selected_characters: List[str]):
    """当角色下拉列表变化时，此函数被调用以显示角色详情"""
    if not selected_characters:
        # 如果没有选择任何角色，则隐藏详情面板
        return gr.update(visible=False), ""

    full_details_md = ""
    for char_name in selected_characters:
        character_data = find_character_file(char_name)
        if character_data:
            full_details_md += format_character_details_to_markdown(character_data)
            full_details_md += "\n---\n" # 为多个角色添加分隔线
    
    if full_details_md:
        # 如果找到了角色信息，则显示详情面板并更新内容
        return gr.update(visible=True), full_details_md
    else:
        # 如果选择的角色没有找到文件，也隐藏面板
        return gr.update(visible=False), ""

def create_novel_app():
    """创建并返回小说生成工具的Gradio应用"""
    with gr.Blocks(title="小说设计与生成工具") as demo:
        gr.Markdown("# 小说设计与生成工具")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 历史查询")
                history_dropdown = gr.Dropdown(
                    choices=get_novel_history_list(),
                    label="选择历史小说章节",
                    interactive=True
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
                save_output = gr.Textbox(label="保存状态", interactive=False)
            
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
                
                with gr.Accordion("出场角色详情", open=False, visible=False) as character_details_accordion:
                    character_details_display = gr.Markdown()
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
        
        # --- 事件处理 ---
        
        chapter_number_textbox.change(
            fn=update_last_passage, inputs=chapter_number_textbox, outputs=last_passage_textbox
        )
        history_btn.click(
            fn=load_history_novel, inputs=history_dropdown, outputs=passage_textbox
        )
        generate_btn.click(
            fn=writting_novel,
            inputs=[
                last_passage_textbox, this_passage_request_textbox, article_name_textbox,
                chapter_number_textbox, chapter_name_textbox, characters_dropdown,
                world_dropdown, styles_dropdown, words_slider, is_npc_checkbox
            ],
            outputs=passage_textbox
        )
        save_btn.click(
            fn=save_novel_with_confirm,
            inputs=[
                passage_textbox, article_name_textbox, chapter_number_textbox,
                chapter_name_textbox, confirm_save_checkbox
            ],
            outputs=[save_output, history_dropdown]
        )

        characters_dropdown.change(
            fn=show_character_details,
            inputs=characters_dropdown,
            outputs=[character_details_accordion, character_details_display]
        )
        return demo

if __name__ == "__main__":
    app = create_novel_app()
    app.launch()