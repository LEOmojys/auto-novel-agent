from tools_call_deepseek import access_deepseek
import tools
from tools import (
    CHARACTER_TEMPLATE,
    create_name_to_filename_mapping,
    find_and_extract_summaries,
    extract_dicts_with_number,
)
import json
import os
from typing import Dict, Optional, Any
from config import (
    HELPER_PROMPT_PATH,
    NOVEL_STYLES_FOLDER,
    DEFAULT_PROJECT,
    ensure_project_dirs,
    get_project_paths,
)


TEMPLATE = CHARACTER_TEMPLATE


def writting_helper(last_passage: str,
                    this_passage_request: str,
                    article_name: str,
                    chapter_number: str,
                    chapter_name: str,
                    characters: list,
                    world: str,
                    styles: str,
                    words: int,
                    is_NPC: bool,
                    project: str = DEFAULT_PROJECT) -> str:

    ensure_project_dirs(project)
    paths = get_project_paths(project)

    history = find_and_extract_summaries(paths["novel"], int(chapter_number))
    history_str = json.dumps(history, ensure_ascii=False, indent=2)

    result_mapping = create_name_to_filename_mapping(paths["characters"])

    character_history_list = []
    for character in characters:
        character_json_name = result_mapping[character] + ".json"
        character_json_path = os.path.join(paths["characters"], character_json_name)
        character_info = tools.load_json(character_json_path)

        character_name = character_info["name"]
        role_introduction = character_info["role_introduction"]
        role_experience = character_info["role_experience"]
        role_personality = character_info["role_personality"]
        plot_history = character_info["plot_history"]

        plot_history = extract_dicts_with_number(plot_history, int(chapter_number))
        plot_history_str = json.dumps(plot_history, ensure_ascii=False, indent=2)

        character_history_list.append(TEMPLATE.format(
            character_name=character_name,
            role_introduction=role_introduction,
            role_experience=role_experience,
            role_personality=role_personality,
            plot_history=plot_history_str,
        ))

    character_history_list_str = json.dumps(character_history_list, ensure_ascii=False, indent=2)

    PROMPT = tools.read_txt_file(HELPER_PROMPT_PATH)
    FINAL_PROMPT = PROMPT.format(
        last_passage=last_passage,
        this_passage_request=this_passage_request,
        history=history_str,
        characters=character_history_list_str,
        world=world,
        styles=styles,
        words=words,
        is_NPC=is_NPC,
    )
    passage = access_deepseek(FINAL_PROMPT)

    return passage


if __name__ == "__main__":
    last_passage = ""
    this_passage_request = "没有别的什么要求，你来自由发挥。"
    article_name = "入学篇"
    chapter_number = "64"
    chapter_name = "知识补习"
    characters = ["艾尔维/艾尔", "贝拉", "欧斯", "伊万", "克罗埃", "卡伦"]
    world = tools.read_txt_file(os.path.join(NOVEL_WORLD_FOLDER, "常用世界观.txt"))
    styles = tools.read_txt_file(os.path.join(NOVEL_STYLES_FOLDER, "常用文风.txt"))
    words = 6000
    is_NPC = False

    print(writting_helper(
        last_passage, this_passage_request, article_name,
        chapter_number, chapter_name, characters,
        world, styles, words, is_NPC,
    ))
