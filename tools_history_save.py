from tools_call_deepseek import access_deepseek
import tools
from tools import CHARACTER_TEMPLATE, create_name_to_filename_mapping
import json
import os
import json_repair
import re
from typing import Dict, Optional, Any
from config import (
    HISTORY_PROMPT_PATH,
    DEFAULT_PROJECT,
    ensure_project_dirs,
    get_project_paths,
)


PROMPT = tools.read_txt_file(HISTORY_PROMPT_PATH)

CHARACTERS_MAP = """
角色名称：{character_name}
角色简介：{role_introduction}
角色性格：{role_personality}
角色背景：{role_experience}
"""


def normalize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[^\u4e00-\u9fff]', '', name)
    return name


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(c in it for c in needle)


def _overlap_ratio(str1: str, str2: str) -> float:
    overlap_set = set(str1) & set(str2)
    overlap_len = len(overlap_set)
    ratio1 = overlap_len / len(set(str1)) if set(str1) else 0
    ratio2 = overlap_len / len(set(str2)) if set(str2) else 0
    return max(ratio1, ratio2)


def find_matching_name(character_name: str, result_mapping: Dict[str, Any]) -> Optional[str]:
    normalized_input = normalize_name(character_name)
    similarity_matches = []

    for key in result_mapping.keys():
        normalized_key = normalize_name(key)
        if normalized_input == normalized_key:
            return key
        if _is_subsequence(normalized_input, normalized_key) or _is_subsequence(normalized_key, normalized_input):
            return key
        overlap_score = _overlap_ratio(normalized_input, normalized_key)
        similarity_matches.append((key, overlap_score))

    if not similarity_matches:
        return None
    similarity_matches.sort(key=lambda x: x[1], reverse=True)
    best_match = similarity_matches[0]
    if best_match[1] > 0.80:
        return best_match[0]
    return None


def tidy_paragraphs(text: str) -> str:
    text = re.sub(r'[\t \u3000]*\n[\t \u3000\n]*', '\n\n', text.strip())
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraphs = ['\u3000\u3000' + p for p in paragraphs]
    return '\n\n'.join(paragraphs)


def sort_key(d):
    filename = next(iter(d))
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


NEW_CHARACTER_PROMPT = """你是一位角色档案管理员。下面是一章小说内容，其中出现了一个新角色。

## 章节内容（节选）
{passage_sample}

## 章节摘要
{summary}

请在章节中找出角色「{name}」的相关信息，以 JSON 格式输出：
{{
    "name": "{name}",
    "role_introduction": "一句话简介（20-50字，说明角色身份和在故事中的定位）",
    "role_experience": "从本章内容推断的角色背景（30-100字）",
    "role_personality": "从本章言行推断的角色性格（10-30字）"
}}

只输出 JSON，不要其他内容。如果章节中确实没有足够信息推断，字段可以写一个简短的合理推测。"""


def _create_new_character(character_name, passage, summary,
                          characters_folder, file_name, character_history):
    prompt = NEW_CHARACTER_PROMPT.format(
        name=character_name,
        passage_sample=passage[:3000],
        summary=summary,
    )
    try:
        raw = access_deepseek(prompt, model_name="deepseek-v4-flash", max_retry=1,
                              caller="auto_create_character")
        data = json_repair.loads(raw)
    except Exception as e:
        print(f"  ❌ 无法生成角色卡 {character_name}: {e}")
        return False

    data.setdefault("name", character_name)
    if not isinstance(data.get("plot_history"), list):
        data["plot_history"] = []
    data["plot_history"].append({file_name: character_history})

    name_for_file = character_name.replace("/", "_")
    filepath = os.path.join(characters_folder, f"{name_for_file}.json")

    if os.path.exists(filepath):
        base = filepath.replace(".json", "")
        counter = 2
        while os.path.exists(f"{base}_{counter}.json"):
            counter += 1
        filepath = f"{base}_{counter}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"  ✅ 自动创建角色卡: {character_name} → {filepath}")
    return True


def novel_save(passage, article_name=None, chapter_number=None, chapter_name=None,
               this_passage_request="", project=DEFAULT_PROJECT):

    ensure_project_dirs(project)
    paths = get_project_paths(project)
    novel_folder = paths["novel"]
    characters_folder = paths["characters"]

    character_info_list = []
    for filename in os.listdir(characters_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(characters_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                character_info = CHARACTERS_MAP.format(
                    character_name=data.get("name", ""),
                    role_introduction=data.get("role_introduction", ""),
                    role_experience=data.get("role_experience", ""),
                    role_personality=data.get("role_personality", ""))
                character_info_list.append(character_info)

            except json.JSONDecodeError as e:
                print(f"无法解析JSON文件 {filename}: {e}")
            except IOError as e:
                print(f"处理文件 {filename} 时出错: {e}")

    character_info_list_str = json.dumps(character_info_list, ensure_ascii=False, indent=2)

    FINAL_PROMPT = PROMPT.format(passage=passage, character_info_list=character_info_list_str)
    result, tokens = access_deepseek(FINAL_PROMPT, caller="novel_save", return_tokens=True)
    print(f"[Token] 摘要提取 → prompt: {tokens['prompt_tokens']}, completion: {tokens['completion_tokens']}, total: {tokens['total_tokens']}")
    final_result = json_repair.loads(result)

    json_name = f"{chapter_number} {article_name} {chapter_name}.json"
    file_name = f"{article_name} {chapter_number} {chapter_name}"
    json_path = os.path.join(novel_folder, json_name)

    novel_passage = tidy_paragraphs(passage)
    novel_summary = final_result["summary"]
    novel = {
        "passage": novel_passage,
        "summary": novel_summary,
        "chapter_requirement": this_passage_request,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(novel, f, ensure_ascii=False, indent=4)

    novel_history = final_result["history"]
    result_mapping = create_name_to_filename_mapping(characters_folder)
    unmatched_characters = []

    for history in novel_history:
        character_name = history["character"]
        character_history = history["event"]

        matched_name = find_matching_name(character_name, result_mapping)
        if matched_name is None:
            unmatched_characters.append((character_name, character_history))
            continue

        character_json_name = result_mapping[matched_name] + ".json"
        character_dict = tools.load_json(os.path.join(characters_folder, character_json_name))

        if "plot_history" not in character_dict:
            character_dict["plot_history"] = []

        character_dict["plot_history"][:] = [d for d in character_dict["plot_history"] if file_name not in d]
        character_dict["plot_history"].append({file_name: character_history})
        character_dict["plot_history"] = sorted(character_dict["plot_history"], key=sort_key)
        tools.save_json(os.path.join(characters_folder, character_json_name), character_dict)

    for char_name, char_history in unmatched_characters:
        print(f"  🆕 检测到新角色: {char_name}，正在自动创建角色卡...")
        _create_new_character(char_name, passage, novel_summary,
                              characters_folder, file_name, char_history)

    return json_path


if __name__ == "__main__":
    passage = ""
    print(novel_save(passage, "入学篇", "63", "无用之人"))
