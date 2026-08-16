import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


CHARACTER_TEMPLATE = """
角色名称：{character_name}
角色简介：{role_introduction}
角色性格：{role_personality}
角色背景：{role_experience}
近期剧情：{plot_history}
"""


def read_txt_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误: 文件不存在 [{file_path}]")
        return ""
    except IOError as e:
        print(f"读取文件时出错: {str(e)}")
        return ""


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: str, data: Any) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_names_from_json(data: Any) -> List[str]:
    names: List[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "name":
                names.append(value)
            else:
                names.extend(extract_names_from_json(value))
    elif isinstance(data, list):
        for item in data:
            names.extend(extract_names_from_json(item))
    return names


def create_name_to_filename_mapping(folder_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.exists(folder_path):
        return mapping
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            data = load_json(file_path)
            names = extract_names_from_json(data)
            base_filename = os.path.splitext(filename)[0]
            for name in names:
                mapping[name] = base_filename
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing file {filename}: {e}")
    return mapping


def find_and_extract_summaries(folder: str, n: int) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for file in Path(folder).glob("*.json"):
        stem = file.stem
        if not stem or not stem[0].isdigit():
            continue
        prefix_digits: List[str] = []
        for ch in stem:
            if ch.isdigit():
                prefix_digits.append(ch)
            else:
                break
        if not prefix_digits:
            continue
        num = int("".join(prefix_digits))
        if n - 9 <= num <= n:
            try:
                with file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                summary = data.get("summary")
                if summary is not None:
                    result[stem] = summary
            except (json.JSONDecodeError, IOError) as e:
                print(f"读取 {file} 出错: {e}")
    return result


def extract_dicts_with_number(lst: List[dict], target_number: int) -> List[dict]:
    result: List[dict] = []
    for item in lst:
        for key in item:
            match = re.search(r"\d+", key)
            if match:
                key_number = int(match.group())
                if key_number == target_number or (
                    key_number < target_number and target_number - key_number <= 10
                ):
                    result.append(item)
                    break
    return result
