import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

PROJECTS_DIR = os.path.join(BASE_DIR, "data", "projects")
DEFAULT_PROJECT = "default"

HISTORY_PROMPT_PATH = os.path.join(BASE_DIR, "prompt", "history_prompt", "history_prompt_v1.txt")
HELPER_PROMPT_PATH = os.path.join(BASE_DIR, "prompt", "helper_prompt", "helper_prompt_v1.txt")
GERERATE_PROMPT_PATH = os.path.join(BASE_DIR, "prompt", "generate_prompt", "prompt_v6.txt")
NOVEL_STYLES_FOLDER = os.path.join(BASE_DIR, "data", "styles")


def get_project_paths(project_name: str = DEFAULT_PROJECT) -> dict:
    root = os.path.join(PROJECTS_DIR, project_name)
    return {
        "novel": os.path.join(root, "novel"),
        "characters": os.path.join(root, "characters"),
        "worlds": os.path.join(root, "worlds"),
        "evaluate": os.path.join(root, "evaluate"),
    }


def get_project_list() -> list:
    if not os.path.exists(PROJECTS_DIR):
        return [DEFAULT_PROJECT]
    projects = [
        d for d in os.listdir(PROJECTS_DIR)
        if os.path.isdir(os.path.join(PROJECTS_DIR, d))
    ]
    projects.sort()
    if not projects:
        return [DEFAULT_PROJECT]
    if DEFAULT_PROJECT not in projects:
        projects.insert(0, DEFAULT_PROJECT)
    return projects


def ensure_project_dirs(project_name: str = DEFAULT_PROJECT):
    paths = get_project_paths(project_name)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


def current_novel_folder(project: str = DEFAULT_PROJECT) -> str:
    return get_project_paths(project)["novel"]


def current_characters_folder(project: str = DEFAULT_PROJECT) -> str:
    return get_project_paths(project)["characters"]


def current_worlds_folder(project: str = DEFAULT_PROJECT) -> str:
    return get_project_paths(project)["worlds"]


NOVEL_FOLDER = current_novel_folder()
CHARACTERS_FOLDER = current_characters_folder()
NOVEL_WORLD_FOLDER = current_worlds_folder()
