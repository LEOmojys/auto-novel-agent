from api_v1 import create_novel_app

from novel_agent.webui.server import launch_console


__all__ = ("create_novel_app",)


if __name__ == "__main__":
    # 默认入口：新的零依赖 Web 控制台（python api.py）
    launch_console()
