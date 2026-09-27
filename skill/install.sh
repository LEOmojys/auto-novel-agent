#!/usr/bin/env bash
# novel-agent skill 安装脚本（Linux / macOS）
# 用法: ./install.sh
set -euo pipefail

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="novel-agent"

install_to() {
  local target="$1"
  mkdir -p "$target"
  # 复制全部内容，排除安装脚本自身
  (cd "$SOURCE" && tar cf - --exclude='install.sh' --exclude='install.ps1' .) | (cd "$target" && tar xf -)
  echo "已安装到: $target"
}

install_to "$HOME/.cache/opencode/skills/$NAME"
install_to "$HOME/.codex/skills/$NAME"

echo "完成。重新打开 opencode / codex 后，对话输入'写小说'即可触发。"
