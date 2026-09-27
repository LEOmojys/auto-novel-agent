# novel-agent skill 安装脚本（Windows PowerShell）
# 用法: powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"

$Source = Join-Path $PSScriptRoot "."   # skill/ 目录
$Name = "novel-agent"

$Targets = @(
    (Join-Path $env:USERPROFILE ".cache\opencode\skills\$Name"),
    (Join-Path $env:USERPROFILE ".codex\skills\$Name")
)

foreach ($target in $Targets) {
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    # 复制全部内容（SKILL.md / manifest.yaml / commands / references / schemas / templates）
    Get-ChildItem -Path $Source -Force | Where-Object { $_.Name -notmatch "^install\.(ps1|sh)$" } | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $target -Recurse -Force
    }
    Write-Host "已安装到: $target"
}

Write-Host "完成。重新打开 opencode / codex 后，对话输入'写小说'即可触发。"
