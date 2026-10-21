# Auto Novel Prompt — AI 小说创作智能体

基于 DeepSeek 大模型与自研 **Plan-and-Execute** 编排框架的全流程 AI 小说创作 Agent：需求分析 → LLM 规划 → 依赖图执行（世界观 → 角色 → 大纲 → 章节）→ 质量门控，支持分层记忆、多轮续写、人工确认与全链路审计，并附带零依赖 Web 控制台。

## 特性

- **Plan-and-Execute 编排引擎**：需求经 LLM 拆解为可执行步骤，按依赖图逐节点执行；执行计划与产出物均支持人工确认门控（`plan_approval_required` / `artifact_approval_required`），失败自动修复重排（REPAIR/REVISE 状态机）。
- **M0-M5 六层记忆架构**：候选 / 规范 / 计划 / 确认 / 检索 / 工作记忆分层管理，基于 SQLite 内容寻址存储与版本化迁移；跨运行的作品资产自动复用——续写新章节时复用既有世界观 / 角色 / 大纲，已写章节自动跳过，不重复生成。
- **续写智能规划**：识别"续写第 N 章"类需求并确定性裁剪已存在资产的生成步骤；章节支持定向重写（revision + rebase 分支确认）。
- **质量门控流水线**：六维度 LLM 评审 + 篇幅指标自动评估，不达标自动限次修复，全程可人工介入。
- **意图识别 OOD 防御**：对闲聊 / 乱码等无关输入做意图识别拦截并返回明确失败原因；LLM 分析失败时自动降级启发式兜底，不静默产出垃圾。
- **调用治理与审计**：工具调用幂等化防重复执行；全链路调用轨迹、失败留档（`provider_failures`）、Token 消耗统计可回溯。
- **零依赖 Web 控制台**：内置 `http.server` 前端，多轮对话自动管理运行标识（run_id 内部化），支持运行历史、产物浏览、审计与一键导出。

## 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key（仅 `legacy_real` 真实模型需要；`agent_loop` 同理需要真实模型）

### 安装

```bash
git clone <repo-url>
cd auto_novel_prompt
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY，并确认超时与默认 provider
```

关键环境变量（详见 `.env.example`）：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_API_KEY_1` | DeepSeek 密钥（后者优先） | 无 |
| `NOVEL_AGENT_STAGE_TIMEOUT_SECONDS` | 单阶段超时；`legacy_real` 建议 7200 | 30 |
| `NOVEL_AGENT_PROVIDER_MODE` | 默认 provider：`legacy_real` / `agent_loop`（`offline_fake` 仅供测试，UI 不暴露） | `legacy_real` |

### 启动控制台

```bash
python webui.py            # 默认 http://127.0.0.1:7860
python webui.py --port 8000
```

### 使用流程

1. 选择项目（一本小说一个项目文件夹）
2. 输入创作需求（如"写一本魔法学院题材的小说，主角是孤儿，20 章"），点击启动
3. 系统自动完成需求分析 → 规划 → 世界/角色/大纲/章节创作（规划与产物可在控制台确认或重新生成）
4. 完成后输入"续写第二章"继续创作——自动复用既有设定，只写新章节
5. 点击"导出文件"将 SQLite 产物导出为 `worlds/`、`characters/`、`novel/` 目录下的文本/JSON 文件

### 运行测试

```bash
python -m pytest -q          # 全量（400+ 用例）
```

## 架构概览

```
用户需求 ──► 需求分析(BriefAnalyzer) ──► 规划器(PlanGenerator) ──► 计划确认
                                                              │
                                                              ▼
                       依赖图执行 TaskGraphExecutor（world → cast → outline → chapter×N）
                                                              │
                        记忆 M0-M5 │ 质量门控 │ 工具网关（幂等+审计）│ 人工确认/修复重排
                                                              ▼
                                            SQLite 权威存储（内容寻址 + 迁移）
                                                              │
                                            Web 控制台 / 导出（worlds/ characters/ novel/）
```

| 模块 | 说明 |
| --- | --- |
| `novel_agent/agent/` | 需求分析、规划器、任务图执行 |
| `novel_agent/domains/` | 小说领域节点（世界观/角色/大纲/章节）、资产复用检测 |
| `novel_agent/adapters/` | DeepSeek governed 适配器、旧版兼容适配器 |
| `novel_agent/run_*` | 运行状态机、调度器（单 Worker）、协调器、恢复 |
| `novel_agent/webui/` | 零依赖 Web 控制台（静态前端 + REST API） |
| `tests/` | 400+ 自动化测试 |

## 项目结构（节选）

```
auto_novel_prompt/
├── novel_agent/            # 核心包
│   ├── agent/              # 需求分析 / 规划 / 执行
│   ├── domains/            # 小说领域节点与提交
│   ├── adapters/           # 模型适配层
│   ├── webui/              # Web 控制台
│   └── migrations/         # SQLite 版本化迁移
├── tests/                  # 自动化测试
├── tools_call_deepseek.py  # DeepSeek 调用封装
├── webui.py                # 控制台入口
└── .env.example            # 环境变量模板
```

## 实现边界（预留 / 参考实现）

以下模块已实现并有测试，但当前运行路径不依赖它们，**不能**视为已交付的运行时特性：

- `novel_agent/provider.py::ProviderClient`、`RunCoordinator.record_provider_attempt/commit_task_result` 与 `provider_calls/task_results` 表：受管 provider 引擎的参考实现。当前 stage 通过 `LegacyDeepSeekTransport` 调用模型，调用轨迹记录在 `run_events`、`provider_failures` 与 `logs/token_usage.jsonl`。
- `novel_agent/legacy_import.py`：旧项目导入库，未暴露 CLI/HTTP 入口。
- `novel_agent/memory.py::MemoryGovernance`：候选晋升的参考实现；生产路径走 `memory_promotion` 与 `MemoryReviewService`。
- `adapters/governance.py::LegacyEvaluationAdapter` 及 `contracts` 中的 `ContextPack` / `AuditReplayManifest`：兼容/合同层预留。

已接线的可靠性能力：artifact outbox 由调度器在运行结束后物化到 `.governed/artifacts/`；连续性审计默认走 provider 结构化评审（`NOVEL_AGENT_CONTINUITY_AUDIT=offline` 可回退）；agent_loop 的章节质量门使用真实六维评审。

## 许可

本项目尚未指定开源许可证。对外发布前请先补充 LICENSE 文件。
