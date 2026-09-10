# 小说 Agent → opencode 式 Agent Loop 改造方案

> 目标：保持小说创作领域能力不变，将"固定 Plan-and-Execute DAG"替换为
> "模型每步自决的工具调用循环"（opencode/codex 工作模式）。
> 本文档确定各核心功能点的实现边界，作为落地蓝图。

## 0. 现状与目标对照

| 维度 | 现状（固定 DAG） | 目标（Agent Loop） |
|---|---|---|
| 执行结构 | `TaskGraphExecutor` 按预定义图跑 world→cast→outline→chapter | `AgentLoop` 循环：模型每步选工具+参数，观察结果，直到 `done` |
| 规划 | LLM 从 5 个预定义工具选步骤顺序（可裁剪） | 模型自由组合任意工具，可跳过/重试/乱序（软约束在工具描述） |
| 决策权 | 节点执行器确定性驱动 | 模型自决 |
| 终止 | 图跑完 = 结束 | 模型 `done` / max_steps / 人工确认门控 |
| 失败处理 | 节点 block → 等 REPAIR | 工具报错回喂 → 模型换策略重试 |

**原则**：领域能力（生成/评审/记忆提交/审批/审计）全部保留；
只替换"谁来决定下一步"这一层。

---

## 1. Agent 核心功能点 — 实现边界

### 1.1 上下文管理（Agent Context）

**现状**（`novel_agent/context.py`）：
- `ContextBuilder.build(ContextRequest) → ContextBuildResult`：加载 m1/m2/m3 记忆行 →
  排除（层/状态/超期/质量意见/溯源校验）→ 排序（tier/relevance/narrative_time）→
  按类别预算（canonical/plan/continuity/long_term/style）截断 → 拼 prompt + digest。
- 它服务于**单章执行**（`ContextRequest` 含 `chapter_plan_id/chapter_number/entity_ids`）。
- 预算模型：`ContextBudget`（total 4096 + 5 类子预算）。

**边界**：
- **保留**：记忆质量校验（溯源哈希/跨度）、类别预算截断、digest 审计——原样用于
  `write_chapter`/`write_world` 等**领域工具内部**（工具需要正文上下文时调它）。
- **新增**：`AgentLoopContext`（每步决策上下文），区别于领域上下文：
  ```
  @dataclass AgentLoopContext:
      system_prompt: str          # 角色 + 工具清单 + 工作规则
      history: list[Message]      # 用户任务 + 之前每轮 [工具调用 + 观察]
      memory_snapshot: str        # 精简的已有设定摘要（detect_existing_assets）
      recent_artifacts: list[str] # 最近产物摘要（标题/章号/评分）
  ```
- **实现边界**：`AgentLoopContext` 由 `build_loop_context(history, tools, existing_assets)`
  构造；**领域上下文**（`ContextBuilder`）只在工具内部按需构建。两者互不混用——
  loop 层看"做了什么"，工具层看"写什么"。

### 1.2 工具调用（Tool Calling）

**现状**（`novel_agent/tool_registry.py` + `workflow_tools.py`）——**已完备，直接复用**：
- `ToolRegistry.register(ToolSpec, handler)` / `resolve(tool_ref)`
- `ToolGateway.invoke(ToolInvocationRequest) → ToolCallResult`：
  - **幂等**：`input_hash = sha256(input_data.model_dump_json())`，
    `reserve_result`（进行中去重）→ `persist_result`（成功落库），同参重复调用返回缓存
  - **审计**：`traces()`（InvocationTrace）
  - **安全**：UNKNOWN_TOOL / SHADOW_NO_SIDE_EFFECT / IDEMPOTENCY_CONFLICT 决策
- `register_workflow_tools(...)` 已把领域能力注册为工具（world.generate 等）。

**边界**：
- **保留**：`ToolRegistry`/`ToolGateway` 全部机制（幂等、审计、安全）。
- **调整**：新增 `novel_agent/novel_tools.py`，把领域节点包成 **loop 工具**（在现有
  `register_workflow_tools` 之上加一层面向模型的描述）：
  | 工具名 | 底层 | 说明 |
  |---|---|---|
  | `write_world` | `_world` 节点 | 生成/修订世界观 |
  | `write_cast` | `_cast` 节点 | 生成/重做角色（含 revise） |
  | `write_outline` | `_outline` 节点 | 生成大纲 |
  | `write_chapter` | `_chapter` 节点 | 写/重写一章（参数 chapter_number） |
  | `review_chapter` | 质量评审器 | 六维度评分，返回达标/修复建议 |
  | `mem_query` | M4 检索 | 查询已确认设定（按相关性返回） |
  | `finish` | — | 模型声明任务完成（`{"done": true}`） |
- **新增字段**：每个工具补 `description`（模型可读）+ `input_schema`（JSON Schema）+
  `requires_approval`（复用 waiting_review 门控）。

### 1.3 记忆系统（Memory）

**现状**（`novel_agent/memory.py` + `memory_store.py`）：
- M0 candidate → M1 canonical / M2 plan / M3 confirmed（可注入上下文）→ M4 retrieval（检索层）
  → M5 working（工作层）。
- 提交路径：章节完成 → 提取候选（M0）→ 质量门 → 提升 M3 → 供后续上下文注入。
- `detect_existing_assets` 项目级复用（跨 run）。

**边界**：
- **保留**：M0-M3 完整链路（写入/提升/溯源校验）、`detect_existing_assets`。
- **调整**：
  - M4 检索从"隐式注入"改为**显式工具** `mem_query`（模型主动查设定，避免长上下文塞满）。
  - M5 working 作为 loop 的**工作暂存**：模型中间产物（如"我打算先写角色"）不落 canonical，
    只在当前 run 内可见；run 结束清理（requeue purge 已覆盖 m5）。
  - `build_loop_context` 的记忆快照 = `existing_assets_summary` 精简版（世界/角色/大纲/已写章）。

### 1.4 模型决策与审批（Model Decision & Approval）

**现状**：
- 决策解析：`_parse_analysis`/`_parse_plan`（strict → 去围栏 → json_repair 三级）。
- 审批门控：`plan_approval_required` / `artifact_approval_required` / `chapter_rebase_proposed`
  → run 停 `waiting_review` → REPAIR/REVISE 重排。
- 状态机：SUCCEEDED/FAILED/CANCELLED → requeue；WAITING_REVIEW → REPAIR。

**边界**（已对照 opencode/codex 修正）：
- **保留**：三级解析、审批事件、状态机、REPAIR/REVISE、requeue。
- **新增**：`AgentLoop.model_step(prompt) → AgentAction`
  ```python
  @dataclass AgentAction:
      kind: Literal["tool", "done"]
      tool: str | None
      args: dict | None
  ```
  **终止对齐共识**：模型输出无 tool_call 即视为完成（`done` 是 harness 判定，不是模型显式
  输出）；解析失败 → 重试 1 次（模型层退避，复用 provider_failures）→ 仍失败则强制结束并
  保留已达成的产物。
- **审批语义**：`requires_approval=True` 的工具调用 → staged 后 `waiting_review`（复用现有
  `artifact_approval_required`）；approve → **从断点续跑**（见风险 R5.1，loop 不重头）；
  拒绝 → 带 reason 回喂模型（对齐 codex `Reject{reason}` / opencode `RejectedError`）。
- **终止条件**：模型无 tool_call（首选）→ `agent.steps` 上限（最后一步注入合成消息逼收尾，
  对齐 opencode）→ provider 连续失败 → 人工 `waiting_review` 打断。
- **循环检测**（对齐 opencode doom-loop）：连续 N 次(同 tool, 同 input_hash) →
  触发审批询问（选继续/换策略/终止），不静默熔断。

---

## 2. 文件级改动清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `novel_agent/agent_loop.py` | **新增** | `AgentLoop`、`AgentAction`、`build_loop_context`、`AgentLoopContext`（~180 行） |
| `novel_agent/novel_tools.py` | **新增** | `build_novel_loop_tools(plugin, gateway, ...) → dict[str, AgentTool]`（~130 行，包已有节点） |
| `novel_agent/default_stage.py` | **改** | 新增 `_AgentLoopStage`（与 `_LegacyStage` 平级）；`default_stage_executor_factory` 增加 provider 分支 `"agent_loop"`（~90 行） |
| `novel_agent/service_models.py` | **改** | `provider` Literal 增加 `"agent_loop"` |
| `novel_agent/context.py` | **改** | 新增 `build_loop_context` 辅助（复用 `_load_rows` 之外的精简快照；`ContextBuilder` 本体不动） |
| `novel_agent/webui/static/app.js` | **改** | 事件渲染：`tool_call:<name>` 显示为"调用工具：写章节"（PLAN_TOOL_LABEL 扩展） |
| `novel_agent/workflow_view.py` | **改** | phase 映射增加 `agent_loop`（"Agent 思考中/正在调用 …"） |
| `tests/test_agent_loop.py` | **新增** | 见 §4 |
| `docs/agent-loop-migration.md` | 本文 | 蓝图 |

**不动**：`run_coordinator`/`run_worker`/`local_run_scheduler`/`run_recovery`（状态机与调度）、
`memory.py`/`memory_store.py`/`memory_models.py`（记忆写入链）、
`tool_registry.py`/`tool_call_ledger.py`/`tool_call_persistence.py`（工具网关）、
`export.py`/`webui/server.py`、全部 `_LegacyStage`/`_OfflineStage` 路径（保留兼容）。

---

## 3. 数据流（目标态）

```
用户需求 ──► BriefAnalyzer（复用）──► detect_existing_assets（复用）
                                        │
                                        ▼
                    build_loop_context（任务 + 历史 + 设定快照 + 工具清单）
                                        │  ▼
                    AgentLoop.run ──► model_step ──► AgentAction
                                        │  ├─ tool: gateway.invoke(工具)（幂等+审计）
                                        │  │     └─ 工具内部：ContextBuilder(领域上下文) + 记忆提交 + 质量门 + 审批
                                        │  │     └─ append_run_event("tool_call:write_chapter") + 观察回喂
                                        │  └─ done / max_steps / 审批打断
                                        ▼
                    _finish_run_if_pending / waiting_review / fail
```

工具调用记录、事件流、Token 审计、审批、导出——全部沿用现有机制。

---

## 4. 测试策略

1. **单元（tests/test_agent_loop.py）**：
   - `model_step` 解析：合法工具 / `done` / 解析失败重试 / 连续失败保底
   - 终止：done / max_steps / 工具报错 N 次
   - 上下文：历史追加、记忆快照注入、预算截断不溢出
   - 审批：`requires_approval` 工具 → waiting_review → approve 续跑
2. **端到端**：可控 transport 脚本化模型行为——"写大纲 → 评审不达标 → 改大纲 →
   再评审 → 写章节 → done"；断言事件序列（tool_call 顺序）与产物齐全。
3. **回归**：现有 367 测试零改动（`_LegacyStage`/`_OfflineStage` 保留为默认入口；
   `agent_loop` 为新 provider，互不影响）。

---

## 5. 实施阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | `agent_loop.py` + `novel_tools.py` 骨架（工具注册/循环/终止/解析） | 单元测试绿 |
| P1 | `_AgentLoopStage` 装配 + provider 分支 + 审批/记忆接入 | offline transport 端到端跑通 |
| P2 | 前端/视图（tool_call 事件、phase 映射） | webui 可见工具调用轨迹 |
| P3 | 真实模型调优（工具描述、上下文快照、max_steps） | legacy_real 续写跑通 |
| P4 | 全量回归 + 文档 | 367+ 测试绿 |

**估算**：P0-P2 约 500-600 行新增，0 破坏；P3 需真实模型迭代。

---

## 6. 风险与对策

> 完整风险清单见 `docs/agent-loop-risks.md`（v1.0，已对照 opencode/codex 源码核实）。
> 此处列实施阻断项。

| 风险 | 对策 |
|---|---|
| 模型循环不收敛（反复改一章） | doom-loop 检测（连续 N 次同 tool+同参 → 转审批询问）；review 提供"达标即停"信号；agent.steps 上限注入合成消息逼收尾 |
| 上下文膨胀（历史无限增长） | 阈值触发压缩（90%/20k buffer）→ head/tail 切分 → 结构化"续跑提示"摘要（不重摘要）；滚动窗口保最近 K 轮；工具结果截断+落盘留路径 |
| 审批打断循环 | **断点续跑**（waiting_review 持久化 loop 状态，approve 后从断点继续，不重头）——与 DAG 最大差异 |
| 审批拒绝 | 拒绝原因回喂模型（"改设定库需审批被拒"），模型换路径；批量拒同 session 其余 pending |
| Prompt 注入（素材夹带指令） | 数据/指令分离（内容作数据块引用）；工具结果定界符+声明非指令；写类工具强制审批 |
| 模型过早声明完成 | 终止时校验关键产物存在（brief 要求章节则须 ≥1 accepted），缺失注入纠正（≤2 次） |
| 崩溃/中断恢复 | 消息 write-ahead 落库 + loop_step 事件重放 + checkpoint；中断半成品标记非丢弃 |
| 工具大输出 | 工具结果 summary 上限 + 截断落盘留路径，模型可再取 |
| 并行工具调用 | 读并行、写串行（工具注册声明 supports_parallel；写类读写锁） |
| Token 成本 | 无硬熔断；按"章节=task、卷=thread"切分 + 每 task 落 token/cost 快照；超预算→压缩而非中止 |

---

*状态：方案已定稿。实施入口 = P0（agent_loop.py + novel_tools.py 骨架）。*
