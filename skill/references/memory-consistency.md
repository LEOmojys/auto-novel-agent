# 记忆与一致性维护

长篇小说最大的风险：**写着写着忘了前面的设定**（人设崩、伏笔丢、时间线乱）。
本 skill 用 `memory.json` 强制维护一致性。

## 铁律

1. **写新章前**：必读 `memory.json` + 最新一章的 `summary`
2. **写新章后**：必更新 `memory.json`
3. **绝不**在没查记忆的情况下写正文

## memory.json 维护内容

### 角色状态（characters 段）
- 新增能力 / 关系 / 当前状态
- 例：写第 4 章主角觉醒冰系 → 更新 `characters.林晚.abilities` 加 "冰系"

### 伏笔（plot_threads 段）
- 埋下：`status: open` + `last_mentioned: 本章号`
- 回收：`status: resolved`
- 例：第 1 章埋"银钥匙" → open；第 8 章揭开 → resolved

### 进度规则（rules 段）
- 已写章节范围、下一步该写什么
- 例：`"已写章节：1-3；下一章从 4 开始"`

## 写前 checklist（本章开写前逐项确认）

- [ ] 主角当前在哪、什么状态？（查 memory.characters）
- [ ] 有没有该回收的伏笔？（查 memory.plot_threads）
- [ ] 本章涉及哪些世界规则？（查 memory.world_rules_used + world.json）
- [ ] 上一章结尾停在哪？（查上一章 summary）

## 写后 checklist（本章落盘后）

- [ ] 出场角色档案更新了吗？
- [ ] 新埋/回收的伏笔登记了吗？
- [ ] 已写章节号 / 进度更新了吗？
- [ ] 有没有与前文矛盾的设定？有 → 立即修（revise.md）

## 成本控制（重要）

- 续写**不要整本重读**——只看 memory.json + 上一章 summary
- 一次续写最多回顾最近 1-2 章 + 记忆，控制 token
- 需要"很久以前"的细节时，从 memory.json 找，而不是翻全部章节
