# 续写流程

用户说"续写 / 接着写 / 下一章 / 写到第 N 章"时走此流程。

## 步骤

1. **定位项目**：找到目标 `novels/<名字>/`。若项目不存在，先按 `create.md` 新建。
2. **读记忆与进度**：读 `project.json`（progress.written_chapters）+ `memory.json`（当前状态/未回收伏笔）。**不要整本重读** chapters/——只看最新一章的 `summary` 和 memory.json，控制 token 成本。
3. **确定下一章**：
   - 常规续写 → 下一章号 = 已写最大章号 + 1
   - 指定章号 → 写该章（若已写，询问用户是重写还是跳过）
4. **规划本章**：依 `outline.json` 取该章 `goal/beats`；若大纲没有（超纲），先补大纲再写。
5. **写正文** → `chapters/NN.json`（含 body/summary/characters/foreshadows）
6. **更新 memory.json**（角色变化、伏笔、已写章节号）
7. **质量审查**（六维）→ 写入 quality_log.json
8. **汇报**：本章写了什么、当前进度、下一步。

## 注意

- **一致性优先**：写前必须确认主角状态、未回收伏笔、世界规则——这些都在 memory.json。
- 若续写时发现前后矛盾，停下来修正已写内容（走 `revise.md`），不要带病往下写。
- 长篇小说建议每章控制在合理篇幅（如 2000-4000 字），避免单章过短/过长。
