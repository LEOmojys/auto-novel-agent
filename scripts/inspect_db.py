import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB = Path("data/projects/test/agent.sqlite")
print("数据库存在:", DB.exists(), "| 大小:", DB.stat().st_size if DB.exists() else 0, "bytes")
if not DB.exists():
    sys.exit(0)

conn = sqlite3.connect(DB)

print("\n=== runs ===")
for row in conn.execute("SELECT run_id, project_id, state, plan_revision, created_at, updated_at FROM runs"):
    print(" ", row)

print("\n=== artifacts（内容产物）===")
rows = conn.execute(
    "SELECT artifact_id, kind, status, is_current, length(raw_bytes), sha256 FROM artifacts ORDER BY created_at"
).fetchall()
for r in rows:
    print(f"  {r[0]:45s} kind={r[1]:15s} status={r[2]:9s} current={r[3]} size={r[4]}")

print("\n=== 章节正文预览 ===")
for artifact_id, in conn.execute(
    "SELECT artifact_id FROM artifacts WHERE kind = 'chapter' AND status = 'accepted' ORDER BY artifact_id"
):
    raw = conn.execute(
        "SELECT raw_bytes FROM artifacts WHERE artifact_id = ? AND status = 'accepted' ORDER BY revision DESC LIMIT 1",
        (artifact_id,),
    ).fetchone()
    if raw is None:
        continue
    text = bytes(raw[0]).decode("utf-8", errors="replace")
    print(f"--- {artifact_id} ({len(text)} 字符) ---")
    print(text[:300])
    print()

print("=== memory_records（记忆）===")
for row in conn.execute(
    "SELECT layer, kind, subject_id, substr(value_json, 1, 80) FROM memory_records ORDER BY created_at"
):
    print(" ", row)

print("\n=== tool_calls（调用审计，判断真实管线）===")
for row in conn.execute("SELECT tool_id, state, task_id FROM tool_calls ORDER BY created_at LIMIT 15"):
    print(" ", row)

print("\n=== provider/token 痕迹 ===")
for row in conn.execute("SELECT COUNT(*) FROM provider_calls"):
    print("  provider_calls:", row[0])
conn.close()
