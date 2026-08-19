"""诊断脚本：查 run 的真实状态、最新事件、失败记录、产物。

用法：
  python scripts/diagnose_run.py <项目目录> <run_id>
  python scripts/diagnose_run.py --find <run_id>      # 自动在所有项目里找
例如：
  python scripts/diagnose_run.py data/projects/女主角手 tst-4
  python scripts/diagnose_run.py --find tst-4
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def diagnose(db_path: str, run_id: str) -> None:
    conn = sqlite3.connect(db_path)
    print(f"=== 数据库: {db_path} ===")

    print("\n=== run 状态 ===")
    row = conn.execute("SELECT run_id, state, created_at, updated_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        print(f"run {run_id} 不存在")
        conn.close()
        return
    print(row)

    print("\n=== 最新 10 个事件 ===")
    for r in conn.execute(
        "SELECT sequence, event_type, payload_json, created_at FROM run_events WHERE run_id=? ORDER BY sequence DESC LIMIT 10",
        (run_id,),
    ):
        print(f"#{r[0]} {r[3]} {r[1]} {r[2][:100]}")

    print("\n=== 关键事件（失败/阻塞/等待确认/完成）===")
    for r in conn.execute(
        "SELECT event_type, payload_json FROM run_events WHERE run_id=? AND event_type IN "
        "('run_failed','plan_blocked','artifact_approval_required','run_succeeded','chapter_rebase_proposed') ORDER BY sequence",
        (run_id,),
    ):
        print(r[0], r[1][:120])

    print("\n=== provider_failures（LLM 调用失败记录）===")
    for r in conn.execute(
        "SELECT caller, failure_kind, reason, created_at FROM provider_failures WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ):
        print(r)

    print("\n=== artifacts（全部状态）===")
    for r in conn.execute(
        "SELECT artifact_id, revision, kind, status, length(raw_bytes) FROM artifacts WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ):
        print(r)

    print("\n=== tool_calls（规划相关）===")
    for r in conn.execute(
        "SELECT tool_id, state, created_at FROM tool_calls WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ):
        print(r)

    print("\n=== run_leases ===")
    for r in conn.execute("SELECT run_id, holder_id, fencing_token, expires_at FROM run_leases WHERE run_id=?", (run_id,)):
        print(r)
    conn.close()


def find_and_diagnose(projects_root: str, run_id: str) -> None:
    root = Path(projects_root)
    found = False
    for project_dir in sorted(root.iterdir()):
        db = project_dir / "agent.sqlite"
        if not db.is_file():
            continue
        try:
            conn = sqlite3.connect(str(db))
            hit = conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
            conn.close()
        except sqlite3.Error:
            continue
        if hit:
            found = True
            diagnose(str(db), run_id)
    if not found:
        print(f"在所有项目（{root}）中未找到 run: {run_id}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "--find":
        find_and_diagnose(args[1], args[2])
    elif len(args) >= 2 and args[0] == "--find":
        import config

        find_and_diagnose(config.PROJECTS_DIR, args[1])
    elif len(args) >= 2:
        diagnose(args[0], args[1])
    else:
        print(__doc__)
