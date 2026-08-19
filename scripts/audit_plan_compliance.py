from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TASK_RE = re.compile(r"^- \[x\] (\d+)\. ", re.MULTILINE)
RECEIPT_RE = re.compile(r"(?:python|node|Get-Content|curl|pytest|npm|git) .{0,240}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    plan_text = args.plan.read_text(encoding="utf-8")
    task_ids = [int(value) for value in TASK_RE.findall(plan_text)]
    if task_ids != list(range(1, 18)):
        raise SystemExit(f"expected checked implementation tasks 1..17, found {task_ids}")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for task_id in task_ids:
        receipts = sorted(args.evidence.glob(f"task-{task_id}-*"))
        receipts = [path for path in receipts if path.is_file() and path.stat().st_size > 0]
        commands: list[str] = []
        for receipt in receipts:
            text = receipt.read_text(encoding="utf-8", errors="replace")
            commands.extend(RECEIPT_RE.findall(text))
        row = {
            "task": task_id,
            "evidence": [str(path) for path in receipts],
            "non_empty": bool(receipts),
            "named_commands": len(commands),
        }
        rows.append(row)
        if not receipts or not commands:
            missing.append(str(task_id))

    report = {"plan": str(args.plan), "tasks": rows, "missing": missing}
    output = args.evidence / "f1-plan-compliance.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if missing:
        raise SystemExit(f"missing evidence or commands for task(s): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
