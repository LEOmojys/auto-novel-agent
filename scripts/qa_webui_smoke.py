# -*- coding: utf-8 -*-
"""QA webui HTTP smoke (temporary). Uses create_console_server on an isolated
projects dir; no real project data touched."""
from __future__ import annotations

import http.client
import json
import shutil
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


class Client:
    def __init__(self, port: int):
        self.port = port

    def get(self, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            return resp.status, body
        finally:
            conn.close()

    def post(self, path: str, payload: dict):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            return resp.status, body
        finally:
            conn.close()


def main() -> int:
    import config
    from novel_agent.service import NovelAgentService
    from novel_agent.webui.server import create_console_server

    root = Path(tempfile.mkdtemp(prefix="qa-webui-"))
    projects_root = root / "projects"
    (projects_root / "novel-one").mkdir(parents=True)
    config.PROJECTS_DIR = str(projects_root)

    service = NovelAgentService.from_config(now=lambda: FIXED_TIME)
    server = create_console_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(server.server_port)
    try:
        # GET /api/projects
        try:
            status, body = client.get("/api/projects")
            data = json.loads(body)
            assert status == 200 and "novel-one" in data["projects"], (status, data)
            record("W1 GET /api/projects", True, f"projects={data['projects']}")
        except Exception as e:
            record("W1 GET /api/projects", False, f"{type(e).__name__}: {e}")

        # GET /api/runs
        try:
            status, body = client.get("/api/runs")
            data = json.loads(body)
            assert status == 200 and isinstance(data["runs"], list), (status, data)
            record("W2 GET /api/runs", True, f"runs(list len={len(data['runs'])})")
        except Exception as e:
            record("W2 GET /api/runs", False, f"{type(e).__name__}: {e}")

        # GET / index.html
        try:
            status, body = client.get("/")
            assert status == 200 and "Agent 小说创作控制台" in body, status
            record("W3 GET / static index.html", True, "title present")
        except Exception as e:
            record("W3 GET / static index.html", False, f"{type(e).__name__}: {e}")

        # unknown route 404
        try:
            status, body = client.get("/api/nope")
            assert status == 404, status
            record("W4 unknown route -> 404", True, f"status={status}")
        except Exception as e:
            record("W4 unknown route -> 404", False, f"{type(e).__name__}: {e}")

        # GET /api/health (no key)
        try:
            status, body = client.get("/api/health")
            data = json.loads(body)
            assert status == 200 and data["ok"] is False and data["reason"] == "missing_key", data
            record("W5 GET /api/health missing key", True, f"reason={data['reason']}")
        except Exception as e:
            record("W5 GET /api/health missing key", False, f"{type(e).__name__}: {e}")

        # POST /api/run/start explicit run_id offline_fake -> poll -> succeeded
        try:
            status, body = client.post("/api/run/start", {
                "project": "novel-one", "run_id": "webui-qa-1",
                "requirements": "写一本一章的小说", "provider": "offline_fake",
                "chapter_count": 1, "auto_approve": True,
            })
            started = json.loads(body)
            assert status == 200 and started["status"] == "queued", (status, started)
            deadline = time.monotonic() + 30
            snap = None
            while True:
                st, b = client.get("/api/run/snapshot?project=novel-one&run_id=webui-qa-1")
                assert st == 200
                snap = json.loads(b)
                if snap["status"] in ("succeeded", "failed", "waiting_review", "cancelled"):
                    break
                assert time.monotonic() < deadline, f"stuck at {snap['status']}"
                time.sleep(0.1)
            assert snap["status"] == "succeeded", snap
            assert snap["event_count"] >= 5, snap
            assert any(e["event_type"] == "run_succeeded" for e in snap["events"]), "no run_succeeded event"
            record("W6 POST /api/run/start + poll -> succeeded", True,
                   f"status={snap['status']} events={snap['event_count']}")
        except Exception as e:
            record("W6 POST /api/run/start + poll -> succeeded", False, f"{type(e).__name__}: {e}")

        # workflow + artifacts + audit endpoints
        try:
            st, b = client.get("/api/run/workflow?project=novel-one&run_id=webui-qa-1")
            wf = json.loads(b)
            assert st == 200 and wf["phase"] == "finished", (st, wf)
            assert [n["node_id"] for n in wf["nodes"]] == ["world", "cast", "outline", "chapter_1"], wf
            record("W7 GET /api/run/workflow", True, f"phase={wf['phase']} nodes={[n['node_id'] for n in wf['nodes']]}")
        except Exception as e:
            record("W7 GET /api/run/workflow", False, f"{type(e).__name__}: {e}")

        try:
            st, b = client.get("/api/run/artifacts?project=novel-one&run_id=webui-qa-1")
            arts = json.loads(b)
            kinds = {a["kind"] for a in arts["artifacts"]}
            assert st == 200 and {"world_bible", "character_set", "plan_contract", "chapter"} <= kinds, kinds
            record("W8 GET /api/run/artifacts", True, f"kinds={sorted(kinds)}")
        except Exception as e:
            record("W8 GET /api/run/artifacts", False, f"{type(e).__name__}: {e}")

        try:
            st, b = client.get("/api/run/audit?project=novel-one&run_id=webui-qa-1")
            audit = json.loads(b)
            assert st == 200 and isinstance(audit["manifests"], list) and isinstance(audit["quality"], list), (st, audit)
            record("W9 GET /api/run/audit", True, f"manifests={len(audit['manifests'])} quality={len(audit['quality'])}")
        except Exception as e:
            record("W9 GET /api/run/audit", False, f"{type(e).__name__}: {e}")

        # POST /api/run/start with empty requirements -> error response
        try:
            status, body = client.post("/api/run/start", {
                "project": "novel-one", "run_id": "webui-qa-blank",
                "requirements": "", "provider": "offline_fake",
            })
            assert status != 200, (status, body)
            record("W10 POST /api/run/start empty requirements -> error", True, f"status={status} body={body[:200]}")
        except Exception as e:
            record("W10 POST /api/run/start empty requirements -> error", False, f"{type(e).__name__}: {e}")

        # POST /api/run/start missing requirements key -> error
        try:
            status, body = client.post("/api/run/start", {
                "project": "novel-one", "run_id": "webui-qa-missing",
                "provider": "offline_fake",
            })
            assert status != 200, (status, body)
            record("W11 POST /api/run/start missing requirements key -> error", True, f"status={status} body={body[:200]}")
        except Exception as e:
            record("W11 POST /api/run/start missing requirements key -> error", False, f"{type(e).__name__}: {e}")

        # POST /api/run/start WITHOUT run_id (multi-turn) -> auto-assigned
        try:
            status, body = client.post("/api/run/start", {
                "project": "novel-one", "requirements": "写一本一章的科幻小说",
                "provider": "offline_fake", "chapter_count": 1,
            })
            started = json.loads(body)
            # Explicit run_id "webui-qa-1" is not in the {project}-run-N scheme,
            # so auto-allocation starts at run-1.
            assert status == 200 and started["run_id"] == "novel-one-run-1", (status, started)
            auto_run_id = started["run_id"]
            record("W12 POST /api/run/start multi-turn run_id auto-assign", True, f"run_id={started['run_id']}")
        except Exception as e:
            auto_run_id = None
            record("W12 POST /api/run/start multi-turn run_id auto-assign", False, f"{type(e).__name__}: {e}")

        # POST /api/export -> files; verify worlds/characters/novel written
        try:
            deadline = time.monotonic() + 30
            while True:
                st, b = client.get(f"/api/run/snapshot?project=novel-one&run_id={auto_run_id}")
                snap = json.loads(b)
                if snap.get("status") in ("succeeded", "failed", "cancelled", "waiting_review"):
                    break
                assert time.monotonic() < deadline, f"stuck: {snap}"
                time.sleep(0.1)
            status, body = client.post("/api/export", {"project": "novel-one"})
            data = json.loads(body)
            assert status == 200 and isinstance(data["files"], list), (status, data)
            pdir = projects_root / "novel-one"
            world_files = list((pdir / "worlds").glob("*.txt"))
            char_files = list((pdir / "characters").glob("*.json"))
            novel_files = list((pdir / "novel").glob("*.json"))
            assert world_files and char_files and novel_files, (world_files, char_files, novel_files)
            record("W13 POST /api/export writes worlds/characters/novel", True,
                   f"files={len(data['files'])} worlds={len(world_files)} chars={len(char_files)} novel={len(novel_files)}")
        except Exception as e:
            record("W13 POST /api/export writes worlds/characters/novel", False, f"{type(e).__name__}: {e}")

        # GET /api/runs history now contains both runs
        try:
            status, body = client.get("/api/runs")
            runs = json.loads(body)["runs"]
            run_ids = {r["run_id"] for r in runs}
            assert "webui-qa-1" in run_ids and auto_run_id in run_ids, run_ids
            record("W14 GET /api/runs history lists runs", True, f"run_ids={sorted(run_ids)}")
        except Exception as e:
            record("W14 GET /api/runs history lists runs", False, f"{type(e).__name__}: {e}")

    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(root, ignore_errors=True)

    print("\n==== SUMMARY ====")
    failed = [r for r in RESULTS if not r[1]]
    print(f"total={len(RESULTS)} passed={len(RESULTS) - len(failed)} failed={len(failed)}")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
