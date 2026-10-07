"""Runs Support Desk against evals/cases.jsonl with the real model and scores each case.

    OPENAI_API_KEY=... python evals/run.py [--out evals/results.json]

Each case starts from a fresh copy of the demo orders. Approvals are answered from the case's
"approve" field, so the approval step is exercised without a person at the keyboard.
"""
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path

from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent as support  # noqa: E402

CASES = Path(__file__).with_name("cases.jsonl")


def run_case(case: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        orders = Path(tmp) / "orders.json"
        shutil.copy(support.DATA / "orders.json", orders)
        support.ORDERS_FILE = orders
        config = {"configurable": {"thread_id": case["id"] + str(time.time())}, "recursion_limit": support.MAX_STEPS}
        out = support.agent.invoke({"messages": [("user", case["message"])]}, config)
        asked = False
        while out.get("__interrupt__"):
            asked = True
            out = support.agent.invoke(Command(resume="approve" if case.get("approve") else "decline"), config)
        tools = [c["name"] for m in out["messages"] for c in (getattr(m, "tool_calls", None) or [])]
        reply = out["messages"][-1].content
        refunded = {k: v["refunded"] for k, v in json.loads(orders.read_text()).items()}
    e, checks = case["expect"], {}
    if "tools" in e: checks["used_expected_tools"] = all(t in tools for t in e["tools"])
    if "reply_mentions" in e: checks["reply_mentions"] = all(s.lower() in reply.lower() for s in e["reply_mentions"])
    if e.get("no_refund"): checks["no_refund"] = all(v == 0 for v in refunded.values())
    if "approval_asked" in e: checks["approval_asked"] = asked == e["approval_asked"]
    if "refunded" in e: checks["refunded"] = all(refunded.get(k) == v for k, v in e["refunded"].items())
    return {"id": case["id"], "passed": all(checks.values()), "checks": checks, "tools": tools, "approval_asked": asked, "reply": reply}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="evals/results.json"); args = ap.parse_args()
    results = [run_case(json.loads(line)) for line in CASES.read_text().splitlines() if line.strip()]
    passed = sum(r["passed"] for r in results)
    Path(args.out).write_text(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2))
    print(f"{passed}/{len(results)} cases passed")
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["id"] + ("" if r["passed"] else f"  {r['checks']}"))
