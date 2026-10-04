"""
Streamlit UI for the Manager AI multi-agent system.

Run (from the project root):   streamlit run app.py

The graph lives in a background thread with its own asyncio loop so the
Gmail MCP session and the checkpointer survive Streamlit's script re-runs.
"""

import asyncio
import importlib
import queue
import sys
import threading
import traceback
import uuid
from datetime import date

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

MODULE_NAME = "agent"  # your graph file (agent.py), without ".py"

st.set_page_config(page_title="Manager AI", page_icon="🧭", layout="wide")

AGENT_ICONS = {
    "planner": "📋",
    "router": "🔀",
    "replanner": "♻️",
    "project_agent": "🗂️",
    "project_tools": "🔧",
    "project_complete": "✅",
    "communication_agent": "✉️",
    "communication_tools": "🔧",
    "communication_complete": "✅",
    "calendar_agent": "📅",
    "calendar_tools": "🔧",
    "calendar_complete": "✅",
    "meeting_agent": "🎙️",
    "meeting_complete": "✅",
    "final_agent": "🏁",
}


# ============================================================
# Background runtime (keeps MCP session + checkpointer alive)
# ============================================================

def _root_causes(exc):
    """Unwrap (nested) ExceptionGroups so the real error is visible."""
    subs = getattr(exc, "exceptions", None)
    if subs:
        out = []
        for s in subs:
            out.extend(_root_causes(s))
        return out
    return [exc]


class Runtime:
    def __init__(self):
        self.error = None
        self.app = None
        self.ready = threading.Event()
        try:
            self.ma = importlib.import_module(MODULE_NAME)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not import {MODULE_NAME}.py: {e!r}") from e

        # Windows needs a Proactor loop to spawn MCP stdio subprocesses.
        self.loop = (
            asyncio.ProactorEventLoop()
            if sys.platform == "win32"
            else asyncio.new_event_loop()
        )
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.wait(timeout=90)
        if self.error:
            raise RuntimeError(self.error)
        if self.app is None:
            raise RuntimeError("Timed out starting the Gmail MCP session.")

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())

    async def _main(self):
        try:
            async with self.ma.gmail_tools() as comm_tools:
                self.app = self.ma.build_graph(comm_tools).compile(
                    checkpointer=MemorySaver()
                )
                self.ready.set()
                await asyncio.Event().wait()  # hold the session open forever
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()  # full trace in the terminal
            causes = _root_causes(e)
            self.error = "Gmail MCP failed to start:\n" + "\n".join(
                f"{type(c).__name__}: {c!r}" for c in causes
            )
            self.ready.set()

    def stream(self, graph_input, config):
        """Yield ("update", chunk) ... then ("done", info) or ("error", msg)."""
        q: queue.Queue = queue.Queue()

        async def go():
            try:
                async for chunk in self.app.astream(
                    graph_input, config, stream_mode="updates"
                ):
                    q.put(("update", chunk))
                snap = await self.app.aget_state(config)
                interrupts = [i.value for t in snap.tasks for i in t.interrupts]
                q.put(("done", {"values": snap.values, "interrupts": interrupts}))
            except BaseException as e:  # noqa: BLE001
                traceback.print_exc()
                causes = _root_causes(e)
                q.put(("error", "\n".join(f"{type(c).__name__}: {c}" for c in causes)))

        asyncio.run_coroutine_threadsafe(go(), self.loop)
        while True:
            kind, payload = q.get()
            yield kind, payload
            if kind in ("done", "error"):
                return


@st.cache_resource(show_spinner="Connecting to Gmail MCP and building the graph…")
def get_runtime() -> Runtime:
    return Runtime()


# ============================================================
# Helpers
# ============================================================

def describe(node: str, data) -> str:
    data = data if isinstance(data, dict) else {}
    icon = AGENT_ICONS.get(node, "•")
    if node == "planner":
        return f"{icon} **planner** created a plan with {len(data.get('task_plan', []))} task(s)"
    if node == "router":
        t = data.get("current_task")
        return (
            f"{icon} **router** → `{t['agent']}` : {t['task']}"
            if t
            else f"{icon} **router** → all tasks done"
        )
    if node == "replanner":
        return f"{icon} **replanner** revised the remaining plan"
    if node.endswith("_complete"):
        results = data.get("results", {})
        last = list(results.values())[-1] if results else ""
        return f"{icon} **{node}** finished. {str(last)[:160]}"
    if node.endswith("_tools"):
        return f"{icon} **{node}** ran a tool"
    return f"{icon} **{node}**"


def initial_state(query: str) -> dict:
    return {
        "messages": [],
        "task_messages": [],
        "user_query": query,
        "task_plan": [],
        "current_task": None,
        "completed_tasks": [],
        "results": {},
        "next_agent": "",
        "final_response": "",
        "replan_reason": None,
        "meeting_transcript": st.session_state.transcript or None,
        "meeting_title": st.session_state.meeting_title or None,
        "meeting_date": str(st.session_state.meeting_date),
        "pending_approvals": {},
    }


def run_graph(graph_input, thread_id: str, entry: dict):
    """Stream a run (or resume) into `entry`, then rerun the page."""
    try:
        rt = get_runtime()
    except Exception as e:  # noqa: BLE001
        entry["error"] = str(e)
        st.rerun()
        return

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=True) as status:
            for kind, payload in rt.stream(graph_input, config):
                if kind == "update":
                    for node, data in payload.items():
                        if node == "__interrupt__":
                            line = "⏸️ Paused — waiting for your approval"
                        else:
                            line = describe(node, data)
                            if node == "planner" and isinstance(data, dict):
                                entry["plan"] = data.get("task_plan", [])
                        entry["log"].append(line)
                        st.write(line)
                elif kind == "error":
                    entry["error"] = payload
                    status.update(label="Failed", state="error")
                elif kind == "done":
                    values = payload["values"]
                    entry["results"] = values.get("results", {})
                    if payload["interrupts"]:
                        st.session_state.pending = {
                            "thread_id": thread_id,
                            "payload": payload["interrupts"][0],
                            "entry": entry,
                        }
                        status.update(label="Needs your approval", state="running")
                    else:
                        st.session_state.pending = None
                        entry["final"] = values.get("final_response", "")
                        status.update(label="Done", state="complete", expanded=False)
    st.rerun()


# ============================================================
# Session state
# ============================================================

ss = st.session_state
ss.setdefault("entries", [])
ss.setdefault("pending", None)
ss.setdefault("transcript", "")
ss.setdefault("meeting_title", "")
ss.setdefault("meeting_date", date.today())

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("🧭 Manager AI")
    st.caption("Planner → Router → Project / Communication / Calendar / Meeting agents")

    st.subheader("Meeting context")
    st.caption("Used by the meeting agent when your request involves a transcript.")
    ss.meeting_title = st.text_input("Title", ss.meeting_title)
    ss.meeting_date = st.date_input("Date", ss.meeting_date)
    ss.transcript = st.text_area("Transcript", ss.transcript, height=220)

    st.subheader("Try")
    examples = [
        "Summarize my 3 most recent unread emails.",
        "List my Google Meet meetings from the last 30 days.",
        "Analyze the meeting transcript and file tickets for the action items.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, disabled=bool(ss.pending)):
            ss.queued_query = ex
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        ss.entries = []
        ss.pending = None
        st.rerun()

# ============================================================
# Main: history
# ============================================================

st.header("Manager AI")

if not ss.entries:
    st.info("Ask me about email, calendar, projects, or meetings. Try an example from the sidebar.")

for e in ss.entries:
    with st.chat_message("user"):
        st.write(e["query"])
    with st.chat_message("assistant"):
        if e.get("plan"):
            with st.expander(f"Plan ({len(e['plan'])} tasks)"):
                st.dataframe(
                    [
                        {
                            "id": t["id"],
                            "agent": t["agent"],
                            "task": t["task"],
                            "depends_on": ", ".join(t.get("depends_on", [])),
                        }
                        for t in e["plan"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
        if e["log"]:
            with st.expander("Agent activity"):
                for line in e["log"]:
                    st.markdown(line)
        if e.get("error"):
            st.error(e["error"])
        if e.get("final"):
            st.markdown(e["final"])
        elif e is (ss.pending or {}).get("entry"):
            st.warning("Waiting on your approval below.")

# ============================================================
# Main: human approval panel
# ============================================================

if ss.pending:
    payload = ss.pending["payload"]
    items = payload.get("action_items", [])

    st.divider()
    st.subheader("⏸️ Approval required")
    st.caption("Review the action items extracted from the meeting. Edit, then approve.")

    with st.form("approval_form"):
        decisions = {}
        for item in items:
            c1, c2, c3 = st.columns([0.1, 0.5, 0.2])
            approved = c1.checkbox("OK", value=True, key=f"ap_{item['id']}")
            c2.markdown(f"**{item['id']}** — {item['task']}")
            owner = c3.text_input(
                "Owner", item.get("owner") or "", key=f"ow_{item['id']}"
            )
            deadline = st.text_input(
                "Deadline",
                item.get("deadline") or "",
                key=f"dl_{item['id']}",
                label_visibility="collapsed",
                placeholder="Deadline",
            )
            decisions[item["id"]] = (approved, owner, deadline, item)

        col_a, col_b = st.columns(2)
        submit = col_a.form_submit_button("✅ Submit decision", type="primary")
        reject = col_b.form_submit_button("🚫 Reject all")

    if submit or reject:
        approved_ids, edits = [], {}
        if submit:
            for iid, (ok, owner, deadline, orig) in decisions.items():
                if ok:
                    approved_ids.append(iid)
                change = {}
                if owner != (orig.get("owner") or ""):
                    change["owner"] = owner
                if deadline != (orig.get("deadline") or ""):
                    change["deadline"] = deadline
                if change:
                    edits[iid] = change
        pend = ss.pending
        ss.pending = None
        run_graph(
            Command(resume={"approved_ids": approved_ids, "edits": edits}),
            pend["thread_id"],
            pend["entry"],
        )

# ============================================================
# Main: chat input
# ============================================================

query = st.chat_input("Ask the Manager AI…", disabled=bool(ss.pending))
query = query or ss.pop("queued_query", None)

if query:
    entry = {"query": query, "log": [], "plan": [], "results": {}, "final": None}
    ss.entries.append(entry)
    with st.chat_message("user"):
        st.write(query)
    run_graph(initial_state(query), f"run-{uuid.uuid4().hex[:8]}", entry)