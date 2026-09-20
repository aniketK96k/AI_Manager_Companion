"""
meeting_agent_node.py
======================

Adapts the Phase 1 meeting_task_agent subgraph (transcript -> tasks/
decisions/blockers -> human approval -> ticket creation) into a specialist
agent that plugs into the Manager AI router graph in main.py, in place of
the old calendar_check stub.

Why this is a class, not a plain function:
    The meeting_task_agent subgraph pauses on a human-in-the-loop
    interrupt() (see human_approval() in meeting_task_agent.py) and needs
    the SAME checkpointer + thread_id to resume. That resume typically
    happens well after the outer Manager AI run has already returned its
    final_response — e.g. a human reviews the extracted action items in a
    UI later. So MeetingAgent keeps its checkpointer alive across calls
    and exposes `resume_approval(...)` as a separate entry point your app
    calls once a human has made a decision, independent of the outer
    graph's own invoke/response cycle.

How it fits the router graph's contract:
    `.node`, `.decision`, `.complete` have the same shape as the
    (agent_node, decision, complete_task) triplet that `_make_agent(...)`
    produces for project_agent/communication_agent in main.py, so they
    wire into graph.add_node / graph.add_conditional_edges the same way.
    The only difference: `.decision` always returns "complete" because all
    tool use (creating Jira/ADO tickets) happens *inside* the subgraph, so
    this agent never needs the outer agent<->tools loop.
"""

from typing import Dict, Any, Optional, List

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from Agents.MeetingAgent.meeting_task_agent import build_meeting_task_agent, make_initial_state


class MeetingAgent:
    def __init__(self, checkpointer=None):
        # Kept alive on `self` (not re-created per call) so resume_approval()
        # can find the paused thread later.
        self.checkpointer = checkpointer or MemorySaver()
        self.subgraph = build_meeting_task_agent(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # node / decision / complete — same shape as _make_agent's triplet
    # ------------------------------------------------------------------

    def node(self, state) -> Dict[str, Any]:
        task = state["current_task"]
        transcript = state.get("meeting_transcript")

        if not transcript:
            msg = AIMessage(
                content=(
                    f"No meeting transcript was provided for task '{task['task']}'. "
                    "Nothing to extract."
                )
            )
            return {"messages": [msg]}

        # One subgraph thread per outer task, so re-running the outer graph
        # (e.g. after a replan) doesn't collide with an already-paused
        # extraction thread for a different task.
        sub_config = {"configurable": {"thread_id": f"meeting-{task['id']}"}}
        sub_state = make_initial_state(
            meeting_title=state.get("meeting_title", "Untitled meeting"),
            meeting_date=state.get("meeting_date", ""),
            transcript=transcript,
        )

        result = self.subgraph.invoke(sub_state, sub_config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            summary = (
                f"Extracted {len(payload['action_items'])} action item(s), "
                f"{len(payload['decisions'])} decision(s), "
                f"{len(payload['blockers'])} blocker(s) from '{task['task']}'. "
                f"Awaiting human approval before any tickets are created "
                f"(task_id={task['id']})."
            )
            msg = AIMessage(content=summary)
            return {
                "messages": [msg],
                "pending_approvals": {
                    **state.get("pending_approvals", {}),
                    task["id"]: payload,
                },
            }

        msg = AIMessage(content=result.get("final_response", "Meeting agent finished."))
        return {"messages": [msg]}

    def decision(self, state) -> str:
        return "complete"

    def complete(self, state) -> Dict[str, Any]:
        task = state["current_task"]
        last_ai_text = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content:
                last_ai_text = m.content
                break

        replan_reason = None
        lowered = last_ai_text.lower()
        if any(
            p in lowered
            for p in ("could not", "couldn't", "doesn't exist", "no such", "failed", "no meeting transcript")
        ):
            replan_reason = f"Task {task['id']} ({task['task']}) outcome: {last_ai_text}"

        return {
            "completed_tasks": state["completed_tasks"] + [task],
            "results": {**state["results"], task["id"]: last_ai_text},
            "current_task": None,
            "replan_reason": replan_reason,
        }

    # ------------------------------------------------------------------
    # Independent entry point: resume a paused extraction after a human
    # has reviewed it. Not part of the outer graph's invoke() cycle.
    # ------------------------------------------------------------------

    def resume_approval(
        self,
        task_id: str,
        approved_ids: List[str],
        edits: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call once a human has reviewed the pending action items for
        `task_id` (found in state["pending_approvals"][task_id] after the
        outer graph run). Resumes the paused subgraph on its own thread
        and creates tickets for the approved items.

        Returns the subgraph's final state — includes `final_response`
        and `created_tickets`.
        """
        sub_config = {"configurable": {"thread_id": f"meeting-{task_id}"}}
        resume_payload = {"approved_ids": approved_ids, "edits": edits or {}}
        return self.subgraph.invoke(Command(resume=resume_payload), sub_config)