"""
meeting_agent_node.py
======================

Adapts the Phase 1 meeting_task_agent subgraph (transcript -> tasks/
decisions/blockers -> human approval -> ticket creation) into a specialist
agent for the Manager AI router graph.

Approval design
---------------
The subgraph calls interrupt() for human approval. Because it is invoked
from inside an outer-graph node WITH the outer `config`, LangGraph
propagates that interrupt up to the outer graph:

    1. outer app.ainvoke(...) returns paused
    2. caller reads `snapshot.tasks[*].interrupts` (the approval payload)
    3. caller resumes with: app.ainvoke(Command(resume={...}), config)
    4. this node re-runs from the top; the subgraph resumes from its
       interrupt (it does NOT re-extract) and creates the tickets

So there is no separate resume entry point and no pending_approvals
bookkeeping: one outer thread_id, one resume path.

For this to work the subgraph must share the outer checkpointer, so
build it with checkpointer=None (the default here) and compile the OUTER
graph with a checkpointer (main.py does: MemorySaver()).
"""

from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from Agents.MeetingAgent.meeting_task_agent import (
    build_meeting_task_agent,
    make_initial_state,
)


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")
        )
    return str(content)


class MeetingAgent:
    def __init__(self, checkpointer=None):
        # None => the subgraph inherits the outer graph's checkpointer,
        # which is what lets interrupt()/resume work across the boundary.
        self.subgraph = build_meeting_task_agent(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # node / decision / complete — same shape as _make_agent's triplet
    # ------------------------------------------------------------------

    async def node(self, state, config: RunnableConfig) -> Dict[str, Any]:
        task = state["current_task"]
        transcript = state.get("meeting_transcript")

        if not transcript:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"No meeting transcript was provided for task "
                            f"'{task['task']}'. Nothing to extract."
                        )
                    )
                ]
            }

        sub_state = make_initial_state(
            # `or` (not .get default): the UI may send explicit None values
            meeting_title=state.get("meeting_title") or "Untitled meeting",
            meeting_date=state.get("meeting_date") or "",
            transcript=transcript,
        )

        # ainvoke (not invoke), and RETURN a state update. If the subgraph
        # hits interrupt(), GraphInterrupt propagates and the outer graph
        # pauses here; on resume this node runs again and the subgraph
        # continues from its interrupt.
        result = await self.subgraph.ainvoke(sub_state, config)

        text = _text(result.get("final_response") or "").strip()
        tickets = result.get("created_tickets") or []
        if tickets:
            text += f"\nCreated {len(tickets)} ticket(s)."
        return {"messages": [AIMessage(content=text or "Meeting processed.")]}

    def decision(self, state) -> str:
        return "complete"

    def complete(self, state) -> Dict[str, Any]:
        task = state["current_task"]

        last_ai_text = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content:
                last_ai_text = _text(m.content)
                break

        replan_reason = None
        lowered = last_ai_text.lower()
        if any(
            p in lowered
            for p in (
                "could not",
                "couldn't",
                "doesn't exist",
                "no such",
                "failed",
                "no meeting transcript",
            )
        ):
            replan_reason = f"Task {task['id']} ({task['task']}) outcome: {last_ai_text}"

        return {
            "completed_tasks": state["completed_tasks"] + [task],
            "results": {**state["results"], task["id"]: last_ai_text},
            "current_task": None,
            "replan_reason": replan_reason,
        }