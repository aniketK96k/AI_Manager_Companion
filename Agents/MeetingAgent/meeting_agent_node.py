"""
meeting_agent_node.py
======================

Meeting specialist agent for the Manager AI router graph.

No keyword rules or hardcoded routing: the LLM is given tools and decides
which to call from the task text.

Tools the LLM can use
---------------------
- get_last_30_days_meetings   : your Google Calendar / Meet tool (always available)
- analyze_meeting_transcript  : runs the meeting_task_agent subgraph
                                (extract -> human approval -> tickets).
                                Only offered when the user supplied a transcript,
                                so the transcript stays OPTIONAL.

Approval flow
-------------
The subgraph's interrupt() propagates through the tool call to the OUTER graph
(the tool is invoked with the outer `config`). The caller reads
`snapshot.tasks[*].interrupts`, then resumes with
`app.ainvoke(Command(resume={...}), config)`. This node re-runs from the top
on resume and the subgraph continues from its interrupt. Build the subgraph
with checkpointer=None so it shares the outer checkpointer.
"""

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp

from Agents.MeetingAgent.GoogleMeet import get_last_30_days_meetings
from Agents.MeetingAgent.meeting_task_agent import (
    build_meeting_task_agent,
    make_initial_state,
    llm as _llm,
)

MAX_TOOL_ROUNDS = 4  # loop guard

MEETING_SYSTEM_PROMPT = """
You are the meeting agent. You work with the user's meetings using the tools
you have been given.

Rules:
- Decide from the task which tool(s) you need and call only those.
- Use tools to get real data. Never invent meetings, dates, links, action
  items or tickets.
- get_last_30_days_meetings returns calendar metadata only (no transcripts).
  Use it to list, count or filter meetings as the task asks.
- analyze_meeting_transcript, when available, analyzes the transcript the user
  provided and files tickets after human approval. Report its result as
  returned, including any ticket IDs.
- If the task needs a transcript and you have no analyze_meeting_transcript
  tool, say that no transcript was provided.
- If a tool returns an error, say so plainly and give the reason.
- Finish with a short, plain-text answer to the task.
"""


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

    def _build_tools(self, state, config: RunnableConfig) -> list:
        tools = [get_last_30_days_meetings]

        transcript = state.get("meeting_transcript")
        if transcript:
            sub_state = make_initial_state(
                # `or` (not .get default): the UI may send explicit None values
                meeting_title=state.get("meeting_title") or "Untitled meeting",
                meeting_date=state.get("meeting_date") or "",
                transcript=transcript,
            )

            @tool
            async def analyze_meeting_transcript() -> str:
                """Analyze the meeting transcript the user provided: extract action
                items, decisions and blockers, then (after human approval) create
                tickets for the approved action items. Takes no arguments."""
                # Passing the outer config lets interrupt() reach the outer graph.
                result = await self.subgraph.ainvoke(sub_state, config)
                return _text(result.get("final_response") or "Transcript processed.")

            tools.append(analyze_meeting_transcript)

        return tools

    async def node(self, state, config: RunnableConfig) -> Dict[str, Any]:
        task = state["current_task"]
        tools = self._build_tools(state, config)
        by_name = {t.name: t for t in tools}
        bound = _llm.bind_tools(tools)

        msgs = [
            SystemMessage(content=MEETING_SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task['task']}"),
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            resp = await bound.ainvoke(msgs)
            msgs.append(resp)

            calls = getattr(resp, "tool_calls", None) or []
            if not calls:
                text = _text(resp.content).strip() or "No answer was produced."
                return {"messages": [AIMessage(content=text)]}

            for c in calls:
                tool_obj = by_name.get(c["name"])
                if tool_obj is None:
                    out = f"Error: unknown tool '{c['name']}'"
                else:
                    try:
                        data = await tool_obj.ainvoke(c["args"])
                        out = data if isinstance(data, str) else json.dumps(data, default=str)
                    except GraphBubbleUp:
                        raise  # human-approval interrupt: must reach the outer graph
                    except Exception as e:  # noqa: BLE001
                        out = f"Error: {type(e).__name__}: {e}"
                msgs.append(ToolMessage(content=out, tool_call_id=c["id"], name=c["name"]))

        return {
            "messages": [
                AIMessage(
                    content="Stopped: the meeting agent kept calling tools without reaching an answer.",
                    additional_kwargs={"failed": True},
                )
            ]
        }

    def decision(self, state) -> str:
        return "complete"

    def complete(self, state) -> Dict[str, Any]:
        task = state["current_task"]

        last = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        text = _text(last.content) if last else ""

        # Replan only on an explicit failure signal from the node (not on
        # words that happen to appear in the answer).
        failed = bool(last and last.additional_kwargs.get("failed"))
        replan_reason = f"Task {task['id']} ({task['task']}) did not finish: {text}" if failed else None

        return {
            "completed_tasks": state["completed_tasks"] + [task],
            "results": {**state["results"], task["id"]: text},
            "current_task": None,
            "replan_reason": replan_reason,
        }