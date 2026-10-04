"""
Manager AI multi-agent system built with LangGraph.

Architecture:
    - `planner`  : LLM call, runs exactly ONCE. Produces the full task graph.
    - `router`   : Pure Python. Picks the next task whose dependencies are done.
    - `replanner`: LLM call, only runs when an agent flags replan_reason.

Agents:
    - project_agent       : stub Jira/Linear tools (swap for real ones later)
    - communication_agent : REAL Gmail tools loaded from the Gmail MCP server
    - meeting_agent       : meetings. Lists the last 30 days of Google Meet /
                            Calendar meetings; if a transcript is given, runs
                            the meeting_task_agent subgraph (transcript ->
                            tickets, with a human-approval interrupt)

The graph must be built AFTER the Gmail MCP session is opened and run with
`await app.ainvoke(...)` / `app.astream(...)`. It must be compiled with a
checkpointer (needed for the meeting agent's approval interrupt).
"""

import asyncio
import os
from typing import TypedDict, List, Annotated, Dict, Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    RemoveMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from Agents.CommunicationAgent.Communicationtools import (
    gmail_tools,
    COMMUNICATION_SYSTEM_PROMPT,
)
from Agents.MeetingAgent.meeting_agent_node import MeetingAgent

load_dotenv()

# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=4096,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # Per-task scratchpad for the agent<->tools loop. Reset by the router
    # every time a new task is handed off.
    task_messages: Annotated[List[BaseMessage], add_messages]

    user_query: str

    task_plan: List[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]
    results: Dict[str, Any]

    next_agent: str
    final_response: str
    replan_reason: Optional[str]

    # --- meeting_agent inputs/outputs ---
    meeting_transcript: Optional[str]
    meeting_title: Optional[str]
    meeting_date: Optional[str]
    pending_approvals: Dict[str, Any]


AgentName = Literal["project_agent", "communication_agent", "meeting_agent"]


class Task(BaseModel):
    id: str = Field(description="Unique task ID, e.g. task_1")
    agent: AgentName = Field(description="Agent responsible for this task")
    task: str = Field(description="Specific instruction the agent must perform")
    depends_on: List[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    tasks: List[Task] = Field(description="Complete task plan for the user request")


# ============================================================
# PLANNER (LLM — runs exactly once)
# ============================================================

PLANNER_SYSTEM_PROMPT = """
You are the planner for a Manager AI system.

Break the user's request into the smallest possible set of tasks and
assign each to the right agent:

1. project_agent — Jira, Linear, project status, tasks, blockers
2. communication_agent — Gmail: searching, reading, summarizing and
   drafting emails (it has no Slack access)
3. meeting_agent — everything about meetings: listing the user's Google
   Meet / Calendar meetings (e.g. the last 30 days) and, only if the user
   supplied a transcript, extracting action items, decisions and blockers
   and filing tickets. A transcript is OPTIONAL — never plan a task to
   "retrieve" or "obtain" a transcript.

Rules:
- Use task IDs task_1, task_2, ...
- Fill depends_on with the IDs of tasks that must finish first.
- Create exactly one task per thing the user asked for, and nothing more.
  Never add extra search, read, verify, summarize or notify tasks.
  Example: "send an email to X" is ONE communication_agent task.
  Example: "what are my meetings in the last 30 days" is ONE meeting_agent task.
- If the user says which agent to use or avoid, obey that.
- Do not invent work the user didn't ask for.
"""


def planner(state: State) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(PlannerOutput)

    decision = structured_llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=state["user_query"]),
        ]
    )

    return {
        "task_plan": [t.model_dump() for t in decision.tasks],
        "completed_tasks": [],
        "results": {},
        "pending_approvals": {},
    }


# ============================================================
# ROUTER (pure Python — no LLM call, runs every loop)
# ============================================================

def _next_ready_task(
    task_plan: List[Dict[str, Any]], completed_tasks: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    completed_ids = {t["id"] for t in completed_tasks}
    for task in task_plan:
        if task["id"] in completed_ids:
            continue
        if all(dep in completed_ids for dep in task.get("depends_on", [])):
            return task
    return None


def router(state: State) -> Dict[str, Any]:
    next_task = _next_ready_task(state["task_plan"], state["completed_tasks"])

    if next_task is None:
        return {"current_task": None, "next_agent": "final_agent"}

    return {
        "current_task": next_task,
        "next_agent": next_task["agent"],
        # Wipe the previous task's scratchpad so the new agent starts clean.
        "task_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }


def router_decision(state: State) -> str:
    return state["next_agent"]


# ============================================================
# REPLANNER (LLM — only runs when an agent flags replan_reason)
# ============================================================

class ReplanOutput(BaseModel):
    remaining_tasks: List[Task] = Field(
        description="Replacement task list for everything NOT yet completed. "
        "Do not include already-completed tasks."
    )


REPLAN_SYSTEM_PROMPT = """
You are revising a task plan for a Manager AI system mid-execution.

Some tasks are already done — their results are given below and must
NOT be redone or reversed. Given why replanning was triggered, produce
an updated list of the REMAINING tasks only (new IDs, e.g. task_4,
task_5, continuing from where the plan left off). Depends_on may
reference already-completed task IDs too.

Agents available: project_agent, communication_agent, meeting_agent.

Only plan work needed to recover from the problem and finish the ORIGINAL
request. Never add new work the user did not ask for. If the request is
already satisfied or cannot be fixed, return an empty list.
"""


def replanner(state: State) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(ReplanOutput)

    context = f"""
Original request: {state['user_query']}

Completed tasks and results:
{[(t['id'], state['results'].get(t['id'], '')) for t in state['completed_tasks']]}

Reason replanning was triggered:
{state['replan_reason']}
"""
    decision = structured_llm.invoke(
        [SystemMessage(content=REPLAN_SYSTEM_PROMPT), HumanMessage(content=context)]
    )

    new_remaining = [t.model_dump() for t in decision.remaining_tasks]
    return {
        "task_plan": state["completed_tasks"] + new_remaining,
        "replan_reason": None,
    }


def needs_replan(state: State) -> str:
    return "replanner" if state.get("replan_reason") else "router"


# ============================================================
# TOOLS for project_agent (stubs — replace with real Jira/Linear)
# communication_agent tools come from the Gmail MCP server at runtime.
# ============================================================

@tool
def jira_lookup(query: str) -> str:
    """Look up Jira/Linear issues or project status matching the query."""
    return f"[stub] Jira results for: {query}"


@tool
def create_jira_ticket(summary: str, owner: str = "", priority: str = "Medium") -> str:
    """Create a new Jira/Linear ticket. Use this — not jira_lookup — when
    the task is to file/create/open a ticket, not just check status."""
    import uuid

    ticket_id = f"PROJ-{uuid.uuid4().hex[:5].upper()}"
    return f"[stub] Created {ticket_id}: '{summary}' (owner={owner or 'unassigned'}, priority={priority})"


project_tools = [jira_lookup, create_jira_ticket]
project_tool_node = ToolNode(project_tools, messages_key="task_messages")


# ============================================================
# AGENT FACTORY (project_agent, communication_agent)
# ============================================================

MAX_TOOL_ROUNDS = 6  # loop guard: Gmail tasks often need search -> read -> read


def _make_agent(agent_name: str, system_prompt: str, tools: list):
    bound_llm = llm.bind_tools(tools)

    # async so that async-only MCP tools work (graph must run via ainvoke)
    async def agent_node(state: State) -> Dict[str, Any]:
        task = state["current_task"]
        conversation = state.get("task_messages") or []

        if not conversation:
            # First turn for this task: seed the conversation.
            seed = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Task: {task['task']}"),
            ]
            response = await bound_llm.ainvoke(seed)
            return {"task_messages": seed + [response]}

        # Later turns: conversation holds the seed plus every prior
        # AI(tool_call) + Tool(result) pair for this task.
        response = await bound_llm.ainvoke(conversation)
        return {"task_messages": [response]}

    def decision(state: State) -> str:
        task_messages = state.get("task_messages") or []
        if not task_messages:
            return "complete"

        last = task_messages[-1]
        wants_tool_call = isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))
        tool_rounds = sum(
            1 for m in task_messages if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )

        if wants_tool_call and tool_rounds > MAX_TOOL_ROUNDS:
            return "complete"  # loop guard tripped

        return "tools" if wants_tool_call else "complete"

    def complete_task(state: State) -> Dict[str, Any]:
        task = state["current_task"]
        task_messages = state.get("task_messages") or []

        tool_rounds = sum(
            1 for m in task_messages if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )

        last_ai_text = ""
        for m in reversed(task_messages):
            if isinstance(m, AIMessage) and m.content:
                last_ai_text = (
                    "\n".join(
                        block.get("text", "")
                        for block in m.content
                        if isinstance(block, dict) and block.get("text")
                    )
                    if isinstance(m.content, list)
                    else str(m.content)
                )
                break

        replan_reason = None
        if tool_rounds > MAX_TOOL_ROUNDS:
            warning = (
                f"[loop guard] {agent_name} called a tool {tool_rounds} times on task "
                f"{task['id']} ({task['task']}) without reaching a final answer — stopped "
                f"to avoid an infinite loop / burning API quota."
            )
            print(warning)
            last_ai_text = warning + (f" Last output: {last_ai_text}" if last_ai_text else " No output produced.")
            replan_reason = (
                f"Task {task['id']} ({task['task']}) got stuck in a tool-call loop "
                f"({tool_rounds} rounds). This usually means the available tools can't "
                f"actually satisfy the task — check whether a different tool or agent is needed."
            )
        else:
            lowered = last_ai_text.lower()
            if any(p in lowered for p in ("could not", "couldn't", "doesn't exist", "no such", "failed")):
                replan_reason = f"Task {task['id']} ({task['task']}) outcome: {last_ai_text}"

        return {
            "completed_tasks": state["completed_tasks"] + [task],
            "results": {**state["results"], task["id"]: last_ai_text},
            "current_task": None,
            "replan_reason": replan_reason,
        }

    return agent_node, decision, complete_task


project_agent, project_agent_decision, project_complete = _make_agent(
    "project_agent",
    "You help with Jira/Linear project status. Use tools when you need real data.",
    project_tools,
)


# ============================================================
# SPECIALIST AGENT: meeting_agent (backed by the meeting_task_agent subgraph)
# ============================================================

meeting_agent_instance = MeetingAgent()


# ============================================================
# FINAL AGENT
# ============================================================

def final_agent(state: State) -> Dict[str, Any]:
    summary_lines = [f"- {task_id}: {result}" for task_id, result in state["results"].items()]
    final_response = (
        "Here's what I did:\n" + "\n".join(summary_lines)
        if summary_lines
        else "I didn't find any tasks that needed doing."
    )

    pending = state.get("pending_approvals") or {}
    if pending:
        final_response += "\n\nWaiting on human approval for:\n" + "\n".join(
            f"- task {tid}: {len(p['action_items'])} action item(s) pending review"
            for tid, p in pending.items()
        )

    return {"final_response": final_response}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(comm_tools: list) -> StateGraph:
    """Build the graph. `comm_tools` are the Gmail MCP tools (already loaded
    from an open MCP session)."""
    graph = StateGraph(State)

    # communication_agent is created here because its tools only exist at runtime
    communication_tool_node = ToolNode(comm_tools, messages_key="task_messages")
    communication_agent, communication_agent_decision, communication_complete = _make_agent(
        "communication_agent",
        COMMUNICATION_SYSTEM_PROMPT,
        comm_tools,
    )

    graph.add_node("planner", planner)
    graph.add_node("router", router)

    graph.add_node("project_agent", project_agent)
    graph.add_node("project_tools", project_tool_node)
    graph.add_node("project_complete", project_complete)

    graph.add_node("communication_agent", communication_agent)
    graph.add_node("communication_tools", communication_tool_node)
    graph.add_node("communication_complete", communication_complete)

    # meeting_agent has no outer tools node: ticket creation happens inside
    # the meeting_task_agent subgraph.
    graph.add_node("meeting_agent", meeting_agent_instance.node)
    graph.add_node("meeting_complete", meeting_agent_instance.complete)

    graph.add_node("final_agent", final_agent)
    graph.add_node("replanner", replanner)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("replanner", "router")

    graph.add_conditional_edges(
        "router",
        router_decision,
        {
            "project_agent": "project_agent",
            "communication_agent": "communication_agent",
            "meeting_agent": "meeting_agent",
            "final_agent": "final_agent",
        },
    )

    graph.add_conditional_edges(
        "project_agent",
        project_agent_decision,
        {"tools": "project_tools", "complete": "project_complete"},
    )
    graph.add_edge("project_tools", "project_agent")
    graph.add_conditional_edges(
        "project_complete", needs_replan, {"replanner": "replanner", "router": "router"}
    )

    graph.add_conditional_edges(
        "communication_agent",
        communication_agent_decision,
        {"tools": "communication_tools", "complete": "communication_complete"},
    )
    graph.add_edge("communication_tools", "communication_agent")
    graph.add_conditional_edges(
        "communication_complete", needs_replan, {"replanner": "replanner", "router": "router"}
    )

    graph.add_conditional_edges(
        "meeting_agent",
        meeting_agent_instance.decision,
        {"complete": "meeting_complete"},
    )
    graph.add_conditional_edges(
        "meeting_complete", needs_replan, {"replanner": "replanner", "router": "router"}
    )

    graph.add_edge("final_agent", END)

    return graph


# ============================================================
# RUN (command-line alternative to the Streamlit UI)
# ============================================================

async def main():
    sample_transcript = """
    [Architecture Review - Sep 10]

    Priya: So where are we on the TILCON to Qt migration?
    Aniket: API migration is basically done. I'll finish the last two
    modules and get QNX testing started by September 20.
    Priya: Great. Also - we're going with Qt as the permanent replacement
    for TILCON, that's settled.
    Rahul: One issue - the IO Manager memory fault is still unresolved,
    this is the third meeting we've raised it in. It's blocking QNX
    integration testing.
    """

    initial_state = {
        "messages": [],
        "task_messages": [],
        "user_query": "Summarize my 3 most recent unread emails.",
        "task_plan": [],
        "current_task": None,
        "completed_tasks": [],
        "results": {},
        "next_agent": "",
        "final_response": "",
        "replan_reason": None,
        "meeting_transcript": sample_transcript,
        "meeting_title": "Architecture Review",
        "meeting_date": "2026-09-10",
        "pending_approvals": {},
    }
    config = {"configurable": {"thread_id": "manager-run-1"}, "recursion_limit": 50}

    # The Gmail MCP session must stay open for the entire graph run.
    async with gmail_tools() as comm_tools:
        app = build_graph(comm_tools).compile(checkpointer=MemorySaver())

        try:
            result = await app.ainvoke(initial_state, config)
        except GraphRecursionError:
            print(
                "Graph hit its recursion limit (50 steps) without finishing — "
                "something is still looping."
            )
            raise

        while True:
            snapshot = await app.aget_state(config)
            interrupts = [i for t in snapshot.tasks for i in t.interrupts]
            if not interrupts:
                break

            payload = interrupts[0].value
            print("\n--- APPROVAL REQUIRED ---")
            for item in payload["action_items"]:
                print(f"  {item['id']}: {item['task']} (owner={item['owner']}, deadline={item['deadline']})")

            # Replace with real input. Here: approve everything.
            approved_ids = [item["id"] for item in payload["action_items"]]
            result = await app.ainvoke(
                Command(resume={"approved_ids": approved_ids, "edits": {}}),
                config,
            )

    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())