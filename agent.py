"""
Manager AI multi-agent system built with LangGraph.

Architecture (unchanged from the original design):
    - `planner`  : LLM call, runs exactly ONCE. Produces the full task
                   graph and stores it in state.task_plan (a fixed,
                   immutable list for the rest of the run).
    - `router`   : Pure Python, no LLM call. On every loop it looks at
                   task_plan + completed_tasks and deterministically
                   picks the next task whose dependencies are all
                   satisfied.
    - `replanner`: LLM call, only runs when an agent flags replan_reason.

What changed:
    `meeting_agent` is no longer a single calendar_check tool call. It's
    now backed by the full Phase 1 meeting_task_agent subgraph (transcript
    -> action items/decisions/blockers -> human approval -> Jira/ADO
    ticket creation), via the MeetingAgent adapter in meeting_agent_node.py.

    Because that subgraph can pause for human approval and resume much
    later, `meeting_agent` never blocks the outer graph waiting for a
    human. If it hits that pause, it records the pending payload in
    state["pending_approvals"][task_id] and finishes its turn normally.
    Your application calls `meeting_agent_instance.resume_approval(...)`
    separately, whenever a human has actually reviewed the items — see
    the bottom of this file.

    project_agent and communication_agent are unchanged stubs — swap their
    tools for real Jira/Slack/email clients when ready.
"""

from typing import TypedDict, List, Annotated, Dict, Any, Literal, Optional
import os

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from Agents.MeetingAgent.meeting_agent_node import MeetingAgent

from dotenv import load_dotenv

load_dotenv()
# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # Per-task scratchpad for the agent<->tools loop: [System, Human,
    # AI(tool_call), Tool(result), AI(...), ...]. Reset to empty every
    # time the router hands off a new task (see router()), so an agent
    # always sees the full back-and-forth for ITS task — including tool
    # results — instead of re-sending the same bare instruction every
    # loop with no memory of what it already tried.
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
    # Supply these alongside user_query when the request involves a
    # meeting transcript (e.g. "summarize yesterday's architecture review
    # and file tickets for the action items").
    meeting_transcript: Optional[str]
    meeting_title: Optional[str]
    meeting_date: Optional[str]
    # Populated by meeting_agent when a subgraph run pauses for human
    # approval: {task_id: {action_items, decisions, blockers, ...}}
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
2. communication_agent — Slack, email, messages, finding people
3. meeting_agent — analyzing a meeting transcript into action items,
   decisions and blockers, and filing tickets for them

Rules:
- Use task IDs task_1, task_2, ...
- Fill depends_on with the IDs of tasks that must finish first.
- Prefer the fewest tasks that fully satisfy the request.
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
        # Wipe the previous task's scratchpad so the new agent starts
        # with a clean conversation, not the last agent's tool history.
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
# TOOLS for project_agent / communication_agent
# (replace with real Jira/Slack integrations)
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


@tool
def send_message(channel: str, text: str) -> str:
    """Send a Slack message or email. `channel` is a person, #channel, or email address."""
    return f"[stub] Sent to {channel}: {text}"


project_tools = [jira_lookup, create_jira_ticket]
communication_tools = [send_message]

project_tool_node = ToolNode(project_tools, messages_key="task_messages")
communication_tool_node = ToolNode(communication_tools, messages_key="task_messages")


# ============================================================
# SPECIALIST AGENTS: project_agent, communication_agent
# ============================================================

MAX_TOOL_ROUNDS = 3  # loop guard: safety net even after fixing the memory bug


def _make_agent(agent_name: str, system_prompt: str, tools: list):
    bound_llm = llm.bind_tools(tools)

    def agent_node(state: State) -> Dict[str, Any]:
        task = state["current_task"]
        conversation = state.get("task_messages") or []

        if not conversation:
            # First turn for this task: seed the conversation. Everything
            # returned here (seed + response) gets appended to
            # task_messages via the add_messages reducer.
            seed = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Task: {task['task']}"),
            ]
            response = bound_llm.invoke(seed)
            return {"task_messages": seed + [response]}

        # Later turns: `conversation` already holds the System/Human seed
        # plus every prior AI(tool_call) + Tool(result) pair for this
        # task (persisted in state via the reducer), so the model can see
        # what it already tried instead of repeating it blind.
        response = bound_llm.invoke(conversation)
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
            # Loop guard tripped: stop feeding the tool loop even though
            # the model is still asking for another tool call.
            return "complete"

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
                f"actually satisfy the task (e.g. only a lookup tool exists, not a "
                f"creation tool) — check whether a different tool or agent is needed."
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


project_agent, project_agent_decision_raw, project_complete = _make_agent(
    "project_agent",
    "You help with Jira/Linear project status. Use tools when you need real data.",
    project_tools,
)
communication_agent, communication_agent_decision_raw, communication_complete = _make_agent(
    "communication_agent",
    "You help send Slack messages and emails. Use tools when you need to actually send something.",
    communication_tools,
)


# ============================================================
# SPECIALIST AGENT: meeting_agent (backed by the Phase 1 subgraph)
# ============================================================

meeting_agent_instance = MeetingAgent()


# ============================================================
# FINAL AGENT
# ============================================================

def final_agent(state: State) -> Dict[str, Any]:
    summary_lines = [f"- {task_id}: {result}" for task_id, result in state["results"].items()]
    final_response = "Here's what I did:\n" + "\n".join(summary_lines) if summary_lines else \
        "I didn't find any tasks that needed doing."

    pending = state.get("pending_approvals") or {}
    if pending:
        final_response += "\n\nWaiting on human approval for:\n" + "\n".join(
            f"- task {tid}: {len(p['action_items'])} action item(s) pending review" for tid, p in pending.items()
        )

    return {"final_response": final_response}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("planner", planner)
    graph.add_node("router", router)

    graph.add_node("project_agent", project_agent)
    graph.add_node("project_tools", project_tool_node)
    graph.add_node("project_complete", project_complete)

    graph.add_node("communication_agent", communication_agent)
    graph.add_node("communication_tools", communication_tool_node)
    graph.add_node("communication_complete", communication_complete)

    # meeting_agent has no outer tools node: ticket creation happens inside
    # the meeting_task_agent subgraph, so its decision() always goes
    # straight to "complete".
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
        project_agent_decision_raw,
        {"tools": "project_tools", "complete": "project_complete"},
    )
    graph.add_edge("project_tools", "project_agent")
    graph.add_conditional_edges(
        "project_complete", needs_replan, {"replanner": "replanner", "router": "router"}
    )

    graph.add_conditional_edges(
        "communication_agent",
        communication_agent_decision_raw,
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


if __name__ == "__main__":
    app = build_graph().compile()

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

    from langgraph.errors import GraphRecursionError

    try:
        result = app.invoke(
            {
                "messages": [],
                "task_messages": [],
                "user_query": "Go through yesterday's architecture review and file tickets for the action items.",
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
            },
            # Belt-and-suspenders on top of the per-agent loop guard above:
            # if something still runs away, LangGraph raises instead of
            # looping indefinitely / burning API quota unattended.
            config={"recursion_limit": 50},
        )
    except GraphRecursionError:
        print(
            "Graph hit its recursion limit (50 steps) without finishing — "
            "something is still looping. Check task_messages for the task "
            "that was active when this happened."
        )
        raise
    print(result["final_response"])

    # If the meeting task paused for approval, result["pending_approvals"]
    # holds it. Resume it independently, whenever a human has reviewed it:
    pending = result.get("pending_approvals") or {}
    for task_id, payload in pending.items():
        print(f"\n--- pending approval for {task_id} ---")
        for item in payload["action_items"]:
            print(f"  {item['id']}: {item['task']} (owner={item['owner']})")

        # Example: approve everything as-is.
        approved_ids = [item["id"] for item in payload["action_items"]]
        resume_result = meeting_agent_instance.resume_approval(task_id, approved_ids)
        print(resume_result["final_response"])