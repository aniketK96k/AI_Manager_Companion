"""
meeting_task_agent.py
======================

Meeting -> Task Tracking Agent, packaged for import.

Standalone usage (needs a checkpointer because of the approval interrupt):

    from meeting_task_agent import build_meeting_task_agent, make_initial_state
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    app = build_meeting_task_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "meeting-001"}}

    result = app.invoke(make_initial_state("Title", "2026-09-10", text), config)
    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        result = app.invoke(
            Command(resume={"approved_ids": [...], "edits": {...}}), config
        )
    print(result["final_response"])

Inside the Manager AI graph, leave checkpointer=None: the subgraph then
inherits the outer graph's checkpointer, which is what lets the approval
interrupt pause the outer graph and resume correctly.

Nothing runs on import except creating the LLM client.
"""

from typing import TypedDict, List, Annotated, Dict, Any, Literal, Optional
import os
import uuid

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

__all__ = [
    "State",
    "ActionItem",
    "Decision",
    "Blocker",
    "ExtractionOutput",
    "build_meeting_task_agent",
    "make_initial_state",
]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=4096,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# ============================================================
# SCHEMA
# ============================================================

class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = Field(default=None, description="Person's name as said in the transcript")
    deadline: Optional[str] = Field(default=None, description="Deadline phrase as spoken, e.g. 'by Sep 20'")
    priority: Literal["Low", "Medium", "High"] = "Medium"
    context: str = Field(description="One sentence of surrounding context / why this matters")
    source_quote: str = Field(description="The transcript snippet this was extracted from")


class Decision(BaseModel):
    decision: str
    made_by: Optional[str] = None
    context: str


class Blocker(BaseModel):
    description: str
    raised_by: Optional[str] = None
    affects: Optional[str] = Field(default=None, description="Task/area this blocks")
    severity: Literal["Low", "Medium", "High"] = "Medium"


class ExtractionOutput(BaseModel):
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    blockers: List[Blocker] = Field(default_factory=list)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    meeting_title: str
    meeting_date: str
    transcript: str

    action_items: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    blockers: List[Dict[str, Any]]

    approved_action_items: List[Dict[str, Any]]
    created_tickets: List[Dict[str, Any]]

    final_response: str


def make_initial_state(meeting_title: str, meeting_date: str, transcript: str) -> State:
    """Convenience constructor for the initial state passed to invoke(...)."""
    return {
        "messages": [],
        "meeting_title": meeting_title,
        "meeting_date": meeting_date,
        "transcript": transcript,
        "action_items": [],
        "decisions": [],
        "blockers": [],
        "approved_action_items": [],
        "created_tickets": [],
        "final_response": "",
    }


# ============================================================
# EXTRACTION NODE (LLM, runs once)
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
You are a meeting analysis agent. You will be given a raw transcript of a
meeting.

Extract three things, and ONLY things that were actually said:

1. action_items — concrete tasks someone committed to. For each, capture
   the task, the owner (person's name as said in the transcript, or null
   if unclear), a deadline if one was mentioned (keep the phrasing the
   speaker used, e.g. "by Sep 20" or "next Friday"), a priority you infer
   from urgency/tone, a one-sentence context, and the exact transcript
   snippet ("source_quote") that supports the extraction.

2. decisions — explicit decisions the group reached (e.g. "we're going
   with Qt instead of TILCON"). Include who made/announced it if clear.

3. blockers — anything described as blocking, unresolved, or repeatedly
   causing problems. Include severity based on language used ("critical",
   "still stuck", etc.) and what it affects if stated.

Rules:
- Do not invent tasks, owners, or deadlines that are not in the text.
- If the owner or deadline is not stated, leave it null — do not guess.
- Keep source_quote short (one sentence) and verbatim from the transcript.
- If nothing qualifies for a category, return an empty list for it.
"""


def _extractor(state: State) -> Dict[str, Any]:
    # Plain model: no tools bound, so structured output works reliably.
    structured_llm = llm.with_structured_output(ExtractionOutput)

    result = structured_llm.invoke(
        [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Meeting: {state['meeting_title']} ({state['meeting_date']})\n\n"
                    f"Transcript:\n{state['transcript']}"
                )
            ),
        ]
    )

    action_items = []
    for i, item in enumerate(result.action_items, start=1):
        d = item.model_dump()
        d["id"] = f"task_{i}"
        d["status"] = "Not Started"
        action_items.append(d)

    return {
        "action_items": action_items,
        "decisions": [d.model_dump() for d in result.decisions],
        "blockers": [b.model_dump() for b in result.blockers],
    }


# ============================================================
# HUMAN APPROVAL (interrupt — pauses the graph, resumes on human input)
# ============================================================

def _human_approval(state: State) -> Dict[str, Any]:
    """Pauses for a human to review/edit/accept the extracted action items
    before anything gets written to Jira/ADO. Resume with:

        Command(resume={"approved_ids": [...], "edits": {...}})

    - approved_ids: task ids the human wants ticketed
    - edits: optional {task_id: {field: new_value}} overrides applied
      before ticket creation (e.g. fixing a misheard owner name)

    Keep this node free of side effects before interrupt(): it re-runs
    from the top when the graph resumes.
    """
    decision = interrupt(
        {
            "type": "approval_request",
            "message": "Review the extracted action items before ticket creation.",
            "action_items": state["action_items"],
            "decisions": state["decisions"],
            "blockers": state["blockers"],
        }
    )

    approved_ids = set(decision.get("approved_ids", []))
    edits = decision.get("edits", {})

    approved = []
    for item in state["action_items"]:
        if item["id"] not in approved_ids:
            continue
        merged = {**item, **edits.get(item["id"], {})}
        approved.append(merged)

    return {"approved_action_items": approved}


# ============================================================
# TICKET CREATION (stub — swap for real Jira/Azure DevOps client)
# ============================================================

@tool
def create_ticket(task: str, owner: Optional[str], deadline: Optional[str], priority: str) -> str:
    """Create a Jira/Azure DevOps ticket for a single approved action item."""
    ticket_id = f"PROJ-{uuid.uuid4().hex[:5].upper()}"
    return f"[stub] Created {ticket_id}: '{task}' (owner={owner}, deadline={deadline}, priority={priority})"


def _create_tickets(state: State) -> Dict[str, Any]:
    created = []
    for item in state["approved_action_items"]:
        result = create_ticket.invoke(
            {
                "task": item["task"],
                "owner": item.get("owner"),
                "deadline": item.get("deadline"),
                "priority": item.get("priority", "Medium"),
            }
        )
        created.append({"task_id": item["id"], "result": result})
    return {"created_tickets": created}


# ============================================================
# FINAL SUMMARY
# ============================================================

def _final_agent(state: State) -> Dict[str, Any]:
    lines = [f"Meeting analyzed: {state['meeting_title']} ({state['meeting_date']})", ""]

    lines.append(f"Action items found: {len(state['action_items'])}")
    lines.append(f"Approved & ticketed: {len(state.get('created_tickets', []))}")
    for t in state.get("created_tickets", []):
        lines.append(f"  - {t['task_id']}: {t['result']}")

    if state["decisions"]:
        lines.append("\nDecisions:")
        for d in state["decisions"]:
            lines.append(f"  - {d['decision']}")

    if state["blockers"]:
        lines.append("\nBlockers:")
        for b in state["blockers"]:
            lines.append(f"  - [{b['severity']}] {b['description']}")

    return {"final_response": "\n".join(lines)}


# ============================================================
# PUBLIC FACTORY
# ============================================================

def build_meeting_task_agent(checkpointer=None):
    """Compiles and returns the graph.

    checkpointer=None (default): the graph inherits the parent graph's
    checkpointer when used as a subgraph (what the Manager AI does).
    For standalone use, pass MemorySaver() or a persistent saver — the
    approval interrupt cannot work without one.
    """
    graph = StateGraph(State)

    graph.add_node("extractor", _extractor)
    graph.add_node("human_approval", _human_approval)
    graph.add_node("create_tickets", _create_tickets)
    graph.add_node("final_agent", _final_agent)

    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "human_approval")
    graph.add_edge("human_approval", "create_tickets")
    graph.add_edge("create_tickets", "final_agent")
    graph.add_edge("final_agent", END)

    return graph.compile(checkpointer=checkpointer)