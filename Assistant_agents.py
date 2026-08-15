# Assistant_agents.py
# Local Orchestrator + specialist pool (Ollama)

from crewai import Agent, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

local_llm = LLM(
    model="ollama/llama3.2",  # match: ollama list
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=600,
)

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for recent statistics, trends, and sources."""
    return DuckDuckGoSearchRun().run(query)

# ----- Orchestrator (plans the swarm; does not do all the work alone) -----
orchestrator = Agent(
    role="Chief Swarm Orchestrator",
    goal=(
        "Turn the human goal into a clear swarm plan: which specialists to use, "
        "task sequence, success criteria, and any human checkpoints"
    ),
    backstory=(
        "You are the lead orchestrator for a local agentic system. "
        "You do NOT invent cloud APIs or leave the local workflow. "
        "You design work for this fixed specialist pool only: "
        "Researcher, Writer, Validation Analyst. "
        "Output a concise markdown plan with: "
        "1) Objective restatement "
        "2) Agents to engage (subset of the pool) "
        "3) Ordered tasks (what each agent must produce) "
        "4) Suggested task descriptions the specialists will follow "
        "5) Human input needed (if any) "
        "6) Done definition. "
        "No JSON. No tool traces. No long essays."
    ),
    llm=local_llm,
    tools=[],
    allow_delegation=False,  # plan only in sequential mode; see hierarchical note below
    verbose=True,
    max_iter=6,
)

researcher = Agent(
    role="R&D Intelligence Researcher",
    goal="Execute research tasks assigned by the orchestrator plan for the human goal",
    backstory=(
        "Specialist researcher. Follow the orchestrator task description closely. "
        "Use search when needed. Prefer 2025–2026 sources. "
        "Bullets only. No meta-commentary."
    ),
    llm=local_llm,
    tools=[duckduckgo_search],
    allow_delegation=False,
    verbose=False,
    max_iter=6,
)

writer = Agent(
    role="Content and Documentation Writer",
    goal="Execute writing tasks assigned by the orchestrator plan",
    backstory=(
        "Specialist writer. Follow the orchestrator task description exactly. "
        "Output ONLY the artifact requested (posts, brief, outline, spec section). "
        "No JSON, no planning narration."
    ),
    llm=local_llm,
    tools=[],
    allow_delegation=False,
    verbose=False,
    max_iter=6,
)

report_writer = Agent(
    role="Validation and Next-Steps Analyst",
    goal="Execute validation/reporting tasks assigned by the orchestrator plan",
    backstory=(
        "Specialist analyst. Third-person markdown: Executive Summary, Score (0-100), "
        "Findings, Risks, Recommendations, Bottom Line. "
        "Ground in research/writer outputs. No fluff."
    ),
    llm=local_llm,
    tools=[],
    allow_delegation=False,
    verbose=False,
    max_iter=6,
)
