# Assistant_agents.py
# Local Orchestrator + specialist pool (Ollama)

import os
import requests
import streamlit as st
from crewai import Agent, LLM
from crewai.tools import tool

def _secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

# Cloud LLM (Groq)
local_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=_secret("GROQ_API_KEY"),
    temperature=0.1,
)

# --- Lightweight DuckDuckGo search ---
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time market signals."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": query, "kl": "us-en"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params=params,
            headers=headers,
            timeout=10
        )
        if resp.status_code != 200:
            return f"Search unavailable (HTTP {resp.status_code}). Use general knowledge."

        from html.parser import HTMLParser

        class SnippetParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._capture = False
                self._current = []

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self._capture = True
                    self._current = []

            def handle_endtag(self, tag):
                if self._capture and tag == "a":
                    self.results.append("".join(self._current).strip())
                    self._capture = False

            def handle_data(self, data):
                if self._capture:
                    self._current.append(data)

        parser = SnippetParser()
        parser.feed(resp.text)
        snippets = parser.results[:10]

        if not snippets:
            return "No results found. Use your general knowledge of 2026 market conditions."

        return "\n".join(f"- {s}" for s in snippets)

    except requests.Timeout:
        return "Search timed out. Use your general knowledge of 2026 market conditions."
    except Exception as e:
        return f"Search error: {str(e)}. Use your general knowledge of 2026 market conditions."


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
