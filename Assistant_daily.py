# Assistant_daily.py
# Local Orchestrator Assistant — plan, human gate, then specialists

import os
from datetime import datetime

from crewai import Crew, Process, Task

from Assistant_agents import orchestrator, researcher, writer, report_writer

today_date = datetime.now().strftime("%Y-%m-%d")
daily_folder = f"./daily_artifacts/{today_date}"
os.makedirs(daily_folder, exist_ok=True)

# --- Phase 1: Orchestrator plans the swarm ---
plan_task = Task(
    description=(
        "Human goal:\n{goal}\n\n"
        "Design a swarm plan using only: Researcher, Writer, Validation Analyst. "
        "Write task descriptions those agents should follow. "
        "Call out any information the human should provide before execution."
    ),
    expected_output="Markdown swarm plan with objective, agents, ordered tasks, human checkpoints, done definition",
    agent=orchestrator,
)

# --- Phase 2: Specialists execute (use plan + goal) ---
research_task = Task(
    description=(
        "Human goal:\n{goal}\n\n"
        "Orchestrator plan:\n{plan}\n\n"
        "Execute the research portion of the plan. "
        "If the plan has no research, provide a short note and 3 key context bullets."
    ),
    expected_output="Research bullets with sources when possible",
    agent=researcher,
    context=[plan_task],
)

write_task = Task(
    description=(
        "Human goal:\n{goal}\n\n"
        "Orchestrator plan:\n{plan}\n\n"
        "Research:\n{research}\n\n"
        "Execute the writing portion of the plan. "
        "Deliver only the requested artifact(s)."
    ),
    expected_output="Writer deliverable only",
    agent=writer,
    context=[plan_task, research_task],
)

validate_task = Task(
    description=(
        "Human goal:\n{goal}\n\n"
        "Orchestrator plan:\n{plan}\n\n"
        "Produce the validation / next-steps report per the plan and your role."
    ),
    expected_output="Markdown validation report",
    agent=report_writer,
    context=[plan_task, research_task, write_task],
)

def run_planner(goal: str) -> str:
    plan_crew = Crew(
        agents=[orchestrator],
        tasks=[plan_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )
    return str(plan_crew.kickoff(inputs={"goal": goal}))

def run_execution(goal: str, plan: str) -> str:
    # Inject plan into specialist tasks via inputs + description placeholders
    research_task.description = (
        f"Human goal:\n{goal}\n\nOrchestrator plan:\n{plan}\n\n"
        "Execute the research portion of the plan. "
        "If none, give a short note and 3 context bullets."
    )
    write_task.description = (
        f"Human goal:\n{goal}\n\nOrchestrator plan:\n{plan}\n\n"
        "Execute the writing portion of the plan. Output only the artifact(s)."
    )
    validate_task.description = (
        f"Human goal:\n{goal}\n\nOrchestrator plan:\n{plan}\n\n"
        "Produce the validation / next-steps report."
    )

    exec_crew = Crew(
        agents=[researcher, writer, report_writer],
        tasks=[research_task, write_task, validate_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )
    return str(exec_crew.kickoff(inputs={"goal": goal, "plan": plan}))

if __name__ == "__main__":
    print("=" * 60)
    print("LOCAL ORCHESTRATOR ASSISTANT (Ollama)")
    print("=" * 60)

    user_goal = input("\nGoal: ").strip()
    if not user_goal:
        user_goal = "Plan and draft an approach for a local agentic project assistant"

    print("\n--- Phase 1: Orchestrator planning ---\n")
    plan = run_planner(user_goal)
    print(plan)

    plan_path = f"{daily_folder}/swarm_plan_{datetime.now().strftime('%H-%M')}.md"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(f"# Swarm Plan\n\nGoal: {user_goal}\n\n{plan}\n")
    print(f"\nPlan saved: {plan_path}")

    # Human-in-the-loop gate
    print("\n" + "-" * 60)
    extra = input("Additional human input for the team (or Enter to skip): ").strip()
    if extra:
        plan = plan + f"\n\n## Human input\n{extra}\n"

    confirm = input("Run specialist swarm on this plan? (y/n): ").strip().lower()
    if confirm != "y":
        print("Stopped after planning. Edit the plan file and re-run when ready.")
        raise SystemExit(0)

    print("\n--- Phase 2: Specialist execution ---\n")
    result = run_execution(user_goal, plan)

    out_path = f"{daily_folder}/assistant_output_{datetime.now().strftime('%H-%M')}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Assistant Output\n\nGoal: {user_goal}\n\n## Plan\n{plan}\n\n## Result\n{result}\n")
    print(f"\nSaved: {out_path}")
    print("Done.")
