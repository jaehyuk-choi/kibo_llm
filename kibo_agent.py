#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# =========================
# 0) 환경변수 & LLM 설정
# =========================
load_dotenv()

ollama_llm = LLM(
    model=os.getenv("CREWAI_MODEL", "ollama/llama3.1"),
    base_url=os.getenv("CREWAI_API_BASE", "http://localhost:11434"),
    api_key=os.getenv("CREWAI_API_KEY", "none")
)

print("=====================================")
print("1. Provider:", os.getenv("CREWAI_PROVIDER"))
print("2. Model:", os.getenv("CREWAI_MODEL"))
print("3. API Base:", os.getenv("CREWAI_API_BASE"))
print("4. API Key:", os.getenv("CREWAI_API_KEY"))
print("=====================================\n")

# =========================
# 1) 에이전트 정의
# =========================
supervisor = Agent(
    role="Supervisor",
    goal="Analyze the user's query and route it to the appropriate specialized agent.",
    backstory="Central orchestrator for the KIBO technology evaluation workflow; it determines which agent should handle the query.",
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm
)

basic_agent = Agent(
    role="Basic Info Agent",
    goal="Provide concise and accurate explanations about technologies or terms.",
    backstory="Specializes in explaining technologies and concepts clearly and simply for evaluators.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

code_agent = Agent(
    role="Industry & Tech Code Agent",
    goal="Recommend appropriate industry and technology classification codes (KSIC, NTIS).",
    backstory="Expert in Korean standard industrial classification and national science technology taxonomy.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

onsite_agent = Agent(
    role="On-site Evaluation Support Agent",
    goal="Provide likely on-site inspection Q&A and review checklists for evaluators.",
    backstory="A virtual assistant trained on past KIBO inspection reports to generate evaluation questions and answers for technology, market, and business aspects.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

draft_agent = Agent(
    role="Evaluation Draft Writer Agent",
    goal="Generate structured draft opinions for technology evaluation reports.",
    backstory="Produces professional evaluation drafts covering technical, market, and business feasibility, and synthesizes them into a comprehensive conclusion.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# DRAFT 작성 위해 2단계 실행
researcher = Agent(
    role="Researcher",
    goal="Gather accurate, up-to-date insights about the target technology and its market trends.",
    backstory="A senior research analyst at KIBO who analyzes industry trends, extracts technology insights, and finds verified information from reports, patents, and market analyses with strong focus on accuracy and relevance.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

writer = Agent(
    role="Writer",
    goal="Write a clear, structured, and professional technology evaluation summary based on the research findings.",
    backstory="An experienced technology evaluation report writer for KIBO who transforms research data into concise and objective summaries aligned with internal evaluation documents.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# =========================
# 2) Supervisor Task (의도 분류를 JSON으로 강제)
# =========================
supervisor_task = Task(
    description="Given the user query, identify the intent and respond ONLY as a one-line JSON like {\"agent\":\"BASIC|CODE|ONSITE|DRAFT\",\"reason\":\"short reason\"}. Valid agents are BASIC for simple explanations, CODE for classification codes, ONSITE for inspection Q&A, DRAFT for evaluation draft writing.",
    agent=supervisor,
    expected_output="A single-line JSON object with keys 'agent' and 'reason' such as {\"agent\":\"CODE\",\"reason\":\"Asks for KSIC classification\"}."
)

# =========================
# 3) 의도별 Task 생성기
# =========================
def build_task_for_intent(intent: str, user_query: str) -> tuple[Agent, Task, list[Agent], list[Task]]:
    """
    intent: BASIC | CODE | ONSITE | DRAFT
    returns: (primary_agent, primary_task, extra_agents, extra_tasks)
    - BASIC/CODE/ONSITE: 단일 Task 실행
    - DRAFT: Researcher -> Writer 순차 2단계
    """
    intent = intent.upper().strip()

    if intent == "BASIC":
        task = Task(
            description=f"Answer the following question with a concise, accurate explanation suitable for technology evaluators: {user_query}",
            agent=basic_agent,
            expected_output="A short, factual, and evaluator-friendly explanation in 1-3 paragraphs."
        )
        return basic_agent, task, [], []

    if intent == "CODE":
        task = Task(
            description=f"Recommend relevant industry and technical classification codes (e.g., KSIC, NTIS) for the following description and justify briefly: {user_query}",
            agent=code_agent,
            expected_output="A one-paragraph answer listing the most probable KSIC and technical codes with a brief justification."
        )
        return code_agent, task, [], []

    if intent == "ONSITE":
        task = Task(
            description=f"Provide likely on-site inspection Q&A and a compact checklist across technical, market, and business aspects for: {user_query}",
            agent=onsite_agent,
            expected_output="3-5 suggested questions with brief model answers plus a compact checklist."
        )
        return onsite_agent, task, [], []

    # DRAFT: Researcher -> Writer 2단계
    research_task = Task(
        description=f"Research the topic and context implied by the user's request focusing on technology, market, competition, and risks: {user_query}",
        agent=researcher,
        expected_output="A bullet-point research brief covering technology overview, differentiators, market context, competitors, and key risks."
    )
    writing_task = Task(
        description="Using the previous research brief, write a professional evaluation draft covering technical feasibility, marketability, business feasibility, and a concise overall opinion.",
        agent=writer,
        expected_output="A polished multi-paragraph draft aligned with KIBO evaluation tone including technical, market, business sections and a succinct overall opinion."
    )
    return draft_agent, None, [researcher, writer], [research_task, writing_task]

# =========================
# 4) Supervisor 출력 파서
# =========================
def parse_supervisor_decision(text: str) -> str:
    """Supervisor 결과에서 agent 코드를 추출(BASIC|CODE|ONSITE|DRAFT). JSON 우선, 키워드 백업."""
    # JSON 시도
    try:
        j = json.loads(text.strip())
        agent = j.get("agent", "")
        if agent and agent.upper() in {"BASIC", "CODE", "ONSITE", "DRAFT"}:
            return agent.upper()
    except Exception:
        pass
    # 키워드 백업
    m = re.search(r"(BASIC|CODE|ONSITE|DRAFT)", text.upper())
    return m.group(1) if m else "BASIC"

# =========================
# 5) 메인 실행
# =========================
if __name__ == "__main__":
    print("Running KIBO Multi-Agent System (Supervisor → Auto-Execute)...\n")
    user_query = input("User query >> ").strip()

    # 1) Supervisor에게 의도 판단 요청
    sup_crew = Crew(agents=[supervisor], tasks=[supervisor_task], verbose=True, process=Process.sequential)
    supervisor_task.description = f"Given the user query, identify the intent and respond ONLY as a one-line JSON like {{\"agent\":\"BASIC|CODE|ONSITE|DRAFT\",\"reason\":\"short reason\"}}. Query: {user_query}"
    sup_result = sup_crew.kickoff()
    print("\n === Supervisor Decision Raw ===")
    print(sup_result)

    intent = parse_supervisor_decision(str(sup_result))
    print(f"\n Parsed intent: {intent}")

    # 2) 의도별 Task 생성 및 실행
    primary_agent, primary_task, extra_agents, extra_tasks = build_task_for_intent(intent, user_query)

    if intent == "DRAFT":
        # Researcher -> Writer 2단계 실행
        draft_crew = Crew(agents=extra_agents, tasks=extra_tasks, verbose=True, process=Process.sequential)
        print("\n Running Researcher → Writer pipeline...\n")
        draft_result = draft_crew.kickoff()
        print("\n === Final Draft ===\n")
        print(draft_result)
    else:
        # 단일 에이전트 실행
        run_crew = Crew(agents=[primary_agent], tasks=[primary_task], verbose=True, process=Process.sequential)
        print(f"\n Running {primary_agent.role}...\n")
        final = run_crew.kickoff()
        print("\n === Result ===\n")
        print(final)
