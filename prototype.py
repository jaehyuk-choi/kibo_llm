#!/usr/bin/env python
# -*- coding: utf-8 -*-

from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Ollama LLM 구성 ----
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

researcher = Agent(
    role="Researcher",
    goal="Gather accurate, up-to-date insights about the target technology and its market trends.",
    backstory=(
        "A senior research analyst at the Korea Technology Finance Corporation (KIBO). "
        "This agent specializes in analyzing industry trends, extracting technology insights, "
        "and finding verified information from reports, patents, and market analyses. "
        "The Researcher focuses on factual accuracy, objectivity, and relevance to technology evaluation."
    ),
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,   
)

writer = Agent(
    role="Writer",
    goal="Write a clear, structured, and professional technology evaluation summary based on the research findings.",
    backstory=(
        "An experienced technology evaluation report writer who has prepared hundreds of professional reports for KIBO. "
        "This agent transforms raw research data into concise, well-structured summaries suitable for internal technical "
        "evaluation documents. The tone should be formal, factual, and objective."
    ),
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,  
)

research_task = Task(
    description=(
        "Research the topic of 'AI-driven technology evaluation systems in financial institutions'. "
        "Focus on how generative AI and LLMs are being used for assessing technology, automating evaluation reports, "
        "and improving decision-making accuracy."
    ),
    agent=researcher,
    expected_output=(
        "A detailed bullet-point summary covering:\n"
        "• Use cases of AI in technology evaluation\n"
        "• Benefits and limitations observed in real-world financial or R&D organizations\n"
        "• Emerging research or pilot projects related to AI-driven evaluation systems\n"
        "• Relevant examples or benchmarks from Korea or global institutions"
    )
)

writing_task = Task(
    description=(
        "Using the findings from the Researcher, write a summary report describing how AI is transforming "
        "technology evaluation in financial and public institutions. Emphasize the implications for KIBO and "
        "how such systems can enhance efficiency, consistency, and insight depth."
    ),
    agent=writer,
    expected_output=(
        "A professional, well-written summary report in paragraph form that:\n"
        "• Synthesizes the research findings into a cohesive narrative\n"
        "• Maintains a formal and objective tone\n"
        "• Highlights key takeaways for policymakers or technical evaluators"
    )
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
    process=Process.sequential
)

if __name__ == "__main__":
    print("Running KIBO Technology Evaluation Crew (Ollama Mode)...\n")
    result = crew.kickoff()
    print("\n✅ === Final Report === ✅\n")
    print(result)
