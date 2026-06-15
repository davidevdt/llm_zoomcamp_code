### RAG System 
from pydantic_ai import Agent

def search_tool(query: str) -> str:
    results = index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

    return "\n".join(
        f"{r['section']}\nQ: {r['question']}\nA: {r['answer']}"
        for r in results
    )

# or with ElasticSearch 
def search_faq(query: str, course: str | None = None):
    return search(es, query, course)

# BM25, vector search 
@agent.tool
def bm25_search(query: str):
    return es_bm25(query)

@agent.tool
def vector_search(query: str):
    return es_vector(query)

@agent.tool
def hybrid_search(query: str):
    return {
        "bm25": es_bm25(query),
        "vector": es_vector(query)
    }


# Agent (= prompt + llm)
agent = Agent(
    model="openai:gpt-5.4-mini",
    instructions=INSTRUCTIONS,
    tools=[search_tool]
)

# RAG Execution 
def rag(query: str):
    return await agent.run(query)


### Agentic RAG 
#Tool Definition 
from pydantic_ai import Agent

@agent.tool 
def search(query: str) -> str:
    """Docstrings are not mandatory but highly recommended."""
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

# Agent Setup 
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-5.4-mini",
    instructions=instructions,
    tools=[search],
)

# Async version
result = await agent.run("How do I run Ollama locally?")

# Sync version
result_sync = agent.run_sync("What is the capital of Italy?")
print(result.output)

# run from program
async def run_agent():
    agent_task = asyncio.create_task(
        agent.run(question)
    )
    await agent_task

asyncio.run(run_agent()) 


# Conversation memory
result1 = agent.run("How do I run Ollama locally?")

result2 = agent.run(
    "How do I run a different model?",
    message_history=result1.all_messages()
)

# token usage 
result.usage()

### Guardrails 
# Not really implemented but a code block similar to the guarded agent of the workshop can integrate a pydantic agent: 
async def run(question: str):
    decision = input_guardrail(question)

    if decision.fail:
        return f"[BLOCKED] {decision.reasoning}"

    answer = await agent.run(question) # -> Pydantic agent

    output_decision = output_guardrail(answer.output)

    if output_decision.fail:
        return "[OUTPUT BLOCKED]"

    return answer.output

