# pip install pydantic-ai python-dotenv tavily-python openai

# Simple rag pipeline 

# --> retriever: the rag function
def search(query):

    return index.search(
        query,
        num_results=5,
        boost_dict={
            "question":3.0,
            "section":0.5
        },
        filter_dict={
            "course":"llm-zoomcamp"
        }
    )


from pydantic_ai import Agent


rag_agent = Agent(
    "openai:gpt-5.4-mini",
    system_prompt="""
You are a course assistant.

Answer using only the provided context.

If the answer is not in the context:
say "I don't know".
"""
)


def rag(question):

    docs = search(question)


    context = "\n\n".join(
        [
            f"""
            {d['section']}

            Q: {d['question']}

            A: {d['answer']}
            """
            for d in docs
        ]
    )


    prompt = f"""
Context:

{context}


Question:

{question}
"""

    result = rag_agent.run_sync(prompt)
    return result.output


answer = rag(
    "How do I run another model?"
)

print(answer)


# with token + usage cost 
result = rag_agent.run_sync(prompt)


print(
    result.usage()
)


def cost(usage):

    input_price = 0.15 / 1_000_000
    output_price = 0.60 / 1_000_000


    return (
        usage.request_tokens * input_price
        +
        usage.response_tokens * output_price
    )



print(
    cost(result.usage())
)


### Agentic RAG + web search 
from pydantic_ai import Agent, RunContext


agent = Agent(
    "openai:gpt-5.4-mini",

    system_prompt="""
You are a helpful assistant.

Use course_search for course questions.

Use web_search for external/current information.
"""
)


@agent.tool
def course_search(ctx: RunContext, query: str):

    """
    Search course documents.
    """

    return search(query)


from tavily import TavilyClient


tavily = TavilyClient()


@agent.tool
def web_search(
    ctx: RunContext,
    query: str
):

    """
    Search the internet.
    """

    return tavily.search(
        query=query
    )


result = agent.run_sync(
    "What is the latest OpenAI model?"
)


print(result.output)


### memory state
messages = result.all_messages()

# so first run: 
result = agent.run_sync(
    "How do I use LangChain agents?"
)


history = result.all_messages()

# second run: 
result2 = agent.run_sync(
    "Explain that with code",

    message_history=history
)


print(result2.output)

### Structured output
from pydantic import BaseModel


class Evaluation(BaseModel):

    correct: bool
    score: int
    explanation: str


evaluator = Agent(
    "openai:gpt-5.4-mini",

    output_type=Evaluation,

    system_prompt="""
Evaluate the answer.
"""
)

result = evaluator.run_sync(
"""
Context:

Paris is the capital of France.


Answer:

Paris is in France.
"""
)


print(result.output)

# Returns: 
# Evaluation(
#     correct=True,
#     score=5,
#     explanation="The answer matches the context."
# )


### Evaluation + structured output 
answer = agent.run_sync(
    question
)

evaluation = evaluator.run_sync(
f"""
Question:

{question}


Answer:

{answer.output}
"""
)
print(evaluation.output)



### Add state/dependencies: 
# --> Give tools access to application state

from dataclasses import dataclass

@dataclass
class AppState:

    index: object
    user_id: str


agent = Agent(
    "openai:gpt-5.4-mini",
    deps_type=AppState
)


@agent.tool
def course_search(
    ctx: RunContext[AppState],
    query:str
):
    return ctx.deps.index.search(query)


result = agent.run_sync(
    "Explain embeddings",
    deps=AppState(
        index=index,
        user_id="123"
    )
)


# async runs: 
import asyncio


async def main():

    result = await agent.run(
        "Explain embeddings",
        deps=deps
    )

    print(result.output)

asyncio.run(main())



# async tools: 
@agent.tool
async def search_course(
    ctx: RunContext[Deps],
    query: str
):

    results = await ctx.deps.index.search(query)

    return results


# multiple agents concurrently: 
result1 = await evaluator.run(...)
result2 = await safety.run(...)
result3 = await summarizer.run(...)

import asyncio


results = await asyncio.gather(

    evaluator.run(
        "Evaluate this answer"
    ),

    safety.run(
        "Check this answer"
    ),

    summarizer.run(
        "Summarize this answer"
    )
)


# with fastapi: 
from fastapi import FastAPI


app = FastAPI()


@app.post("/chat")
async def chat(question: str):

    result = await agent.run(
        question,
        deps=deps
    )

    return {
        "answer": result.output
    }



### multi agent (agent-as-a-tool)
from fastapi import FastAPI


app = FastAPI()


@app.post("/chat")
async def chat(question: str):

    result = await agent.run(
        question,
        deps=deps
    )

    return {
        "answer": result.output
    }


main_agent = Agent(
    "openai:gpt-5.4-mini",

    system_prompt="""
You are the main assistant.

Use the RAG agent for course questions.

Use the research agent for external questions.
"""
)


@main_agent.tool
async def ask_rag_agent(
    query: str
):

    """
    Ask the course RAG specialist.
    """

    result = await rag_agent.run(
        query
    )

    return result.output


@main_agent.tool
async def ask_research_agent(
    query: str
):

    """
    Ask the research specialist.
    """

    result = await research_agent.run(
        query
    )

    return result.output


result = await main_agent.run(
    "How do embeddings work?"
)


