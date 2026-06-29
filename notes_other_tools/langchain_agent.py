# pip install langchain langchain-openai langchain-community
# pip install tavily-python

from langchain_core.tools import tool


@tool
def course_search(query: str) -> dict:
    """
    Search the course FAQ database.
    Use this when you need information about the course.
    """

    results = index.search(
        query,
        num_results=5,
        boost_dict={
            "question": 3.0,
            "section": 0.5
        },
        filter_dict={
            "course": "llm-zoomcamp"
        }
    )

    return results


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0
)


instructions = """
You are a course teaching assistant.

Answer questions from students.

If you need information, use the search tool.

Make multiple searches if needed.

If information is unavailable say:
"I don't know".

At the end ask if the student wants to explore another topic.
"""

tools = [
    course_search
]


llm_with_tools = llm.bind_tools(tools)
from langchain.agents import create_agent


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=instructions
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How do I run a different model?"
            }
        ]
    }
)


print(
    response["messages"][-1].content
)


# Record all messages 
from langchain_core.chat_history import InMemoryChatMessageHistory


store = {}


def get_history(session_id):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


chat_agent = RunnableWithMessageHistory(
    agent_executor,

    get_history,

    input_messages_key="input",

    history_messages_key="chat_history"
)


# first message: 
result = chat_agent.invoke(
    {
        "input":
        "How do I run another model?"
    },

    config={
        "configurable":
        {
            "session_id":"student1"
        }
    }
)


print(result["output"])


# Then next message: 
result = chat_agent.invoke(
    {
        "input":
        "Can you explain that with an example?"
    },

    config={
        "configurable":
        {
            "session_id":"student1"
        }
    }
)


print(result["output"])


### Retrieve token usage 
from langchain_community.callbacks import get_openai_callback


with get_openai_callback() as cb:

    response = chat_agent.invoke(
        {
            "input":
            "Explain RAG"
        },

        config={
            "configurable":
            {
                "session_id":"student1"
            }
        }
    )


print(cb.total_tokens)
print(cb.prompt_tokens)
print(cb.completion_tokens)
print(cb.total_cost)


### Add web search tool 
from langchain_community.tools.tavily_search import TavilySearchResults


web_search = TavilySearchResults(
    max_results=3
)

tools = [
    course_search,
    web_search
]



# or: 
from langchain_community.tools.tavily_search import TavilySearchResults


tavily = TavilySearchResults(
    max_results=3
)
# results = tavily.invoke(
#     "latest LangChain agent documentation"
# )
# print(results)
from langchain_core.tools import tool
@tool
def internet_search(query: str):
    """
    Search the internet for information outside the course database.
    Use for current information, documentation, news, and external facts.
    """

    return tavily.invoke(query)




### Simple Chat with RAG retrieval (lesson 1)
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class CourseRetriever(BaseRetriever):

    index: object

    def _get_relevant_documents(self, query):

        results = self.index.search(
            query,
            num_results=5,
            boost_dict={
                "question": 3.0,
                "section": 0.5
            },
            filter_dict={
                "course": "llm-zoomcamp"
            }
        )


        return [
            Document(
                page_content=
                f"""
                {doc['section']}

                Q: {doc['question']}

                A: {doc['answer']}
                """
            )
            for doc in results
        ]
    


retriever = CourseRetriever(
    index=index
)


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0
)


from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
"""
You are a course teaching assistant.

Answer using only the context.

If the answer is not in the context:
say "I don't know".

Context:

{context}


Question:

{input}
"""
)


# The rag chain: 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


document_chain = create_stuff_documents_chain(
    llm,
    prompt
)


rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)


# Run: 
response = rag_chain.invoke(
    {
        "input":
        "How do I run another model?"
    }
)


print(response["answer"])


response["context"] # retrieved docs 


### Conversation memory for simple rag: 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
store = {}


def get_history(session_id):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]



chat_rag = RunnableWithMessageHistory(

    rag_chain,

    get_history,

    input_messages_key="input",

    history_messages_key="chat_history"
)


# first turn: 
chat_rag.invoke(
    {
        "input":
        "How do I run another model?"
    },
    config={
        "configurable":
        {
            "session_id":"student1"
        }
    }
)

# second turn: 
chat_rag.invoke(
    {
        "input":
        "Can you show code?"
    },
    config={
        "configurable":
        {
            "session_id":"student1"
        }
    }
)

# token usage: 
from langchain_community.callbacks import get_openai_callback


with get_openai_callback() as cb:

    result = rag_chain.invoke(
        {
            "input":
            "Explain embeddings"
        }
    )


print(cb.prompt_tokens)
print(cb.completion_tokens)
print(cb.total_tokens)
print(cb.total_cost)



### Structured output: 
from pydantic import BaseModel, Field


class RAGEvaluation(BaseModel):
    answer_correct: bool = Field(
        description="Whether the answer is supported by the context"
    )

    relevance_score: int = Field(
        description="Score from 1-5 indicating answer relevance"
    )

    explanation: str = Field(
        description="Short explanation of the evaluation"
    )


# example return: 
# {
#   "answer_correct": true,
#   "relevance_score": 5,
#   "explanation": "The answer is supported by the retrieved documents."
# }


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0
)


structured_llm = llm.with_structured_output(
    RAGEvaluation
)

# returns:
# RAGEvaluation(
#     answer_correct=True,
#     relevance_score=5,
#     explanation="..."
# )


# Call: 
result = structured_llm.invoke(
"""
Evaluate this RAG answer.

Context:
The Eiffel Tower is located in Paris.

Question:
Where is the Eiffel Tower?

Answer:
The Eiffel Tower is in Paris.
"""
)


print(result)


# Output: 
# RAGEvaluation(
#     answer_correct=True,
#     relevance_score=5,
#     explanation="The answer matches the provided context."
# )

# Extract one field: 
result.answer_correct

### Structured output in rag pipeline: 
answer = rag_chain.invoke(
    {
        "input":
        "Where is the Eiffel Tower?"
    }
)

evaluation = structured_llm.invoke(
f"""
Context:

{answer['context']}


Answer:

{answer['answer']}

Evaluate the answer.
"""
)


# {
# answer_correct: True,
# relevance_score: 5,
# explanation: "..."
# }




### Multi agent (agent-as-a-tool)
from langchain.agents import create_agent


rag_agent = create_agent(
    model=llm,
    tools=[course_search],
    system_prompt="Answer from course docs"
)


research_agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="Research external info"
)


from langchain_core.tools import tool


@tool
def ask_rag_agent(query: str):

    """
    Ask the RAG specialist.
    """

    result = rag_agent.invoke(
        {
            "messages":
            [
                {
                    "role":"user",
                    "content":query
                }
            ]
        }
    )

    return result["messages"][-1].content


@tool
def ask_research_agent(query: str):

    """
    Ask the research specialist.
    """

    result = research_agent.invoke(
        {
            "messages":
            [
                {
                    "role":"user",
                    "content":query
                }
            ]
        }
    )

    return result["messages"][-1].content



supervisor = create_agent(

    model=llm,

    tools=[
        ask_rag_agent,
        ask_research_agent
    ],

    system_prompt="""
You are a supervisor.

Delegate tasks to specialist agents.
"""
)



result = supervisor.invoke(
{
"messages":
[
{
"role":"user",
"content":
"What are the latest changes in LangChain?"
}
]
}
)



