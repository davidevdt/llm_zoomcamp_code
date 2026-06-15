### Langchain for Vector Databases 
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.documents import Document

docs = [
    Document(
        page_content=f"{d['question']} {d['answer']}",
        metadata={"course": d["course"]}
    )
    for d in faq_docs
]

vectorstore = ElasticsearchStore.from_documents(
    documents=docs,
    es_url="http://localhost:9200",
    index_name="faq"
)

# ElasticSearch is automatically persisted; for e.g. FAISS: 
# vectorstore.save_local("faq_index")

# lexical retrieval
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(docs)
retriever.get_relevant_documents("docker install")

# semantic retrieval 
retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("docker install")

# hybrid 
from langchain.retrievers import EnsembleRetriever

hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


### Static RAG System [using ElasticSearch]
# db search 
retriever = es_store.as_retriever(
    search_kwargs={
        "k": 5,
        "query": {
            "multi_match": {
                "query": "{query}",
                "fields": ["question^3", "section^0.5", "answer"]
            }
        }
    }
)

# Context builder: 
def format_docs(docs):
    return "\n\n".join(
        f"{d.metadata.get('section','')}\nQ: {d.page_content}"
        for d in docs
    )

# Prompt template: 
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", INSTRUCTIONS),
    ("user", "Question: {question}\n\nContext:\n{context}")
])

# LLM Call: 
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-mini")

chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
)

# LangchainRag: 
def rag(query: str):
    return chain.invoke(query)


### Agentic RAG 
from langchain_core.tools import tool

# Tool definition
@tool
def search(query: str) -> str:
    """Docstrings are not mandatory but highly recommended."""
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

# Agent Setup (ReAct agent)
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-5.4-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("user", "{input}")
])

# Agent Executor (the agent loop)
agent = create_openai_tools_agent(
    llm=llm,
    tools=[search],
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=[search],
    verbose=True
)

# Run (replaces run())
result = executor.invoke({
    "input": "How do I run Ollama locally?"
})

print(result["output"])

# Conversation memory 
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

executor = AgentExecutor(
    agent=agent,
    tools=[search],
    memory=memory,
    verbose=True
)

# token usage 
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = executor.invoke({"input": "hello"})

print(cb.total_tokens)
print(cb.total_cost)


### Guardrails 
# Input Guardrails (prerunnable)
def input_guardrail(x):
    if "pizza" in x["question"].lower():
        raise ValueError("Blocked by input guardrail")
    return x

chain = RunnableLambda(input_guardrail) | agent_chain # note: this is blocking, not async 

# Output guardrails (post-processing)
def output_guardrail(answer: str):
    if "harmful" in answer.lower():
        return "[BLOCKED]"
    return answer

chain = agent_chain | RunnableLambda(output_guardrail)

# Parallel guardrails 
from langchain_core.runnables import RunnableParallel
guardrails = RunnableParallel({
    "g1": guardrail1,
    "g2": guardrail2
})

chain = guardrails | agent_chain 