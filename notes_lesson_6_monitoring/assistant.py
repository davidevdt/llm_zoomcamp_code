import sys 
from dotenv import load_dotenv
from metrics import RAGWithMetrics
from ingest import load_faq_data, build_index 
from openai import OpenAI
from db_save import save_conversation

load_dotenv()


def create_assistant(): 
    documents = load_faq_data() 
    index = build_index(documents)

    return RAGWithMetrics(
        index=index, 
        llm_client=OpenAI() 
    )


if __name__ == "__main__": 
    assistant = create_assistant() 

    query = "How do I join the course?"
    if len(sys.argv) > 1: 
        query = sys.argv[1] 

    answer = assistant.rag(query)
    save_conversation(assistant.last_call, query, "llm-zoomcamp")
    
    print(answer)