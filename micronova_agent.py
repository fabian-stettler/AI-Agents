import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

# Load environment variables
load_dotenv(dotenv_path="/home/fabian/SpassProjekte/langchain_Tutorials/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = DEEP_SEEK_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# Initialize models
llm = init_chat_model("deepseek-chat", model_provider="deepseek")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector store
vector_store = Chroma(
    collection_name="Micronova",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# State definition
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Prompt
prompt = PromptTemplate.from_template(
    "Answer the following question with the given context\n\ncontext:\n{context}\n\nquestion:\n{question}\n\nanswer:"
)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
