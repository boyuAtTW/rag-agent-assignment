import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# Setup the DB connection for the tool
client = chromadb.PersistentClient(path="./db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection("kb", embedding_function=emb_fn)

# The Agent definition
agent = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="You are a helpful assistant. Use 'search_kb' to find facts before answering.",
)


@agent.tool
def search_kb(ctx: RunContext[None], query: str) -> str:
    """Search the knowledge base for relevant facts."""
    results = collection.query(query_texts=[query], n_results=2)
    return "\n".join(results["documents"][0])
