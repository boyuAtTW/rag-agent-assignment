import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel, Field  # Add these
from pydantic_ai import Agent, RunContext

load_dotenv()


# Define the structure the agent MUST follow
class AgentResponse(BaseModel):
    answer: str = Field(description="The final answer to the user's question.")
    source_snippet: str = Field(
        description="The specific text used from the knowledge_base to answer."
    )


# Setup the DB connection for the tool
client = chromadb.PersistentClient(path="./db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection("knowledge_base", embedding_function=emb_fn)

# The Agent definition
agent = Agent(
    "google-gla:gemini-2.0-flash",
    output_type=AgentResponse,  # Tell the agent to use the model above
    system_prompt=(
        "You are a helpful assistant. Use 'search_kb' to find facts before answering.",
        "You must provide the answer and the exact snippet of text you used.",
    ),
)


@agent.tool
def search_kb(ctx: RunContext[None], query: str) -> str:
    """Search the knowledge base for relevant facts."""
    results = collection.query(query_texts=[query], n_results=2)

    # Check if we actually found anything
    if not results["documents"] or not results["documents"][0]:
        return "No relevant information found in the knowledge base."

    # Joining with clear separators helps the LLM distinguish between chunks
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        formatted_results.append(f"Source {i + 1}:\n{doc}")

    return "\n\n---\n\n".join(formatted_results)
