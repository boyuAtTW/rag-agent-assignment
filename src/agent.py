import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry

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
        "Rules:\n"
        "1. ONLY answer using facts from 'search_kb'.\n"
        "2. If 'search_kb' returns 'No relevant information', state that you do not know.\n"
        "3. NEVER use your own training data to invent project details.",
    ),
    retries=3,  # Allow the agent to try again if it fails validation
)


# 2. The Result Validator (The Hallucination Guard)
@agent.output_validator
def validate_result(ctx: RunContext[None], output: AgentResponse) -> AgentResponse:
    """Checks if the AI's answer is actually supported by the source snippet."""
    # If the AI says it found something, but the source says "No relevant information"
    if "no relevant information" in output.source_snippet.lower():
        if (
            "not found" not in output.answer.lower()
            and "don't know" not in output.answer.lower()
        ):
            # This triggers a retry loop: the AI sees this message and corrects itself
            raise ModelRetry(
                "You provided an answer but the source snippet says no info was found. "
                "Please correct your answer to say you do not know."
            )
    return output


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
