## Journal

a. **Init project**, this creates `pyproject.toml` and `.python-version`

```shell
uv init
```

b. **Add dependencies**

```shell
uv add pydantic-ai google-genai chromadb sentence-transformers python-dotenv
```

c. **Setup project structure**

```plaintext
rag-agent-assignment/
├── .env                # My API keys (Don't commit this!)
├── data/               # My local .txt or .md files
├── db/                 # Where ChromaDB will store its vectors
├── src/
│   ├── agent.py        # The Pydantic AI agent logic
│   └── ingest.py       # Script to load data into the Vector DB
├── main.py             # Entry point to run the agent
└── pyproject.toml      # Project metadata & dependencies
```

+ Configure VS Code python settings: `"python.terminal.useEnvFile": true`
+ Add `.env` to `.gitignore`

d. **Add knowledge**: `data/knowledge.txt`

e. **The ingest script** (`src/ingest.py`)

```python
import chromadb
from chromadb.utils import embedding_functions


def run_ingestion():
    client = chromadb.PersistentClient(path="./db")  # Save data to disk

    # Use a local embedding model (Requirement of your assignment)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection("kb", embedding_function=emb_fn)

    # Example: Load your local file
    with open("data/knowledge.txt", "r") as f:
        text = f.read()

    # Simple chunking by paragraph
    chunks = [c for c in text.split("\n\n") if c.strip()]

    collection.add(documents=chunks, ids=[f"id_{i}" for i in range(len(chunks))])
    print(f"✅ Ingested {len(chunks)} chunks into the database.")


if __name__ == "__main__":
    run_ingestion()

```

f. **The agent script** (`src/agent.py`)

```python
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

```

g. Ingest the data

```shell
uv run python src/ingest.py
```

I ran into `chromadb.errors.InvalidArgumentError`.
![chromadb.errors.InvalidArgumentError](assets/images/chromadb.errors.InvalidArgumentError.png)
