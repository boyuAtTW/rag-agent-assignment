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

h. fix the `InvalidArgumentError`, it turns out the "kb" name is too short (2-char), there should be at least 3 chars. So rename it from "kb" to "knowledge_base"

```python
# ingest.py
collection = client.get_or_create_collection("knowledge_base", embedding_function=emb_fn)
# agent.py
collection = client.get_collection("knowledge_base", embedding_function=emb_fn)
```

i. delete the `data/chroma/sqlite3`, and re-run `uv run python src/ingest.py`
![successfully ingested the knowledge](assets/images/successfully-ingest.png)

j. `main.py` to interact with the agent

```python
from dotenv import load_dotenv
from src.agent import agent  # This imports the 'agent' VARIABLE from the file

load_dotenv()


def chat():
    print("--- Project Titan Assistant (Type 'exit' to quit) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # run_sync is perfect for a simple CLI loop
        result = agent.run_sync(user_input)

        print(f"\nAI: {result.output}")
        print("-" * 20)


if __name__ == "__main__":
    chat()

```

now, it gives me the right answers.

![can indeed retrieve knowledge](assets/images/can-retrieve-knowledge.png)

Yet I noticed:

1. If I asked "What is the name of this project?", I am expecting "Titan" rather than "I am Gemini, a large language model ..."
2. there is a noticeable time window between when I execute this command and when I can actually type in my question.

k. Adding structured output
