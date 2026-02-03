## Journal

**Init project**, this creates `pyproject.toml` and `.python-version`

```shell
uv init
```

**Add dependencies**

```shell
uv add pydantic-ai google-genai chromadb sentence-transformers python-dotenv
```

**Setup Project Structure**

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
