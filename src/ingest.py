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
