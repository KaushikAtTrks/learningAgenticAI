import os
import torch
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from load_publications import chunk_research_paper

def embed_documents(documents: list[str]) -> list[list[float]]:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings

def read_documents_from_folder(folder_path: str) -> list[str]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

def insert_publications(collection, publications):
    """
    Insert documents into a ChromaDB collection.
    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert
    Returns:
        None
    """
    next_id = collection.count()
    for idx, publication in enumerate(publications):
        # Use the filename or index as title for chunking
        title = f"Publication_{next_id+idx}"
        chunks = chunk_research_paper(publication, title)
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = embed_documents(chunk_texts)
        ids = [f"document_{next_id + i}" for i in range(len(chunk_texts))]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunk_texts,
            metadatas=[{'title': chunk['title'], 'chunk_id': chunk['chunk_id']} for chunk in chunks]
        )
        next_id += len(chunk_texts)

if __name__ == "__main__":
    folder = "research_documents"
    documents = read_documents_from_folder(folder)
    
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path="./research_db")
    collection = client.get_or_create_collection(
        name="research_papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Insert publications into the collection
    insert_publications(collection, documents)
    
    print(f"Successfully inserted {len(documents)} publications into the collection.")
    print(f"Total chunks in collection: {collection.count()}")