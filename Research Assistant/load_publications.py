import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_research_publications(documents_path):
    """Load research publications from .txt files and return as list of strings"""
    
    # List to store all documents
    documents = []
    
    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Extract content as strings and return
    publications = []
    for doc in documents:
        publications.append(doc.page_content)
    
    return publications

def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data

if __name__ == "__main__":
    # Path to research documents
    documents_path = "./research_documents"

    # Load all research publications
    publications = load_research_publications(documents_path)

    # Chunk each publication
    all_chunks = []
    for i, paper_content in enumerate(publications):
        title = f"Paper_{i+1}"
        chunks = chunk_research_paper(paper_content, title)
        all_chunks.extend(chunks)
        print(f"Chunked {title}: {len(chunks)} chunks")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Display sample chunk
    if all_chunks:
        print("\nSample chunk:")
        print(f"Title: {all_chunks[0]['title']}")
        print(f"Chunk ID: {all_chunks[0]['chunk_id']}")
        print(f"Content preview: {all_chunks[0]['content'][:200]}...")
