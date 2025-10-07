
import os
import torch
import chromadb
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize embeddings model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)

def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""
    
    # Convert question to vector
    query_vector = embeddings.embed_query(query)
    
    # Search for similar content
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return relevant_chunks

def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""
    
    # Get relevant research chunks
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    
    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}" 
        for chunk in relevant_chunks
    ])
    
    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Based ONLY on the following research findings, answer the researcher's question.
    Do not use any external knowledge. If the answer is not found in the research context, say "I cannot answer this question based on the available research papers."
    Give me only 0.50+ Similarity sources.

    Research Context:
    {context}

    Researcher's Question: {question}

    Answer: Provide a comprehensive answer using ONLY information from the research findings above. Do not add information from outside sources.
    """
        )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    # Store Q&A in JSON
        
    qa_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "answer": response.content,
        "sources": [
            {
                "title": chunk['title'],
                "content": chunk['content']
            }
            for chunk in relevant_chunks
        ]
    }
    
    # Append to JSON file
    qa_file = "qa_history.json"
    try:
        with open(qa_file, 'r') as f:
            qa_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        qa_history = []
    
    qa_history.append(qa_entry)
    
    with open(qa_file, 'w') as f:
        json.dump(qa_history, f, indent=2)
    
    return response.content, relevant_chunks

if __name__ == "__main__":
    # Initialize LLM
    llm = ChatGroq(model="qwen/qwen3-32b")
    
    # Example query
    query = "What are effective techniques for handling class imbalance?"
    print(f"Question: {query}\n")
    
    # Get answer
    answer, sources = answer_research_question(
        query,
        collection, 
        embeddings, 
        llm
    )
    
    print("=" * 80)
    print("AI Answer:")
    print("=" * 80)
    print(answer)
    print("\n" + "=" * 80)
    print("Based on sources:")
    print("=" * 80)
    for source in sources:
        print(f"- {source['title']} (Similarity: {source['similarity']:.2f})")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Research Assistant")
    print("=" * 80)
    print("Ask questions about your research papers (type 'quit' to exit)\n")
    
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        print("\nSearching research papers...\n")
        answer, sources = answer_research_question(
            user_query,
            collection, 
            embeddings, 
            llm
        )
        
        print("=" * 80)
        print("Answer:")
        print("=" * 80)
        print(answer)
        print("\n" + "=" * 80)
        print("Sources:")
        print("=" * 80)
        for source in sources:
            print(f"- {source['title']} (Similarity: {source['similarity']:.2f})")
        print("\n")