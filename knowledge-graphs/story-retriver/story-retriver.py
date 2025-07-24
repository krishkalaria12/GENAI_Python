from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
from math import ceil

# Load environment variables from .env file
load_dotenv()

# Environment variable checks
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
if not os.environ.get("QDRANT_API_KEY"):
    raise ValueError("QDRANT_API_KEY not found in environment variables. Please check your .env file.")
if not os.environ.get("QDRANT_API_URL"):
    raise ValueError("QDRANT_API_URL not found in environment variables. Please check your .env file.")
if not os.environ.get("NEO4J_URI"):
    raise ValueError("NEO4J_URI not found in environment variables. Please check your .env file.")
if not os.environ.get("NEO4J_USERNAME"):
    raise ValueError("NEO4J_USERNAME not found in environment variables. Please check your .env file.")
if not os.environ.get("NEO4J_PASSWORD"):
    raise ValueError("NEO4J_PASSWORD not found in environment variables. Please check your .env file.")

# Initialize clients and connections
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_API_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)

# Neo4j Graph connection
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD")
)

# Initialize LLM for graph transformation and QA
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)

# Structured output schema for enhanced retrieval
class GraphContext(BaseModel):
    """Structured output for graph-based context"""
    entities: List[str] = Field(description="Key entities found in the graph")
    relationships: List[str] = Field(description="Important relationships between entities")
    graph_answer: str = Field(description="Direct answer from graph traversal")
    confidence: float = Field(description="Confidence score for the graph answer", ge=0.0, le=1.0)

class CombinedContext(BaseModel):
    """Combined context from both vector and graph retrieval"""
    vector_summary: str = Field(description="Summary of relevant text chunks")
    graph_context: GraphContext = Field(description="Structured graph context")
    final_answer: str = Field(description="Comprehensive answer combining both contexts")

# Load and process documents
file_path = './example_data/1-the_gift_of_the_magi_0.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

# Text splitting for both vector store and graph construction
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents=docs)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# Create Qdrant collection if not exists
try:
    qdrant_client.create_collection(
        collection_name="stories_collection",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )
except Exception as e:
    print(f"Collection exists or created: {e}")

# Vector Store setup
vector_store = QdrantVectorStore.from_existing_collection(
    url=os.environ.get("QDRANT_API_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="stories_collection",
    embedding=embeddings,
)

def batch_add_to_vector_store(vector_store, documents, batch_size=20):
    """Add documents to vector store in batches to handle large datasets"""
    total_batches = ceil(len(documents) / batch_size)
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            print(f"Adding vector batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            vector_store.add_documents(documents=batch)
            print(f"✓ Vector batch {batch_num} added successfully")
        except Exception as e:
            print(f"✗ Error adding vector batch {batch_num}: {str(e)}")
            continue

# Add documents to vector store in batches
batch_add_to_vector_store(vector_store, split_docs)

# === KNOWLEDGE GRAPH CONSTRUCTION ===
print("Creating knowledge graph from documents...")

# Initialize the LLM Graph Transformer (auto-detects nodes and relationships)
llm_transformer = LLMGraphTransformer(
    llm=llm,
    strict_mode=False  # Let the LLM automatically determine nodes and relationships
)

# Transform documents into graph documents with batching
def process_documents_in_batches(documents, batch_size=10):
    """Process documents in batches to avoid memory issues and API limits"""
    total_batches = ceil(len(documents) / batch_size)
    all_graph_documents = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            # Transform batch into graph documents
            batch_graph_docs = llm_transformer.convert_to_graph_documents(batch)
            all_graph_documents.extend(batch_graph_docs)
            
            # Add batch to Neo4j immediately to manage memory
            if batch_graph_docs:
                graph.add_graph_documents(
                    batch_graph_docs,
                    baseEntityLabel=True,
                    include_source=True
                )
                print(f"✓ Batch {batch_num} processed successfully ({len(batch_graph_docs)} graph documents)")
            else:
                print(f"⚠ Batch {batch_num} produced no graph documents")
                
        except Exception as e:
            print(f"✗ Error processing batch {batch_num}: {str(e)}")
            continue
    
    return all_graph_documents

# Process documents in batches
graph_documents = process_documents_in_batches(split_docs, batch_size=5)

print(f"Knowledge graph created with {len(graph_documents)} total graph documents")

# Create GraphCypherQAChain for graph-based querying
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=False,
    return_intermediate_steps=False,
    allow_dangerous_requests=True
)

def enhanced_retrieval_with_structure(user_query: str, vector_top_k: int = 5):
    """
    Enhanced retrieval using both vector similarity and graph traversal with structured output
    """
    # Vector-based retrieval
    relevant_chunks = vector_store.similarity_search(
        query=user_query,
        k=vector_top_k
    )
    
    # Graph-based retrieval using GraphCypherQAChain
    try:
        graph_result = graph_chain.invoke({"query": user_query})
        graph_answer = graph_result.get("result", "No specific graph information found.")
    except Exception as e:
        print(f"Graph query error: {e}")
        graph_answer = "Graph information unavailable."
    
    # Prepare vector context summary
    vector_context = "\n".join([f"Chunk {i+1}: {doc.page_content[:200]}..." 
                               for i, doc in enumerate(relevant_chunks)])
    
    return {
        "vector_chunks": relevant_chunks,
        "vector_context": vector_context,
        "graph_answer": graph_answer
    }

# Main interaction loop
user_query = input("Enter your query: ")

# Get enhanced retrieval results
retrieval_results = enhanced_retrieval_with_structure(user_query)

# Enhanced system prompt that combines both contexts
SYSTEM_PROMPT = '''
You are an intelligent assistant with access to both textual content and knowledge graph information. 
Use both sources to provide comprehensive, accurate answers.

TEXTUAL CONTEXT (from document chunks):
{vector_context}

KNOWLEDGE GRAPH CONTEXT:
{graph_answer}

Instructions:
1. Analyze both the textual chunks and graph-based information
2. Provide a comprehensive answer that leverages insights from both sources
3. When the graph provides entity relationships or connections, incorporate those insights
4. If information conflicts between sources, explain the differences
5. Prioritize accuracy and provide specific details when available

Respond with a well-structured, informative answer that demonstrates understanding from both retrieval methods.
'''

# Create structured LLM with output schema
structured_llm = llm.with_structured_output(CombinedContext)

# Prepare context analysis prompt for structured output
context_analysis_prompt = f'''
Based on the following information, provide a structured analysis:

Vector Context: {retrieval_results["vector_context"]}
Graph Answer: {retrieval_results["graph_answer"]}
User Query: {user_query}

Analyze this information and provide:
1. A summary of the vector-based text chunks
2. Structured graph context with entities, relationships, and confidence
3. A comprehensive final answer combining both sources
'''

try:
    # Get structured analysis
    structured_response = structured_llm.invoke([("user", context_analysis_prompt)])
    
    # Final response using structured context
    final_messages = [
        (
            "system",
            SYSTEM_PROMPT.format(
                vector_context=retrieval_results["vector_context"],
                graph_answer=retrieval_results["graph_answer"]
            )
        ),
        ("user", user_query)
    ]
    
    # Get comprehensive response
    final_response = llm.invoke(final_messages)
    
    print("\n" + "="*60)
    print("GRAPHRAG RESPONSE:")
    print("="*60)
    print(final_response.content)
    
    print("\n" + "="*60)
    print("STRUCTURED ANALYSIS:")
    print("="*60)
    print(f"Vector Summary: {structured_response.vector_summary}")
    print(f"Graph Entities: {', '.join(structured_response.graph_context.entities)}")
    print(f"Graph Relationships: {', '.join(structured_response.graph_context.relationships)}")
    print(f"Graph Confidence: {structured_response.graph_context.confidence:.2f}")
    print(f"Combined Answer: {structured_response.final_answer}")

except Exception as e:
    # Fallback to regular response if structured output fails
    print(f"Structured output unavailable, using standard response: {e}")
    
    final_messages = [
        (
            "system",
            SYSTEM_PROMPT.format(
                vector_context=retrieval_results["vector_context"],
                graph_answer=retrieval_results["graph_answer"]
            )
        ),
        ("user", user_query)
    ]
    
    response = llm.invoke(final_messages)
    print("\n" + "="*60)
    print("GRAPHRAG RESPONSE:")
    print("="*60)
    print(response.content)