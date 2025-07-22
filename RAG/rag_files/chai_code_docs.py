import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

if not os.environ.get("QDRANT_API_KEY"):
    raise ValueError("QDRANT_API_KEY not found in environment variables. Please check your .env file.")

if not os.environ.get("QDRANT_API_URL"):
    raise ValueError("QDRANT_API_URL not found in environment variables. Please check your .env file.")

# load the docs from the web
# loader = WebBaseLoader([
#     "https://docs.chaicode.com/youtube/getting-started/",
#     "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
#     "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
#     "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
#     "https://docs.chaicode.com/youtube/chai-aur-git/github/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
#     "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/models/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
#     "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
#     "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
#     "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
# ])
# loader.requests_kwargs = {
#     'verify': False,  # Disable SSL verification
# }

# docs = loader.load()
# print("Loaded documents:", len(docs))

# chunking
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# )

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# Vector Store
print("Connecting to Qdrant vector store...")
try:
    vector_store = QdrantVectorStore.from_existing_collection(
        url=os.environ.get("QDRANT_API_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        collection_name="chai_code_docs",
        embedding=embeddings,
    )
    print("Successfully connected to Qdrant vector store")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Please check your QDRANT_API_URL and QDRANT_API_KEY in the .env file")
    print("Also ensure your internet connection is stable and the Qdrant service is accessible")
    exit(1)

# split_docs = text_splitter.split_documents(documents=docs)
# print(f"Split documents into {len(split_docs)} chunks")

# def upload_batch_with_retry(vector_store, batch, max_retries=3, delay=2):
#     """Upload a batch of documents with retry logic"""
#     for attempt in range(max_retries):
#         try:
#             vector_store.add_documents(documents=batch)
#             return True
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#                 delay *= 2  # Exponential backoff
#             else:
#                 print(f"Failed to upload batch after {max_retries} attempts")
#                 return False

# Batch processing to avoid timeout
# batch_size = 10
# total_chunks = len(split_docs)
# successful_uploads = 0

# for i in range(0, total_chunks, batch_size):
#     batch = split_docs[i:i + batch_size]
#     batch_num = i//batch_size + 1
#     total_batches = (total_chunks + batch_size - 1)//batch_size
    
#     print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)")
    
#     if upload_batch_with_retry(vector_store, batch):
#         successful_uploads += len(batch)
#         print(f"Successfully uploaded batch {batch_num}")
#     else:
#         print(f"Failed to upload batch {batch_num}")

# print(f"Upload complete: {successful_uploads}/{total_chunks} chunks uploaded successfully")

# Query translation 

class ImprovedQueries(BaseModel):
    """Information about improved queries."""
    queries: List[str] = Field(..., description="List of 3 improved queries related to the user query")

IMPROVE_USER_QUERY_SYSTEM_PROMPT = '''
you are an helpful ai assitant in a software engineering company called ChaiCode.
you are tasked with improving the user query. You need to give 3 new queries that is related to the user query and improves the user query by giving more context and also predicting what the user might be looking for and what user might ask for.

For example:
user_query = "what is a fs module"
improved_queries = [
    "What is module system in Node.js?",
    "What is the fs module in Node.js and how do I use it?",
    "What are the common methods in the fs module of Node.js?",
]

the given user query is: {user_query} for which you need to generate 3 improved queries and also i have given you an example on how to do it.

IMPORTANT:
1. Do not repeat the user query in the improved queries.
2. RESPOND ONLY WITH THE IMPROVED QUERIES IN A LIST FORMAT.
3. Do not add any additional text or explanation.
'''

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)
structured_llm = llm.with_structured_output(ImprovedQueries)

user_query = input("Enter your query in here: ")

messages = [
    (
        "system",
        IMPROVE_USER_QUERY_SYSTEM_PROMPT.format(user_query=user_query)
    ),
    ("user", user_query)
]

print("Improving the query")
response = structured_llm.invoke(messages)
improved_queries = response.queries

all_documents = []

# Method 1 -> parallel query (FANOUT)
print("Fetching the relevant chunks")

def search_with_retry(vector_store, query, max_retries=3, delay=2):
    """Search with retry logic for connection issues"""
    for attempt in range(max_retries):
        try:
            return vector_store.similarity_search(query=query)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for query '{query}': {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to search for query '{query}' after {max_retries} attempts")
                return []  # Return empty list if all attempts fail

for improved_query in improved_queries:
    relevant_chunks = search_with_retry(vector_store, improved_query)
    all_documents.append(relevant_chunks)

# filtering the unique ones
def filter_unique_by_content(all_documents):
    """Filter unique documents based on their page_content"""
    seen_content = set()
    unique_documents = []
    
    for doc_list in all_documents:
        for doc in doc_list:
            # Use page_content as unique identifier
            content_hash = doc.page_content
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
    
    return unique_documents

unique_relevant_chunks = filter_unique_by_content(all_documents=all_documents)

# Check if we found any relevant chunks
if not unique_relevant_chunks:
    print("No relevant documents found for your query. This could be due to:")
    print("1. Connection issues with the vector database")
    print("2. The query not matching any documents in the database")
    print("3. The vector database being empty or not properly indexed")
    print("Please try again or check your connection.")
    exit(1)

print(f"Found {len(unique_relevant_chunks)} relevant document chunks")

SYSTEM_PROMPT = '''
    you are an helpful ai assistant from a software engineering company named as ChaiCode. You are tasked with the answering the user query. The user query is as follows - {user_query} and the relevant chunks from our documentation is {relevant_chunks}. You are requested to answer the user query in the most best way such that the user gets all the information and also he is satisfied. You have been given the relevant chunks after improving the user query and fetching the relevant chunks.
'''

print("Thinking for your solution")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)

messages = [
    (
        "system",
        SYSTEM_PROMPT.format(user_query=user_query, relevant_chunks=unique_relevant_chunks)
    ),
    ("user", user_query)
]

response = llm.invoke(messages)

print("Response:", response.content)