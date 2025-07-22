from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

if not os.environ.get("QDRANT_API_KEY"):
    raise ValueError("QDRANT_API_KEY not found in environment variables. Please check your .env file.")

# Qdrant client
qdrant_client = QdrantClient(
    url="https://31bd70e4-fc38-401f-89c0-f0f8470bcf3e.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.environ.get("QDRANT_API_KEY"),
)

file_path = "../example_data/Resume.pdf"
# load the pdf file

loader = PyPDFLoader(file_path)
docs = loader.load()

# chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=docs)

# Embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# create a qdrant collection if not exists
# qdrant_client.create_collection(
#     collection_name="resume_collection",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

# Vector Store
vector_store = QdrantVectorStore.from_existing_collection(
    url="https://31bd70e4-fc38-401f-89c0-f0f8470bcf3e.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="resume_collection",
    embedding=embeddings,
)

# add documents to the vector store if not exists
# vector_store.add_documents(documents=split_docs)

user_query = input("Enter yur query in here: ")

relevant_chunks = vector_store.similarity_search(
    query=user_query
)

SYSTEM_PROMPT = '''
You are a helpful assistant who responds based on the available context.
Give your response in the same language as the question. Do not answer the user if the question is not related to the context. If the question is not related to the context, say "I don't know" or "I don't have information about that".
Context: {relevant_chunks}
'''

# call the gemini model 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)

messages = [
    (
        "system",
        SYSTEM_PROMPT.format(relevant_chunks=relevant_chunks)
    ),
    ("user", user_query)
]

response = llm.invoke(messages)

print("Response:", response.content)