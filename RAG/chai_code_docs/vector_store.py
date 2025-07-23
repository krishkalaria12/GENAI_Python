import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

class VectorStoreManager:
    """Manages vector store operations for the ChaiCode RAG system."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store = None
        self._connect_to_vector_store()
    
    def _connect_to_vector_store(self):
        """Connect to the Qdrant vector store."""
        print("Connecting to Qdrant vector store...")
        try:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                url=Config.QDRANT_API_URL,
                api_key=Config.QDRANT_API_KEY,
                collection_name=Config.COLLECTION_NAME,
                embedding=self.embeddings,
            )
            print("Successfully connected to Qdrant vector store")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Please check your QDRANT_API_URL and QDRANT_API_KEY in the .env file")
            print("Also ensure your internet connection is stable and the Qdrant service is accessible")
            raise
    
    def search_with_retry(self, query, max_retries=None, delay=None):
        """Search with retry logic for connection issues."""
        if max_retries is None:
            max_retries = Config.MAX_RETRIES
        if delay is None:
            delay = Config.RETRY_DELAY
            
        for attempt in range(max_retries):
            try:
                return self.vector_store.similarity_search(query=query)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for query '{query}': {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to search for query '{query}' after {max_retries} attempts")
                    return []  # Return empty list if all attempts fail
    
    def upload_batch_with_retry(self, batch, max_retries=None, delay=None):
        """Upload a batch of documents with retry logic."""
        if max_retries is None:
            max_retries = Config.MAX_RETRIES
        if delay is None:
            delay = Config.RETRY_DELAY
            
        for attempt in range(max_retries):
            try:
                self.vector_store.add_documents(documents=batch)
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to upload batch after {max_retries} attempts")
                    return False
    
    def split_documents(self, documents):
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents=documents)
    
    def upload_documents_batch(self, documents):
        """Upload documents in batches to avoid timeout."""
        split_docs = self.split_documents(documents)
        print(f"Split documents into {len(split_docs)} chunks")
        
        total_chunks = len(split_docs)
        successful_uploads = 0
        
        for i in range(0, total_chunks, Config.BATCH_SIZE):
            batch = split_docs[i:i + Config.BATCH_SIZE]
            batch_num = i//Config.BATCH_SIZE + 1
            total_batches = (total_chunks + Config.BATCH_SIZE - 1)//Config.BATCH_SIZE
            
            print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            if self.upload_batch_with_retry(batch):
                successful_uploads += len(batch)
                print(f"Successfully uploaded batch {batch_num}")
            else:
                print(f"Failed to upload batch {batch_num}")
        
        print(f"Upload complete: {successful_uploads}/{total_chunks} chunks uploaded successfully")
        return successful_uploads 