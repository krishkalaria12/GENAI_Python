import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the ChaiCode RAG system."""
    
    # API Keys
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
    QDRANT_API_URL = os.environ.get("QDRANT_API_URL")
    
    # Model configurations
    EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.3
    
    # Vector store settings
    COLLECTION_NAME = "chai_code_docs"
    
    # Text splitting settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Batch processing settings
    BATCH_SIZE = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
        if not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY not found in environment variables. Please check your .env file.")
        
        if not cls.QDRANT_API_URL:
            raise ValueError("QDRANT_API_URL not found in environment variables. Please check your .env file.")
        
        print("Configuration validated successfully!") 