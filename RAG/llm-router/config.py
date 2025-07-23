import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the LLM Routing system."""
    
    # API Keys
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    # Model definitions for routing
    MODELS = {
        "gpt-4": {
            "name": "gpt-4",
            "type": "coding",
            "cost": "high",
            "knowledge": "excellent",
            "description": "Best for complex coding tasks, debugging, and detailed explanations"
        },
        "gpt-3.5-turbo": {
            "name": "gpt-3.5-turbo", 
            "type": "coding",
            "cost": "medium",
            "knowledge": "good",
            "description": "Good for general coding tasks and explanations"
        },
        "claude-3-opus": {
            "name": "claude-3-opus",
            "type": "coding",
            "cost": "high",
            "knowledge": "excellent", 
            "description": "Excellent for complex reasoning and coding tasks"
        },
        "claude-3-sonnet": {
            "name": "claude-3-sonnet",
            "type": "coding",
            "cost": "medium",
            "knowledge": "good",
            "description": "Good balance of cost and performance for coding"
        },
        "gemini-pro": {
            "name": "gemini-pro",
            "type": "coding",
            "cost": "low",
            "knowledge": "good",
            "description": "Cost-effective option for coding tasks"
        },
        "llama-3.1-8b": {
            "name": "llama-3.1-8b",
            "type": "coding",
            "cost": "very_low",
            "knowledge": "basic",
            "description": "Basic coding assistance, good for simple tasks"
        }
    }
    
    # Default settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.3
    
    @classmethod
    def get_available_models(cls):
        """Get list of available models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name):
        """Get information about a specific model."""
        return cls.MODELS.get(model_name, cls.MODELS[cls.DEFAULT_MODEL])
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set."""
        if not cls.GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY not found. Using mock routing instead.")
            return False
        print("Configuration validated successfully!")
        return True 