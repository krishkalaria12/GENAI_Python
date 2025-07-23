from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

class QueryRouter:
    """LLM-based query router that uses Gemini through LangChain to determine the best model for a given query."""
    
    def __init__(self):
        """Initialize the query router."""
        self.models = Config.MODELS
        self.has_api_key = Config.validate_config()
        
        if self.has_api_key:
            # Initialize LangChain Google Generative AI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=Config.TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY,
            )
        else:
            self.llm = None
    
    def route_query(self, query: str) -> str:
        """
        Route a query to the most appropriate model using LLM analysis.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The name of the recommended model
        """
        
        try:
            # Create system prompt with model information
            system_prompt = self._create_system_prompt()
            
            # Create messages for LangChain
            messages = [
                ("system", system_prompt),
                ("user", f"User Query: {query}\n\nRecommended Model:")
            ]
            
            # Get response from LangChain
            response = self.llm.invoke(messages)
            
            # Extract model name from response
            recommended_model = self._extract_model_from_response(response.content)
            
            return recommended_model
            
        except Exception as e:
            print(f"Error in LLM routing: {e}")
            return self._fallback_routing(query)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt with all model information."""
        prompt = """You are an expert at routing coding queries to the most appropriate LLM model. 

Available Models:
"""
        
        for model_name, model_info in self.models.items():
            prompt += f"""
{model_name}:
- Cost: {model_info['cost']}
- Knowledge: {model_info['knowledge']}
- Description: {model_info['description']}
"""
        
        prompt += """
Routing Guidelines:
1. For complex debugging, optimization, architecture, or enterprise tasks → Use gpt-4 or claude-3-opus
2. For general coding tasks and explanations → Use gpt-3.5-turbo or claude-3-sonnet
3. For cost-sensitive simple tasks → Use gemini-pro or llama-3.1-8b
4. Consider the user's query complexity and any cost preferences mentioned

Analyze the user query and respond with ONLY the model name (e.g., "gpt-4", "gemini-pro", etc.) that would be best suited for this query."""
        
        return prompt
    
    def _extract_model_from_response(self, response: str) -> str:
        """Extract the model name from the LLM response."""
        response = response.strip().lower()
        
        # Look for model names in the response
        for model_name in self.models.keys():
            if model_name.lower() in response:
                return model_name
        
        # If no model found, return default
        return Config.DEFAULT_MODEL
    
    def get_model_info(self, model_name: str) -> dict:
        """Get detailed information about a model."""
        return Config.get_model_info(model_name)
    
    def list_models(self) -> list:
        """List all available models with their information."""
        return [
            {
                "name": model_name,
                **model_info
            }
            for model_name, model_info in self.models.items()
        ] 