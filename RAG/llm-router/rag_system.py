from config import Config
from query_router import QueryRouter

class LLMRoutingRAGSystem:
    """LLM-based Query Routing system."""
    
    def __init__(self):
        """Initialize the routing system."""
        self.router = QueryRouter()
        print("LLM Routing System initialized successfully!")
        
        if self.router.has_api_key:
            print("✓ Using Gemini for intelligent query routing")
        else:
            print("⚠ Using fallback keyword-based routing (no API key)")
    
    def route_query(self, user_query: str):
        """
        Route a user query to the most appropriate model using LLM analysis.
        
        Args:
            user_query (str): The user's question
            
        Returns:
            dict: Routing information including recommended model
        """
        print(f"\nAnalyzing query: {user_query}")
        
        try:
            # Route the query to get recommended model
            recommended_model = self.router.route_query(user_query)
            model_info = self.router.get_model_info(recommended_model)
            
            result = {
                "query": user_query,
                "recommended_model": recommended_model,
                "model_info": model_info,
                "routing_method": "LLM-based" if self.router.has_api_key else "Keyword-based"
            }
            
            return result
            
        except Exception as e:
            print(f"Error routing query: {e}")
            return {
                "query": user_query,
                "recommended_model": Config.DEFAULT_MODEL,
                "model_info": Config.get_model_info(Config.DEFAULT_MODEL),
                "routing_method": "Fallback",
                "error": str(e)
            }
    
    def list_models(self):
        """List all available models with their information."""
        return self.router.list_models()
    
    def interactive_mode(self):
        """Run the routing system in interactive mode."""
        print("\n=== LLM Query Routing System - Interactive Mode ===")
        print("This system uses AI to route your queries to the most appropriate LLM model.")
        
        if self.router.has_api_key:
            print("✓ Intelligent routing enabled (Gemini)")
        else:
            print("⚠ Basic routing enabled (no API key)")
        
        print("\nAvailable models:")
        models = self.list_models()
        for model in models:
            print(f"  • {model['name']} ({model['cost']} cost, {model['knowledge']} knowledge)")
            print(f"    {model['description']}")
        
        print("\nEnter 'quit' to exit, 'models' to see model details.")
        
        while True:
            try:
                # Get user query
                user_query = input("\nEnter your query: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_query.lower() == 'models':
                    print("\n" + "="*60)
                    print("AVAILABLE MODELS:")
                    print("="*60)
                    for model in models:
                        print(f"\n{model['name'].upper()}")
                        print(f"  Cost: {model['cost']}")
                        print(f"  Knowledge: {model['knowledge']}")
                        print(f"  Description: {model['description']}")
                    print("="*60)
                    continue
                
                if not user_query:
                    print("Please enter a valid query.")
                    continue
                
                # Route query
                print("\n" + "="*50)
                result = self.route_query(user_query)
                
                print(f"\nROUTING RESULT:")
                print(f"Query: {result['query']}")
                print(f"Recommended Model: {result['recommended_model']}")
                print(f"Model Cost: {result['model_info']['cost']}")
                print(f"Model Knowledge: {result['model_info']['knowledge']}")
                print(f"Model Description: {result['model_info']['description']}")
                print(f"Routing Method: {result['routing_method']}")
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.") 