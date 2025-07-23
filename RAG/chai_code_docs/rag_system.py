from config import Config
from vector_store import VectorStoreManager
from query_improvement import QueryImprover
from retrieval_methods import RetrievalMethods
from answer_generator import AnswerGenerator
from data_loader import DataLoader

class ChaiCodeRAGSystem:
    """Main RAG system orchestrator for ChaiCode documentation."""
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.vector_store_manager = VectorStoreManager()
        self.query_improver = QueryImprover()
        self.retrieval_methods = RetrievalMethods(self.vector_store_manager)
        self.answer_generator = AnswerGenerator()
        self.data_loader = DataLoader()
        
        print("ChaiCode RAG System initialized successfully!")
    
    def load_and_index_documents(self, urls: list = None):
        """Load documents from URLs and index them in the vector store."""
        print("Loading and indexing documents...")
        
        # Load documents
        docs = self.data_loader.load_chai_code_docs(urls)
        
        # Upload to vector store
        self.vector_store_manager.upload_documents_batch(docs)
        
        print("Document loading and indexing completed!")
    
    def get_available_methods(self):
        """Get list of available retrieval methods."""
        return self.retrieval_methods.get_available_methods()
    
    def query(self, user_query: str, method_choice: int = 4):
        """
        Process a user query using the specified retrieval method.
        
        Args:
            user_query (str): The user's question
            method_choice (int): 1-4 for different retrieval methods
        
        Returns:
            str: Generated answer
        """
        print(f"\nProcessing query: {user_query}")
        print(f"Using retrieval method: {method_choice}")
        
        try:
            # Step 1: Improve query (for methods 1 and 2)
            improved_queries = None
            if method_choice in [1, 2]:
                improved_queries = self.query_improver.improve_query(user_query)
                print(f"Generated {len(improved_queries)} improved queries")
            
            # Step 2: Retrieve relevant documents
            print("Fetching relevant chunks...")
            relevant_chunks = self.retrieval_methods.execute_method(
                method_choice, user_query, improved_queries
            )
            
            # Step 3: Generate answer
            answer = self.answer_generator.generate_answer(user_query, relevant_chunks)
            
            return answer
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        print("\n=== ChaiCode RAG System - Interactive Mode ===")
        print("Available retrieval methods:")
        
        methods = self.get_available_methods()
        for method in methods:
            print(f"  {method}")
        
        print("\nEnter 'quit' to exit.")
        
        while True:
            try:
                # Get user query
                user_query = input("\nEnter your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_query:
                    print("Please enter a valid question.")
                    continue
                
                # Get method choice
                print("\nChoose a retrieval method (1-4):")
                for method in methods:
                    print(f"  {method}")
                
                while True:
                    try:
                        method_choice = input("Enter method number (1-4): ").strip()
                        method_choice = int(method_choice)
                        
                        if 1 <= method_choice <= 4:
                            break
                        else:
                            print("Please enter a number between 1 and 4.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                # Process query
                print("\n" + "="*50)
                answer = self.query(user_query, method_choice)
                print("\nAnswer:")
                print(answer)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.") 