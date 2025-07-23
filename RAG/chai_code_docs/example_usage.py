#!/usr/bin/env python3
"""
Example usage of the ChaiCode RAG System

This script demonstrates how to use the modular RAG system with different retrieval methods.
"""

from chai_code_docs.rag_system import ChaiCodeRAGSystem

def demonstrate_methods():
    """Demonstrate all 4 retrieval methods with the same query."""
    
    # Initialize the RAG system
    print("Initializing ChaiCode RAG System...")
    rag_system = ChaiCodeRAGSystem()
    
    # Test query
    test_query = "What are the differences between Django and DevOps, and which one should I learn first for web development?"
    
    print(f"\nTest Query: {test_query}")
    print("="*80)
    
    # Test all methods
    methods = [
        (1, "Parallel Query (FANOUT)"),
        (2, "Rank Fusion"),
        (3, "Query Decomposition"),
        (4, "Hypothetical Document Embedding (HyDE)")
    ]
    
    for method_num, method_name in methods:
        print(f"\n{'='*20} Method {method_num}: {method_name} {'='*20}")
        
        try:
            answer = rag_system.query(test_query, method_choice=method_num)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error with method {method_num}: {e}")
        
        print("\n" + "-"*80)

def interactive_demo():
    """Run an interactive demo where user can choose methods."""
    
    print("Starting Interactive Demo...")
    rag_system = ChaiCodeRAGSystem()
    
    print("\nAvailable Methods:")
    methods = rag_system.get_available_methods()
    for method in methods:
        print(f"  {method}")
    
    while True:
        print("\n" + "="*50)
        
        # Get user query
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid question.")
            continue
        
        # Get method choice
        print("\nChoose a retrieval method:")
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method.split('. ', 1)[1]}")
        
        while True:
            try:
                choice = input("Enter method number (1-4): ").strip()
                method_choice = int(choice)
                
                if 1 <= method_choice <= 4:
                    break
                else:
                    print("Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Process query
        print(f"\nProcessing with Method {method_choice}...")
        try:
            answer = rag_system.query(query, method_choice=method_choice)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error processing query: {e}")

def main():
    """Main function to run examples."""
    
    print("ChaiCode RAG System - Example Usage")
    print("="*50)
    
    while True:
        print("\nChoose an option:")
        print("1. Demonstrate all methods with a test query")
        print("2. Interactive demo (choose your own queries and methods)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            demonstrate_methods()
        elif choice == '2':
            interactive_demo()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Please enter a valid choice (1-3).")

if __name__ == "__main__":
    main() 