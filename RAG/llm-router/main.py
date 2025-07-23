#!/usr/bin/env python3
"""
LLM ROUTING RAG System - Main Entry Point

This module runs the LLM Routing RAG system by default.
"""

from rag_system import LLMRoutingRAGSystem

def main():
    """Main function to run the RAG system in interactive mode."""
    
    # Initialize the RAG system
    print("Initializing LLM Routing RAG System...")
    rag_system = LLMRoutingRAGSystem()
    
    # Start interactive mode
    rag_system.interactive_mode()

if __name__ == "__main__":
    main() 