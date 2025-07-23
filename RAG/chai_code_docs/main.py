#!/usr/bin/env python3
"""
ChaiCode RAG System - Main Entry Point

This module runs the ChaiCode RAG system in interactive mode by default.
"""

from rag_system import ChaiCodeRAGSystem

def main():
    """Main function to run the RAG system in interactive mode."""
    
    # Initialize the RAG system
    print("Initializing ChaiCode RAG System...")
    rag_system = ChaiCodeRAGSystem()
    
    # Start interactive mode
    rag_system.interactive_mode()

if __name__ == "__main__":
    main() 