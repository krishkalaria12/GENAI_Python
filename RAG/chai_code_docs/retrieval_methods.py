from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from config import Config
from vector_store import VectorStoreManager

class SubQueries(BaseModel):
    """Information about sub-queries for query decomposition."""
    sub_queries: List[str] = Field(..., description="List of sub-queries that break down the complex query into simpler parts")

class RetrievalMethods:
    """Implements different document retrieval strategies."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )
        
        self.decomposition_prompt = '''
        You are an AI assistant that breaks down complex queries into simpler sub-queries for better document retrieval.

        Given a complex user query, decompose it into 3-5 focused sub-queries that address different aspects of the original question.

        For example:
        user_query = "Compare the performance and security features of React vs Vue, and which one is better for enterprise applications?"
        sub_queries = [
            "What are the performance characteristics of React?",
            "What are the security features of React?", 
            "What are the performance characteristics of Vue?",
            "What are the security features of Vue?",
            "React vs Vue comparison for enterprise applications"
        ]

        The user query is: {user_query}

        IMPORTANT:
        1. Create focused, specific sub-queries
        2. Each sub-query should address a different aspect of the original query
        3. Sub-queries should be simpler than the original query
        4. RESPOND ONLY WITH THE SUB-QUERIES IN A LIST FORMAT
        5. Do not add any additional text or explanation
        '''
        
        self.hyde_prompt = '''
        Imagine you are an expert writing a detailed explanation on the topic: '{query}'
        Your response should be comprehensive and include all key points that would be found in the top search result. 
        '''
    
    def method_1_parallel_query(self, improved_queries: List[str]) -> List:
        """Method 1: Parallel query (FANOUT) - Filter unique documents based on content."""
        print("Using Method 1: Parallel Query (FANOUT)")
        
        all_documents = []
        for improved_query in improved_queries:
            relevant_chunks = self.vector_store.search_with_retry(improved_query)
            all_documents.append(relevant_chunks)
        
        # Filter unique documents based on their page_content
        seen_content = set()
        unique_documents = []
        
        for doc_list in all_documents:
            for doc in doc_list:
                content_hash = doc.page_content
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_documents.append(doc)
        
        return unique_documents
    
    def method_2_rank_fusion(self, improved_queries: List[str]) -> List:
        """Method 2: Rank Fusion - Combine results using Reciprocal Rank Fusion."""
        print("Using Method 2: Rank Fusion")
        
        all_documents = []
        for improved_query in improved_queries:
            relevant_chunks = self.vector_store.search_with_retry(improved_query)
            all_documents.append(relevant_chunks)
        
        # Rank Fusion implementation
        fused_scores = {}
        k = 60  # RRF parameter
        
        for docs in all_documents:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)
        
        # Sort by fused scores
        unique_relevant_chunks = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Return only the documents (without scores)
        return [doc for doc, _ in unique_relevant_chunks]
    
    def method_3_query_decomposition(self, user_query: str) -> List:
        """Method 3: Query Decomposition - Break complex query into sub-queries."""
        print("Using Method 3: Query Decomposition")
        
        decomposition_llm = self.llm.with_structured_output(SubQueries)
        
        print("Decomposing the query into sub-queries")
        decomposition_messages = [
            ("system", self.decomposition_prompt.format(user_query=user_query)),
            ("user", user_query)
        ]
        
        decomposition_response = decomposition_llm.invoke(decomposition_messages)
        sub_queries = decomposition_response.sub_queries
        
        print(f"Generated {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"{i}. {sq}")
        
        # Collect documents from all sub-queries
        decomposition_documents = []
        for sub_query in sub_queries:
            relevant_chunks = self.vector_store.search_with_retry(sub_query)
            decomposition_documents.append(relevant_chunks)
        
        # Filter unique documents
        seen_content = set()
        unique_documents = []
        
        for doc_list in decomposition_documents:
            for doc in doc_list:
                content_hash = doc.page_content
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_documents.append(doc)
        
        return unique_documents
    
    def method_4_hypothetical_document_embedding(self, user_query: str) -> List:
        """Method 4: Hypothetical Document Embedding (HyDE)."""
        print("Using Method 4: Hypothetical Document Embedding (HyDE)")
        
        messages = [
            ("system", self.hyde_prompt.format(query=user_query)),
            ("user", user_query)
        ]
        
        response = self.llm.invoke(messages)
        hypothetical_document = response.content
        
        # Search using the hypothetical document
        unique_relevant_chunks = self.vector_store.search_with_retry(hypothetical_document)
        return unique_relevant_chunks
    
    def get_available_methods(self) -> List[str]:
        """Return list of available retrieval methods."""
        return [
            "1. Parallel Query (FANOUT)",
            "2. Rank Fusion", 
            "3. Query Decomposition",
            "4. Hypothetical Document Embedding (HyDE)"
        ]
    
    def execute_method(self, method_choice: int, user_query: str, improved_queries: List[str] = None) -> List:
        """Execute the chosen retrieval method."""
        if method_choice == 1:
            if improved_queries is None:
                raise ValueError("Improved queries are required for Method 1")
            return self.method_1_parallel_query(improved_queries)
        elif method_choice == 2:
            if improved_queries is None:
                raise ValueError("Improved queries are required for Method 2")
            return self.method_2_rank_fusion(improved_queries)
        elif method_choice == 3:
            return self.method_3_query_decomposition(user_query)
        elif method_choice == 4:
            return self.method_4_hypothetical_document_embedding(user_query)
        else:
            raise ValueError(f"Invalid method choice: {method_choice}. Choose 1-4.") 