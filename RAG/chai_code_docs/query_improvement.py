from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

class ImprovedQueries(BaseModel):
    """Information about improved queries."""
    queries: List[str] = Field(..., description="List of 3 improved queries related to the user query")

class QueryImprover:
    """Handles query improvement using LLM."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )
        self.structured_llm = self.llm.with_structured_output(ImprovedQueries)
        
        self.improve_query_prompt = '''
        You are a helpful AI assistant in a software engineering company called ChaiCode.
        You are tasked with improving the user query. You need to give 3 new queries that are related to the user query and improve the user query by giving more context and also predicting what the user might be looking for and what user might ask for.

        For example:
        user_query = "what is a fs module"
        improved_queries = [
            "What is module system in Node.js?",
            "What is the fs module in Node.js and how do I use it?",
            "What are the common methods in the fs module of Node.js?",
        ]

        The given user query is: {user_query} for which you need to generate 3 improved queries and also I have given you an example on how to do it.

        IMPORTANT:
        1. Do not repeat the user query in the improved queries.
        2. RESPOND ONLY WITH THE IMPROVED QUERIES IN A LIST FORMAT.
        3. Do not add any additional text or explanation.
        '''
    
    def improve_query(self, user_query: str) -> List[str]:
        """Improve the user query by generating 3 related queries."""
        print("Improving the query")
        
        messages = [
            ("system", self.improve_query_prompt.format(user_query=user_query)),
            ("user", user_query)
        ]
        
        response = self.structured_llm.invoke(messages)
        return response.queries 