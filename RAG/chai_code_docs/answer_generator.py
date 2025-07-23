from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

class AnswerGenerator:
    """Generates final answers using LLM based on retrieved documents."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )
        
        self.system_prompt = '''
        You are a helpful AI assistant from a software engineering company named as ChaiCode. 
        You are tasked with answering the user query. 
        
        The user query is: {user_query}
        
        The relevant chunks from our documentation are: {relevant_chunks}
        
        You are requested to answer the user query in the most comprehensive way such that the user gets all the information and is satisfied. 
        You have been given the relevant chunks after improving the user query and fetching the relevant chunks.
        
        Please provide a detailed, well-structured response that directly addresses the user's question.
        '''
    
    def generate_answer(self, user_query: str, relevant_chunks: list) -> str:
        """Generate a comprehensive answer based on the user query and relevant chunks."""
        print("Thinking for your solution")
        
        # Check if we found any relevant chunks
        if not relevant_chunks:
            return self._generate_no_results_response()
        
        print(f"Found {len(relevant_chunks)} relevant document chunks")
        
        messages = [
            ("system", self.system_prompt.format(
                user_query=user_query, 
                relevant_chunks=relevant_chunks
            )),
            ("user", user_query)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_no_results_response(self) -> str:
        """Generate a response when no relevant documents are found."""
        return """No relevant documents found for your query. This could be due to:

1. Connection issues with the vector database
2. The query not matching any documents in the database  
3. The vector database being empty or not properly indexed

Please try again or check your connection.""" 