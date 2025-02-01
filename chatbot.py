from main import MilvusVectorDB
import logging
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hugging Face token - replace with your token if needed
HF_TOKEN = ""

class ChatBot:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.max_history = 5
        self.vector_db = MilvusVectorDB()
        self.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Using a more accessible model
        
        # Initialize client with token
        self.client = InferenceClient(
            model=self.model_name,
            token=HF_TOKEN
        )
        logger.info(f"ChatBot initialized with model: {self.model_name}")
        self.persona = """Role: Your name is AIVO (Advanced Intelligent Virtual Orator). You are a knowledgeable professor at KTU University.

Objective: Keep your responses strictly focused on the question asked, providing clear, concise explanations using only the relevant context.

Guidelines:
- If any part of the provided input contains image content, ignore it and process the remaining details.
- Ensure responses are direct, non-conversational, and contain no additional or unrelated information.
- If the exact answer is not found in the context provided, state explicitly: "Insufficient information."
- Process only text-based information from the reference materials.
- Maintain a professional, academic tone throughout responses.
- Focus solely on factual information from the provided context."""

    def truncate_text(self, text: str, max_chars: int = 500) -> str:
        """Truncate text to maximum character length at word boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space] + "..."
        return truncated + "..."

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant context from Milvus"""
        logger.info(f"Retrieving context for query: {query}")
        return self.vector_db.search(query, top_k=top_k)

    def format_prompt(self, query: str, contexts: List[str]) -> str:
        truncated_contexts = [self.truncate_text(ctx) for ctx in contexts]
        recent_history = self.conversation_history[-2:] if self.conversation_history else []
        
        history = "\n".join([
            f"Student: {msg['user']}\nProf. AIVO: {self.truncate_text(msg['assistant'], 200)}"
            for msg in recent_history
        ])
        
        context_text = "\n---\n".join(truncated_contexts)
        
        return f"""{self.persona}

Reference Material:
{context_text}

Student: {query}

Provide a focused explanation addressing only the specific question:"""

    def get_response(self, query: str) -> str:
        contexts = self.get_relevant_context(query, top_k=2)  # Reduced from 3 to 2
        context_texts = [text for text, _ in contexts]
        
        prompt = self.format_prompt(query, context_texts)
        
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=256,  # Reduced from 512
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            print("\nProf. AIVO:", response, "\n")
            
            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "contexts": context_texts
            })
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered a technical issue. Please try your question again."

def main():
    chatbot = ChatBot()
    print("Welcome to KTU Virtual Professor Assistant - AIVO")
    print("Type 'quit' to exit the session.")
    print("You can ask questions about the course materials.")
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
                
            if not query:
                continue
                
            chatbot.get_response(query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()