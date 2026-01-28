from .RAG import RAGSystem
from langchain_ollama import OllamaLLM
from colorama import init, Fore, Style
import argparse
from .memory_handler import ContextMemoryHandler
import time  
import json

# Initialize colorama for cross-platform color support
init()

class ChatBot:
    def __init__(self, llm_model_name, embedding_model,document_dir, temperature=0.1, verbose=False, gpu_id=0):
        # Set CUDA device before initializing LLM
        
        self.llm = OllamaLLM(
            model=llm_model_name, 
            temperature=temperature, 
            max_tokens=1000,
            gpu_id=gpu_id            
        )
        self.rag_system = RAGSystem(verbose=verbose, model_name=embedding_model, directory_path=document_dir)
        self.verbose = verbose
        self.system_prompt = self._default_system_prompt()
        self.memory_handler = ContextMemoryHandler(llm=self.llm)  # Pass LLM instance here
    
    def _default_system_prompt(self) -> str:
        return "You are a conversational AI developed by Group 1 of the University of Salerno (UNISA), specializing in NLP and LLM topics. Your only purpose is to assist with questions related to NLP,LLMs, and the course itself."
    
    def set_system_prompt(self, prompt: str):
        """Set a custom system prompt"""
        self.system_prompt = prompt

    def initialize(self) -> bool:
        """Initialize the RAG QA system"""
        return self.rag_system.initialize()

    def format_prompt(self, query: str, contexts: list, is_followup: bool) -> list:
        """Enhanced prompt formatting with follow-up handling"""
        context_str = "\n\n".join(
            f"###{doc.page_content}" 
            for doc, _ in contexts
        ) if contexts is not None else ""
        
        history_str = self.memory_handler.format_history_for_prompt(max_exchanges=3) if is_followup else None
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": self._get_instruction_prompt(is_followup)}
        ]
        
        if history_str:
            messages.append({"role": "system", "content": history_str})
            
        messages.append({
            "role": "user", 
            "content": f"###Context\n{context_str}\n###Question\n{query}\n###Answer"
        })
        
        return messages

    def _get_instruction_prompt(self, is_followup: bool) -> str:
        """Get appropriate instruction prompt based on question type"""
        base_instructions = """INSTRUCTIONS ARE MANDATORY, 
DISREPECTING THEM WOULD CAUSE A LARGE SCALE FAILURE
### Instructions:
1. **Answer the question** based on the provided context.
2. Write responses that are **clear, and sufficiently detailed** with respect to the relevance of the context.
3. only if greeted by the user, introduce yourself briefly: "I am a conversational AI developed by Group 1 of UNISA. I specialize in NLP and LLM topics and can help you with related questions.
4. Answer only based on the relevant information found in the context
5. If there are multiple questions answer only the question related to nlp and llm topics
### example1
question: What is capital of Spain? What is Rasa?
answer: Sorry i can answer only question about nlp and llm so i don't know what the capital of spain is, but i can help you on the other topic. Rasa is a ...
### example2
question: What is Rasa and how it work? How can i make pizza? 
answer: Sorry i can't answer on how to make pizza, but I can help you with Rasa. Rasa is a framework...
"""            
        followup_specific = """
6. Consider previous conversation context to build the answer
7. Make explicit connections to previous answers
8. For confirmation question make the answer direct
9. When a confimation question is made double check your previous answer
10. When a confirmaton question is made ignore the context""" if is_followup else ""
            
        return base_instructions + followup_specific
    def answer_question(self, query: str, k: int = 3):
        """Answer a question using retrieved contexts and conversation history"""
        try:
            is_followup, rephrased_query = self.memory_handler.is_followup_question(query)
            
            
            if is_followup:
                print(f"{Fore.BLUE}[Rephrased Query] {rephrased_query}{Style.RESET_ALL}")
                
                # Use rephrased query for search
                query  = rephrased_query
                contexts = self.rag_system.search(
                    query=query.lower(),
                    k=k                    
                )
            else:
                print(f"\n{Fore.YELLOW}[New Topic] Clearing conversation history...{Style.RESET_ALL}")
                self.memory_handler.clear_history()
                contexts = self.rag_system.search(query, k=k)

            if not contexts:
                contexts = None
            query +=  "\n(rember to answer only the part of the question about nlp and llm)"
            messages = self.format_prompt(query, contexts, is_followup)
            print(messages)
            response_stream = self.llm.stream(input=messages)
            
            return response_stream, contexts

        except Exception as e:
            return iter([f"Error generating answer: {str(e)}"]), []

    def reset(self):
        """Reset the chatbot's memory and print confirmation"""
        self.memory_handler.clear_history()
        print(f"\n{Fore.YELLOW}[System] Memory has been cleared.{Style.RESET_ALL}")

def stream_print(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL, end: str = "") -> None:
    """Print text with color and flush immediately for streaming effect"""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, flush=True)

def main():
    
    with open("config.json",'r') as file:
        config = json.load(file)

    print(f"Initializing AI System...")
    chatbot = ChatBot(
        llm_model_name=config['llm_model'],
        embedding_model=config['embedding_model'],
        document_dir="../documents",
        temperature=0,
        verbose=False
    )
    
    # Decomment this line to use custom system prompt SYS_PROMPT
    # chatbot.set_system_prompt(SYS_PROMPT)
    if not chatbot.initialize():
        return
    print("Done initializing Chatbot")
    while True:
        print(f"{Fore.GREEN}User: {Style.RESET_ALL}", end="")
        query = input().strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == '\\reset':
            chatbot.reset()
            continue
        
        start_time = time.time()  # Start timing
        print(f"{Fore.CYAN}Assistant:{Style.RESET_ALL}", end=" ")
        
        answer_stream, contexts = chatbot.answer_question(query, k=4)
        
        # Collect the full response
        full_response = ""
        elapsed_time=None
        for chunk in answer_stream:
            if elapsed_time is None:
                elapsed_time = time.time() - start_time
            stream_print(chunk, Fore.CYAN, end="")
            full_response += chunk
        
        # Calculate and print elapsed time
        
        print(f"\n{Fore.YELLOW}[Response time: {elapsed_time:.2f} seconds]{Style.RESET_ALL}")

        # Add to memory handler instead of conversation history
        chatbot.memory_handler.add_exchange(query, full_response, contexts)

        # Display source documents if --source flag is True
        if config['show_source'] and contexts:
            print(f"\n{Fore.MAGENTA}Retrieved Documents:{Style.RESET_ALL}")
            for i, (doc, score) in enumerate(contexts, 1):
                # Document header with yellow color
                print(f"\n{Fore.YELLOW}Document {i}{Style.RESET_ALL}")
                # Source file in blue
                print(f"{Fore.BLUE}Source: {doc.metadata.get('source', 'Unknown')}{Style.RESET_ALL}")
                # Similarity score in green
                print(f"{Fore.GREEN}Similarity Score: {score:.4f}{Style.RESET_ALL}")
                # Content in white
                print(f"{Fore.WHITE}Content: {doc.page_content}{Style.RESET_ALL}")
                print("-" * 80)

if __name__ == "__main__":
    main()
