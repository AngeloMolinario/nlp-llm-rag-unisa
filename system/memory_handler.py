"""
Memory Handler Module

This module provides classes for managing conversation history and detecting follow-up questions
in a conversational AI system. It includes two main classes:
- LLMFollowUpDetector: Handles follow-up question detection using an LLM
- ContextMemoryHandler: Manages conversation history and context tracking
"""

class LLMFollowUpDetector:
    """
    A class that uses a Language Model to detect if a question is a follow-up to previous conversation.
    
    This class handles the detection of semantic relationships between questions in a conversation,
    helping to maintain context and provide more relevant answers.
    
    Attributes:
        llm (OllamaLLM): The language model instance used for detection
    """
    
    def __init__(self, llm = None):
        """
        Initialize the detector with an optional LLM instance.
        
        Args:
            llm: Language model instance for detection
        """
        self.llm = llm
    
    def is_followup(self, query, prev_exchanges = None):
        """
        Determine if a query is a follow-up question to previous exchanges.
        
        Args:
            query: The current question to analyze
            prev_exchanges: Previous conversation exchanges
            
        Returns:
            Tuple containing (is_followup, rephrased_query)
        """
        if not prev_exchanges or not self.llm:
            return False, query
            
        last_exchange = prev_exchanges[-1]
        prompt = self._create_detection_prompt(query, last_exchange)
        
        try:
            response = self.llm.invoke(prompt)
            is_followup, rephrased_query = self._parse_llm_response(response)
            return is_followup, rephrased_query
        except Exception as e:
            print(f"LLM detection error: {e}")
            return False, query

    def _create_detection_prompt(self, query, last_exchange):
        """
        Create a prompt for the LLM to analyze the semantic relationship between questions.
        
        Args:
            query: Current question
            last_exchange: Previous question-answer exchange
            
        Returns:
            Formatted prompt with examples and current context
        """
        return [
            {"role": "system", "content": """You are an expert at analyzing conversations. 
Your task is to determine if a ###QUESTION### is semantically linked to the ###OLD QUESTION### and ####ANSWER###.
Respond with exactly two lines:
IS_FOLLOWUP: true/false
REPHRASED: [ optmized version of the question for a RAG system, if not follow-up, or self-contained version of the question]
             
### Example 1
###OLD QUESTION###: What is text classification?
###ANSWER###: Text classification is a fundamental task in natural language processing (NLP) that involves assigning predefined categories to textual documents based on their content. This process is widely used for applications such as topic labeling, intent detection, and sentiment analysis. Unlike document classification, which may incorporate metadata, text classification relies solely on textual content. The approach is typically supervised, requiring a predefined set of classes, and contrasts with unsupervised techniques like clustering.
###QUESTION###: Is there a formal definition?
IS_FOLLOWUP: true
NEW_QUERY: Is there a formal definition for text classification?

### Example 2
###OLD QUESTION###: What is the name of the course?
###ANSWER###: The name of the course is Natural Language Processing (NLP) and Large Language Models (LLMs). It is part of the Master's Degree in Computer Engineering at the University of Salerno (UNISA).
###QUESTION###: What is nlp about?
IS_FOLLOWUP: false
NEW_QUERY: What is nlp about?        

### Example 3
###OLD QUESTION###:  What TF means?
###ANSWER###: TF stands for Term Frequency. It measures how often a word appears in a document. In the context of text representation and information retrieval, term frequency is a foundational concept used to quantify the importance of words within documents. Without normalization, terms that appear frequently might be overrepresented, but normalization techniques can help adjust for varying document lengths, ensuring that the frequency of a term more accurately reflects its significance in the document.
###QUESTION###: Are you sure?
IS_FOLLOWUP: true
NEW_QUERY: Are you sure that TF stands for Term Frequency, and measures how often a word appears in a document?
             
### Example 4
###OLD QUESTION###: What is gpt?
###ANSWER###: GPT stands for Generative Pre-trained Transformer, a type of decoder-only transformer developed by OpenAI. It is known for its ability to generate human-like text by understanding and predicting language. GPT models are trained on vast amounts of text data, allowing them to perform various natural language tasks without task-specific training. The original version, GPT-1, introduced in 2018, had 117 million parameters, while subsequent versions like GPT-2 (with 1.5 billion parameters) and GPT-3 (with 175 billion parameters) significantly increased the model size and capabilities, enhancing their ability to generate coherent long-form text and perform advanced language understanding tasks.
###QUESTION###: What are the differences with llama?
IS_FOLLOWUP: true
NEW_QUERY: What are the differences between GPT and LLAMA?
             
### Example 5
###OLD QUESTION###: What is a decoder only transformer model?
###ANSWER###: In the context of transformer models like GPT, "decoder-only" refers to a model architecture where the Transformer's encoder component is not used during training or inference. Instead, the model relies solely on the decoder to generate text based on the input it receives. This approach allows the model to focus on learning to predict and generate human-like text by understanding context from the input sequence.
###QUESTION###: How are they trained?
IS_FOLLOWUP: true
NEW_QUERY: How are decoder-only models trained?
             

"""},

            {"role": "user", "content": f"""Previous question: {last_exchange['query']}
###OLD QUESTION###: {last_exchange['query']}
###ANSWER###: {last_exchange['response']}
###QUESTION###: {query}
Respond with exactly two lines:
IS_FOLLOWUP: true/false
REPHRASED: [ optmized version of the question for a RAG system, if not follow-up, or self-contained version of the question]"""}
        ]
    
    def _parse_llm_response(self, response):
        """
        Parse the LLM's response to extract follow-up status and rephrased query.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple containing (is_followup, rephrased_query)
        """
        try:
            lines = response.strip().lower().split('\n')
            is_followup = 'true' in lines[0]
            rephrased = lines[1].split(':', 1)[1].strip()
            return is_followup, rephrased
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return False, ""


class ContextMemoryHandler:
    """
    Manages conversation history and context tracking for a conversational AI system.
    
    This class handles the storage and retrieval of conversation history, including
    question-answer pairs and their associated contexts. It also manages the detection
    of follow-up questions using the LLMFollowUpDetector.
    
    Attributes:
        max_history (int): Maximum number of exchanges to keep in history
        conversation_history (List[Dict]): List of conversation exchanges
        llm_detector (LLMFollowUpDetector): Instance for follow-up detection
    """
    
    def __init__(self, llm, max_history = 5):
        """
        Initialize the context memory handler.
        
        Args:
            llm: Language model instance for follow-up detection
            max_history: Maximum conversation history length
        """
        self.max_history = max_history
        self.conversation_history = []
        self.llm_detector = LLMFollowUpDetector(llm)
    
    def add_exchange(self, query, response, contexts = None):
        """
        Add a new exchange to the conversation history.
        
        Args:
            query: User's question
            response: System's response
            contexts: Retrieved contexts and their scores
        """
        exchange = {            
            "query": query,
            "response": response,
            "contexts": [(doc.page_content, score) for doc, score in (contexts or None)] if contexts is not None else ""
        }
        
        self.conversation_history.append(exchange)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_recent_history(self, n = None):
        """
        Retrieve the n most recent conversation exchanges.
        
        Args:
            n: Number of exchanges to retrieve. If None, returns max_history
            
        Returns:
            List of recent exchanges
        """
        n = n or self.max_history
        return self.conversation_history[-n:]
    
    def format_history_for_prompt(self, max_exchanges = 3):
        """
        Format conversation history for inclusion in LLM prompts.
        
        Args:
            max_exchanges: Maximum number of exchanges to include
            
        Returns:
            Formatted conversation history string
        """
        recent = self.get_recent_history(max_exchanges)
        if not recent:
            return ""
            
        history_str = "\nPrevious conversation:\n"
        for exchange in recent:
            context_str = "\n\n".join(
                f"###{doc}" 
                for doc, _ in exchange['contexts']
                ) if exchange['contexts'] is not None else ""
            
            history_str += f"User: context:{context_str}\nQuestion:{exchange['query']}\nAssistant: {exchange['response']}\n"
        return history_str
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history = []

    def is_followup_question(self, query):
        """
        Determine if the current query is a follow-up to previous conversation.
        
        Args:
            query: Current question to analyze
            
        Returns:
            Tuple containing (is_followup, rephrased_query)
        """
        recent_history = self.get_recent_history(1)
        return self.llm_detector.is_followup(query, recent_history)
