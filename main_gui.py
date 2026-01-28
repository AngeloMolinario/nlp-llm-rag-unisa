import streamlit as st
from system.Chatbot import ChatBot
import time
import json



def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

def create_chatbot() -> ChatBot:
    """Create and initialize chatbot instance with fixed settings"""
    with open('system/config.json', 'r') as file:
        config = json.load(file)

    chatbot = ChatBot(
        llm_model_name=config['llm_model'],
        embedding_model=config['embedding_model'],
        document_dir=config['document_directory'],
        temperature=0.1,
        verbose=config['verbose'],
    )
    # Decomment this line to use custom system prompt SYS_PROMPT
    #chatbot.set_system_prompt(SYS_PROMPT)
    
    if not chatbot.initialize():
        st.error("Failed to initialize chatbot")
        return None
    return chatbot

def display_chat_history():
    """Display chat messages from history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="NLP/LLM Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 NLP/LLM Chatbot")
    initialize_session_state()

    # Initialize chatbot if needed
    if st.session_state.chatbot is None:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = create_chatbot()
            if st.session_state.chatbot is None:
                st.stop()

    # Display chat history
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask me about NLP and LLMs!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            start_time = time.time()
            
            # Get response from chatbot with fixed k=5
            response_stream, contexts = st.session_state.chatbot.answer_question(prompt, k=5)
            
            # Stream the response
            for chunk in response_stream:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            # Show response time
            elapsed_time = time.time() - start_time
            st.caption(f"Response time: {elapsed_time:.2f} seconds")
            st.session_state.chatbot.memory_handler.add_exchange(prompt, full_response, contexts)
            
            # Show sources in collapsible section
            if contexts:
                with st.expander("View Sources", expanded=False):
                    for i, (doc, score) in enumerate(contexts, 1):
                        st.markdown(f"**Document {i}**")
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"**Content:**\n{doc.page_content}")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
