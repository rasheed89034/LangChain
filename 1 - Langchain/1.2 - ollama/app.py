import os 
from dotenv import load_dotenv
import streamlit as st 
from langchain_community.llms import Ollama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 


load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Assistant | Neural Intelligence",
    page_icon="🧠",
    layout="centered"
)

with st.sidebar:
    st.title("Developed by Rasheed Ahmad")
    st.markdown("Machine Learning Engineer")
    st.divider()
    st.markdown("This assistant is powered by a locally hosted **LLaMA 2** model, ensuring fast, private, and secure AI capabilities.")
    st.markdown("**Tech Stack:**\n- Streamlit\n- LangChain\n- Ollama\n- Python")

# --- Main Application ---
st.title("🤖 Local AI Assistant")
st.markdown("Ask any question below to get an instant, locally-generated response.")

# System Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a highly capable and professional AI assistant. Please provide clear, concise, and accurate responses to the user's queries."),
        ("user", "Question: {question}")
    ]
)

##Ollama model 
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
## Chain
chain = prompt | llm | output_parser

# User Input Widget
input_text = st.text_input("What would you like to ask?", placeholder="e.g., Explain how a convolutional neural network works...")

# Generation Logic with UI Feedback
if input_text:
    with st.spinner("Generating response..."):
        try:
            # Generate and display response
            response = chain.invoke({"question": input_text})
            st.markdown("### Response:")
            st.info(response)
        except Exception as e:
            # Professional error handling if Ollama isn't running
            st.error("⚠️ Connection Error: Could not connect to the local model.")
            st.warning(f"Please ensure the Ollama app is running on your machine and you have pulled the llama2 model. \n\n**Error Details:** {e}")