from fastapi import FastAPI 
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langserve import add_routes 
import os 
from dotenv import load_dotenv
import json 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=groq_api_key)

## Create Prompt Template. 
system_template = "Translate the following into {language}"
prompt = ChatPromptTemplate.from_messages(
    [("system",system_template),("user","{text}")]
)
output = StrOutputParser()

## Create Chain 

chain = prompt|model|output

## App definition 
app = FastAPI(
    title="LangChain Serve",
    version="1.0.0",
    description="A simpleLangChain Serve example"
)



## Add routes 
add_routes(
    app,
    chain,
    path="/chain"
    
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
