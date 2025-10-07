from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.runnables.base import RunnableSequence
from langchain.prompts import PromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

MODEL_NAME = "openai/gpt-oss-120b"

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=MODEL_NAME
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms. user-friendly and concise. Just give me 2 lines."
)

# Create a RunnableSequence
chain = prompt | llm

response = chain.invoke({"topic": "What is 2+2 ?"})
print(response.content)