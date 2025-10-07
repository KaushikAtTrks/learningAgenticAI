from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

# Choose a currently supported model (replace with one valid in your account)
MODEL_NAME = "openai/gpt-oss-120b"  # or another supported model

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=MODEL_NAME
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms. user-friendly and concise. Just give me 2 lines."
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(topic="quantum computing")
print(response)