from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

question_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 3 questions about {topic}. Return as JSON with key 'questions' containing a list of question strings."
)

answer_prompt = PromptTemplate(
    input_variables=["questions"],
    template="Answer the following questions:\n{questions}\n Return as JSON with key 'qa_pairs' containing a list of objects, each with 'question' and 'answer' keys."
)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=MODEL_NAME
)

output_parser = JsonOutputParser()

question_chain = question_prompt | llm | output_parser

answer_chain = answer_prompt | llm | output_parser

def create_answer_input(output):
    return {"questions": str(output)}

qa_chain = question_chain | create_answer_input | answer_chain

result = qa_chain.invoke({"topic": "intelligence"})
print(result)