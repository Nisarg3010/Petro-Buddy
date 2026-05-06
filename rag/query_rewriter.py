from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import Config

template = """
Given the conversation history and a follow-up question,
rewrite the question to be standalone.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:
"""

def get_query_rewriter():
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )

    llm = ChatOpenAI(
        model=Config.MODEL_NAME,
        temperature=0
    )

    return LLMChain(llm=llm, prompt=prompt)