from langchain_openai import OpenAIEmbeddings
from config import Config

def get_embeddings():
    return OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY
    )