import pinecone
from langchain.vectorstores import Pinecone
from config import Config
from embeddings import get_embeddings

def init_pinecone():
    pinecone.init(
        api_key=Config.PINECONE_API_KEY,
        environment=Config.PINECONE_ENV
    )

    if Config.INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=Config.INDEX_NAME,
            dimension=1536
        )

def index_documents(chunks):
    init_pinecone()

    embeddings = get_embeddings()

    vectorstore = Pinecone.from_documents(
        chunks,
        embeddings,
        index_name=Config.INDEX_NAME
    )

    return vectorstore