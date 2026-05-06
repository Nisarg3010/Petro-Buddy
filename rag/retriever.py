from langchain.vectorstores import Pinecone
from embeddings import get_embeddings
from config import Config
import pinecone

def get_retriever():
    pinecone.init(
        api_key=Config.PINECONE_API_KEY,
        environment=Config.PINECONE_ENV
    )

    vectorstore = Pinecone.from_existing_index(
        Config.INDEX_NAME,
        get_embeddings()
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return retriever