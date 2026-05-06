from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)