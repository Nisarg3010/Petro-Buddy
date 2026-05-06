from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from memory import get_memory
from query_rewriter import get_query_rewriter
from config import Config
from reranker import Reranker

def build_rag_chain():
    retriever = get_retriever()
    memory = get_memory()
    llm = ChatOpenAI(model=Config.MODEL_NAME)

    reranker = Reranker()

    def custom_retriever(query):
        docs = retriever.get_relevant_documents(query)
        return reranker.rerank(query, docs)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain