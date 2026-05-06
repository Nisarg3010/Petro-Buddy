from rag_chain import build_rag_chain

def main():
    print("RAG Chatbot Started (type 'exit' to quit)\n")

    chain = build_rag_chain()

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        response = chain({"question": query})

        print("\nBot:", response["answer"])
        print("\nSources:", [doc.metadata for doc in response["source_documents"]])
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()