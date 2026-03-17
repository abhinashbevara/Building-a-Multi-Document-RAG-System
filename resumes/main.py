import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 🔷 1. Load Documents
def load_documents(folder_path: str):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")

    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    return documents


# 🔷 2. Split Text
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} chunks")
    return chunks


# 🔷 3. Embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 🔷 4. Create Vector Store
def create_vector_store(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma_db",
        collection_name="rag_docs"
    )
    vector_store.persist()  # ✅ Important
    return vector_store


# 🔷 Helper: Format Docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 🔷 5. Query RAG System
def query_rag_system(query_text, vector_store):
    llm = ChatOllama(model="phi")  # ✅ Lightweight model

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # ✅ Better retrieval

    # ✅ Strong prompt for small models
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.

        Rules:
        - Answer ONLY from the context
        - Keep answer short (max 2 lines)
        - Do not guess
        - If not found, say "I don't know"

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query_text)


# 🔷 6. Main Function
def main():
    folder_path = r"C:\Users\ASUS\OneDrive\Desktop\multidoc\resumes"  # ✅ Your folder

    if not os.path.exists("./chroma_db"):
        print("📦 No vector DB found. Creating one...")
        docs = load_documents(folder_path)
        chunks = split_text(docs)
        vector_store = create_vector_store(chunks)
        print("✅ Vector database created")
    else:
        print("📦 Loading existing vector DB...")
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_function,
            collection_name="rag_docs"
        )

    # 🔷 Query Loop
    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        print("🤔 Thinking...")

        try:
            answer = query_rag_system(query, vector_store)
            print("\n🧠 Answer:\n", answer)
        except Exception as e:
            print("❌ Error:", e)


# 🔷 Run
if __name__ == "__main__":
    main()