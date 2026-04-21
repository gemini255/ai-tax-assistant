from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.utils.pdf_loader import load_pdfs


def build_vector_db():

    # Load PDFs
    docs = load_pdfs()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Create FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Save vector DB
    db.save_local("backend/vector_db")

    print("Vector database created!")


if __name__ == "__main__":
    build_vector_db()