import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_embed_documents(data_dir: str, persist_dir: str = "./chroma_db"):
    """
    Loads markdown files, chunks them, and creates a vector store.
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(f"⚠ Warning: Data directory not found: {data_dir}")
        return None
    
    if not os.path.exists(persist_dir):
        print("Ingesting documents...")
        documents = []
        
        # Load all .md files in data dir
        for filename in os.listdir(data_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(data_dir, filename)
                try:
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    # Add metadata source
                    for doc in docs:
                        doc.metadata["source"] = filename
                    documents.extend(docs)
                    print(f"  ✓ Loaded: {filename}")
                except Exception as e:
                    print(f"   Error loading {filename}: {e}")
        
        if not documents:
            print("⚠ No documents found to ingest!")
            return None
        
        # Chunking strategy: 500 chars with 50 overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print(f"✓ Ingestion complete. {len(splits)} chunks created.")
    else:
        print(f"✓ Loading existing vector store from {persist_dir}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
    return vectorstore