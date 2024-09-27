import pickle
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from dotenv import load_dotenv
import os
import nest_asyncio  # noqa: E402
import chardet
nest_asyncio.apply()

# Bring in our LLAMA_CLOUD_API_KEY
load_dotenv()

# LLAMAPARSE API keys
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Load or parse data
def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    # Attempt to load existing parsed data
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            return pickle.load(f)

    # If not available, parse the file
    parsingInstruction = """El documento proporcionado es información sobre una farmacia."""
    parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown",
                        parsing_instruction=parsingInstruction)
    llama_parse_documents = parser.load_data("./data/infoFarmacia.txt")

    # Save the new parsed data to a file
    with open(data_file, "wb") as f:
        pickle.dump(llama_parse_documents, f)

    return llama_parse_documents

# Create vector database
def create_vector_database():
    llama_parse_documents = load_or_parse_data()

    # Ensure documents are valid before proceeding
    if not llama_parse_documents:
        print("No valid documents to process.")
        return

    # Write parsed documents to output.md
    with open('data/output.md', 'w', encoding='utf-8', errors='replace') as f:
        for doc in llama_parse_documents:
            # Ensure doc.text is a string and replace problematic characters
            f.write(str(doc.text).replace('\n', ' ') + '\n')

    # Load documents from the directory
    loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = FastEmbedEmbeddings()

    # Create and persist a Qdrant vector database from the chunked documents
    if docs:  # Ensure there are documents to process
        qdrant = Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            url=qdrant_url,
            collection_name="rag",
            api_key=qdrant_api_key
        )
        print('Vector DB created successfully!')
    else:
        print("No documents to create a vector DB.")

if __name__ == "__main__":
    create_vector_database()
