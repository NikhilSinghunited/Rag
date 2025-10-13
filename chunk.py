from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_text_splitters import RecursiveJsonSplitter
import json
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Import Pinecone
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(f"Pinecone API Key: {PINECONE_API_KEY}")

# Load and process JSON
with open('./swagger.json', 'r') as f:
    json_data = json.load(f)

# Split the raw JSON
splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data=json_data)

print("First 3 chunks:")
for i in range(0, min(3, len(json_chunks))):
    print(f"Chunk {i}: {json_chunks[i]}")
    print("+++++++")

# Create documents
documents = []
for i, chunk in enumerate(json_chunks):
    text = json.dumps(chunk, indent=2)
    doc = Document(
        page_content=text,
        metadata={
            "chunk_id": i,
            "source": "./swagger.json",
            "keys": list(chunk.keys()) if isinstance(chunk, dict) else []
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embeddings model loaded")
