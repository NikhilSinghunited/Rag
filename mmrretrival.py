docs = [
    "The sun rises in the east.",
    "She enjoys reading books in the evening.",
    "Cats are independent animals.",
    "Programming requires logical thinking.",
    "The ocean covers most of the Earth's surface."
]

from dotenv import load_dotenv
load_dotenv()  # Ensure GOOGLE_API_KEY is in your .env

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use Google's latest text-embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 1) Build the vector store from your texts
vectorstore = FAISS.from_texts(docs, embedding=embeddings)

# 2) Create an MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.7}
)

# 3) Query
query = "what is langchain?"
results = retriever.invoke(query)  # or retriever.get_relevant_documents(query)

for i, doc in enumerate(results, start=1):
    print(f"\n-- Result {i} ---")
    print(doc.page_content)
