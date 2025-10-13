import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import  StrOutputParser
# Try new Chroma import; fallback to community
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


# ---------------- Load .env ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("❌ Please set GOOGLE_API_KEY in your .env file")


# ---------------- Step 1: Load ALL markdown files ----------------
documents_dir = Path("./documents")
if not documents_dir.exists():
    raise SystemExit(f" Directory not found: {documents_dir}")

all_files = list(documents_dir.glob("*.md"))
if not all_files:
    raise SystemExit(f" No markdown files found in {documents_dir}")

print(f" Found {len(all_files)} markdown files.")

raw_docs = []
for file_path in all_files:
    text = file_path.read_text(encoding="utf-8")
    print(f"  → Loaded {file_path.name} ({len(text)} chars)")
    raw_docs.append(Document(page_content=text, metadata={"source": str(file_path)}))


# ---------------- Step 2: Split by headers & chunks ----------------
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")],
    strip_headers=False
)

split_docs = []
for doc in raw_docs:
    header_splits = header_splitter.split_text(doc.page_content)

    # Convert dicts/strings to Document if needed
    def as_document(item):
        if isinstance(item, Document):
            return item
        if isinstance(item, dict):
            return Document(page_content=item.get("content", ""), metadata=item.get("metadata", {}))
        if isinstance(item, str):
            return Document(page_content=item, metadata={})
        raise TypeError(f"Unknown type: {type(item)}")

    header_docs = [as_document(x) for x in header_splits]
    for h in header_docs:
        # carry over file source metadata
        h.metadata["source"] = doc.metadata["source"]
    split_docs.extend(header_docs)

print(f"📑 Header-level chunks: {len(split_docs)}")

# Further split into ~500-char chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(split_docs)

print(f"✂️ Final chunk count after recursive split: {len(docs)}")
print("\nExample chunk:\n", docs[0].page_content[:200].replace("\n", " ") + "...")


# ---------------- Step 3: Create embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"🤖 Using HuggingFace Embedding model: {embeddings.model_name}")


# ---------------- Step 4: Store into Chroma (Updated for new API) ----------------
persist_directory = "./vector"
collection_name = "airlines_policies"

# Create persistent client (NEW WAY)
persistent_client = chromadb.PersistentClient(path=persist_directory)
print(f"📊 ChromaDB client: {persistent_client}")

# Create vector store with the new client
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    client=persistent_client,  # Pass client directly
    collection_name=collection_name,
)

print(f"✅ Stored {len(docs)} chunks from {len(all_files)} markdown files into ChromaDB at '{persist_directory}'")

# ---------------- Step 5: Test query of Retrieval(yahi context deta hai hame) ----------------
# Initialize the retriever from the existing collection
vectordb = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

query = "What is the Swiss Airlines cancellation policy?"
print(f"\n🔎 Query: {query}")
results = retriever.get_relevant_documents(query)

# for i, d in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print("Source:", d.metadata.get("source"))
#     print("Content:", d.page_content[:300].re place("\n", " ") + "...")

#--------------------------Augmentation Part------------------#
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
)
prompt = PromptTemplate(
    template="""
You are a helpful assistant 
Answer ONLY from the provided airlines_policy
If the context is insufficient, just say you don't know

Context:
{context}

Question: {question}

Answer:""",
    input_variables=['context','question']
)
question = input("\n enter your question")
retrieved_docs = retriever.invoke(question)
print(retrieved_docs)
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# final_prompt = prompt.invoke({"context": context_text, "question": question})

#-------------------Generation-----------------#
# response = llm.invoke(final_prompt)
# print(f"\n🤖 AI Response:\n{response.content}")

#-------------------Building a chain----------#
def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text
parallel_chain=RunnableParallel(
    {
        'context':retriever | RunnableLambda(format_docs),#pahale question retriver ke pass ja raha hai phir semantic search ho raha hai list of docments ko vector store me la raha hai
        'question':RunnablePassthrough()#question hi input me mil raha hai aur hame question hi output me chaiye
    }

)
output=parallel_chain.invoke('Which tickets/bookings cannot be rebooked online currently?')
# print(output)
parser=StrOutputParser()
main_chain=parallel_chain | prompt | llm | parser
final=main_chain.invoke('what is baggage policy')
print(final)
