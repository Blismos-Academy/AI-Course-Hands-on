# Entire RAG code using LllamaIndex
import os
from dotenv import load_dotenv

from groq import Groq
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Groq client
groq_client = Groq(
    api_key=groq_api_key
)

# Embedding configuration
Settings.embed_model = OpenAIEmbedding(
    model="liquid/lfm-2.5-embedding-350m:free",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1"
)

# Sample document
text = """
LlamaIndex is a framework for building applications using
Large Language Models and external data.

It supports Retrieval Augmented Generation (RAG).
The main steps in a RAG pipeline are data ingestion,
chunking, indexing, retrieval, and response generation.
"""

# Create document
document = Document(text=text)

# Chunking
nodes = Settings.node_parser.get_nodes_from_documents(
    [document]
)

print("Number of chunks:", len(nodes))

# Show chunks
for i, node in enumerate(nodes):
    print(f"\nChunk {i + 1}:")
    print(node.get_content())

# Indexing
index = VectorStoreIndex(
    nodes
)

print("\nIndex created")

# Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=2
)

# User query
user_query = "What are the main steps in a RAG pipeline?"

print("\nUser Query:")
print(user_query)

# Retrieve relevant chunks
retrieved_response = query_engine.retrieve(user_query)

print("\nRetrieved Chunks:")

context = ""

for i, item in enumerate(retrieved_response):
    chunk = item.node.get_content()

    print(f"\nChunk {i + 1}:")
    print(chunk)

    context += chunk + "\n"

# Generate response using Groq
prompt = f"""
Answer the user's question using only the context provided below.

Context:
{context}

Question:
{user_query}

Answer:
"""

response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

# Final response
answer = response.choices[0].message.content

print("\nResponse:")
print(answer)