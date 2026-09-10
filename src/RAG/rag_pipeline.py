from ingestion import DocumentIngestion
from chunking_embedding import ChunkingEmbedding
from vector_store import FAISSVectorStore


print("===== 1. DOCUMENT INGESTION =====")

ingestion = DocumentIngestion()

documents = ingestion.load_document()

print("Document loaded")
print("Number of documents:", len(documents))


print("\n===== 2. CHUNKING + EMBEDDING =====")

processor = ChunkingEmbedding()

chunks, embeddings = processor.process_documents()

print("Total chunks:", len(chunks))
print("Embedding dimension:", len(embeddings[0]))


print("\n===== 3. FAISS VECTOR STORE =====")

vector_store = FAISSVectorStore()

index = vector_store.create_vector_store(
    chunks,
    embeddings
)

print("Number of vectors stored:", index.ntotal)
print("Vector dimension:", index.d)

print("\nPipeline completed successfully!")