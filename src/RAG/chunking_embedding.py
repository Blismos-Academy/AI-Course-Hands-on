import re
from llama_index.core import Document
from sentence_transformers import SentenceTransformer
from ingestion import DocumentIngestion


class ChunkingEmbedding:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def process_documents(self):
        documents = DocumentIngestion().load_document()
        text = documents[0].text

        sections = re.split(r"(?=\d+\.\s)", text)

        chunks = [
            Document(text=section.strip())
            for section in sections
            if section.strip()
        ]

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts)

        return chunks, embeddings