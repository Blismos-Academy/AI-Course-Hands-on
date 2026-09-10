import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self):
        self.index = faiss.read_index("vector_db/weather.index")

        with open("vector_db/chunks.pkl", "rb") as file:
            self.chunks = pickle.load(file)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.texts = [chunk.text for chunk in self.chunks]

        self.vectors = self.index.reconstruct_n(
            0, self.index.ntotal
        )

        self.vectors /= np.linalg.norm(
            self.vectors, axis=1, keepdims=True
        )

    def search(self, question, top_k=1):

        query_vector = self.model.encode([question]).astype("float32")

        query_vector /= np.linalg.norm(
            query_vector, axis=1, keepdims=True
        )

        scores = np.dot(
            self.vectors,
            query_vector[0]
        )

        indexes = np.argsort(scores)[::-1][:top_k]

        return [self.texts[i] for i in indexes]


if __name__ == "__main__":

    retriever = HybridRetriever()

    question = "What should I wear in hot weather?"

    results = retriever.search(question)

    print("\n===== RETRIEVED CONTEXT =====\n")

    for i, text in enumerate(results, 1):
        print(f"[Result {i}]")
        print(text)