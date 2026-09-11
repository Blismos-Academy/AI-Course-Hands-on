import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self, threshold=0.65):

        self.index = faiss.read_index(
            "rag/vector_db/weather.index"
        )

        with open(
            "rag/vector_db/chunks.pkl",
            "rb"
        ) as file:
            self.chunks = pickle.load(file)

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.texts = [
            chunk.text
            for chunk in self.chunks
        ]

        self.threshold = threshold

        self.vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal
        ).astype("float32")

        self.vectors /= np.linalg.norm(
            self.vectors,
            axis=1,
            keepdims=True
        )


    def search(self, question, top_k=3):

        query_vector = self.model.encode(
            [question]
        ).astype("float32")

        query_vector /= np.linalg.norm(
            query_vector,
            axis=1,
            keepdims=True
        )

        scores = np.dot(
            self.vectors,
            query_vector[0]
        )

        indexes = np.argsort(scores)[::-1]

        results = []

        for index in indexes:

            score = float(scores[index])

            if score < self.threshold:
                break

            results.append(
                self.texts[index]
            )

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":

    retriever = HybridRetriever()

    question = (
        "30°C, overcast weather, "
        "51% humidity, light wind. "
        "What clothing and food are recommended?"
    )

    results = retriever.search(
        question,
        top_k=3
    )

    print("\n===== RETRIEVED CONTEXT =====\n")

    if results:

        for i, text in enumerate(results, 1):

            print(f"[Result {i}]")
            print(text)

    else:

        print("No relevant context found.")