import faiss
import numpy as np
import pickle
import os

     

class FAISSVectorStore:

    def create_vector_store(self, chunks, embeddings):

        # Convert embeddings to float32
        embeddings = np.array(embeddings).astype("float32")

        # Get vector dimension
        dimension = embeddings.shape[1]

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to FAISS
        index.add(embeddings)

        # Create vector_db folder
        os.makedirs("vector_db", exist_ok=True)

        # Save FAISS index
        faiss.write_index(
            index,
            "vector_db/weather.index"
        )

        # Save chunks separately
        with open("vector_db/chunks.pkl", "wb") as file:
            pickle.dump(chunks, file)

        print("FAISS vector store saved successfully")

        return index