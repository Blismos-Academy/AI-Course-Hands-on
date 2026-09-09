from llama_index.core import SimpleDirectoryReader


class DocumentIngestion:

    def load_document(self):
        documents = SimpleDirectoryReader(
            input_files=["Data/weather_data.docx"]
        ).load_data()

        return documents