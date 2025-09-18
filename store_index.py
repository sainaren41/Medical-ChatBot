from dotenv import load_dotenv
import os
from src.helpers import load_pdf_file, text_split, download_hugging_face_embeddings, filter_to_minimal_docs
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf_file("data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding vectors
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  index_name=index_name
)
