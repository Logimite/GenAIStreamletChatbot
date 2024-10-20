from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, documents, generator):
    llm = OpenAI(api_key=openai.api_key)
    embeddings = OpenAIEmbeddings(openai_api_key = openai.api_key)
    vector_store = FAISS.from_texts(documents, embeddings)

    def retrieve(query):
        """
        This function retrieves the most relevant documents from the vector store
        based on the user query.

        Args:
            query (str): The user's question.

        Returns:
            list: A list of document IDs or objects representing the most relevant documents.
        """

        # Encode the query using the same embeddings used for documents
        query_embedding = embeddings.encode(query)

        # Search the vector store for similar documents using the query embedding
        top_k, distances = vector_store.search(query_embedding, k=5)

        # Return a list of document IDs or objects based on the search results
        return [documents[i] for i in top_k]  # Assuming documents are indexed


    qa = RetrievalQA(llm=llm, retriever=retrieve)
    answer = qa.run(query)
    return answer


