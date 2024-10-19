from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

OPENAI_API_KEY = "your_openai_api_key"

def generate_answer(query, documents):
    llm = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    vector_store = FAISS.from_texts(documents, embeddings)
    qa = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())
    answer = qa.run(query)