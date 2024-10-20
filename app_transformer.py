import streamlit as st
import os
from ingestion.pdf_ingestion import load_pdf
from ingestion.url_ingestion import load_url
from docs.google_docs import load_google_doc
from retrieval.retriever_transformer import generate_answer
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from transformers import TextGenerationPipeline

st.title("RAG APP")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
urls_input = st.text_area("Enter website URLs, comma separated")
google_docs_ids_input = st.text_area("Enter Google Doc IDs, comma separated")
google_api_key = os.getenv('GOOGLE_API_KEY')


# TextGenerationPipeline for GPT-J
generator = TextGenerationPipeline.from_pretrained(model="EleutherAI/gpt-j-6b", max_length=50, num_return_sequences=1)

if st.button("Process Documents"):
    documents = []
    

    # Process uploaded PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                documents.append(load_pdf(uploaded_file))
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")

    # Process URLs
    for url in urls_input.split(","):
        url = url.strip()  # Remove any leading/trailing whitespace
        if url:  # Avoid processing empty URLs
            try:
                documents.append(load_url(url))
                #print("documents:  ", documents[:30])
            except Exception as e:
                st.error(f"Error processing URL {url}: {e}")

    # Process Google Docs
    credentials = google_api_key  # Assume the key is set correctly
    for doc_id in google_docs_ids_input.split(","):
        doc_id = doc_id.strip()  # Remove any leading/trailing whitespace
        if doc_id:  # Avoid processing empty IDs
            try:
                documents.append(load_google_doc(doc_id, credentials))
            except Exception as e:
                st.error(f"Error processing Google Doc ID {doc_id}: {e}")

    if documents:
        st.write("Documents uploaded successfully")
    else:
        st.warning("No documents were uploaded or processed.")
    query = ""
    if len(query) < 2:
        query = st.text_input("Ask a question about the documents")
        print("query is: ", query)
        answer = generate_answer(query, documents, generator)
        print("answer is ****: ", answer)
        st.write(f"Answer: {answer}")
        
   