import streamlit as st
from ingestion.pdf_ingestion import load_pdf
from ingestion.url_ingestion import load_url
from docs.google_docs import load_google_doc
from retrieval.retriever import generate_answer

st.title("RAG APP")
uploaded_files = st.file_uploader("Uplaod PDFs", type=["pdf", accept_multiple_files == True])
urls = st.text_area("Enter website URLs, comma separated")
google_docs_ids = st.text_area("Enter Google Doc IDs, comma separated")

if st.button("Process Documents"):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            documents.append(load_pdf(uploaded_file))
    for urls in urls.split(","):
        documents.append(load_url(url))
    credentials = None #put google api key here
    for doc_id in google_docs_ids.split(","):
        documents.append(load_google_doc(doc_id, credentials))
        
    st.write("Documents uploaded successfully")
    
    query = st.text_input("Ask a question about the documents")
    
    if query:
        answer = generate_answer (query, documents)
        st.write(f"Answer: {answer}")