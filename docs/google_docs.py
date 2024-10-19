from google.oauth2 import service_account
from googleapiclient.discovery import build

def load_google_doc(doc_id, credentials):
    service = build('docs', 'v1', credentials=credentials)
    doc = service.documents().get(documentID=doc_id).execute()
    doc_content = doc.get('body').get('content')
    text = ""
    for element in doc_content:
        if 'paragraph' in element:
            for run in element['paragraph']['elements']:
                if 'textRun' in run:
                    text += run['textRun']['content']
    return text