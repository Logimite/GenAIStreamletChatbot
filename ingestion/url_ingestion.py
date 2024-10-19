import requests
from bs4 import BeautifulSoup

def load_url (url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()