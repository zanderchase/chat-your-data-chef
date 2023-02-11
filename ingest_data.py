from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CollegeConfidentialLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time 
# Scrape college data links:
browser = webdriver.Chrome()

browser.get("https://www.collegeconfidential.com/colleges/")
time.sleep(1)

elem = browser.find_element(By.TAG_NAME, "body")

no_of_pagedowns = 5

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(14) #10
    no_of_pagedowns-=1
    
html = browser.page_source
print(html)
soup = BeautifulSoup(html, "html.parser")
schools = soup.find_all("div", {"class": "l-row l-gx-3 l-gx-xl-4 l-gy-4"})[0]
print("------")
print(schools)
raw_documents = []
i = 1
for s in schools.find_all("a", {"class": "u-margin-bottom-xxs"}, href=True):
    college_link = s['href']
    print(college_link)
    print(i)
    i += 1
    # Load Data
    loader = CollegeConfidentialLoader("https://www.collegeconfidential.com" + college_link)
    data = loader.load()[0]
    raw_documents.append(data)
    if i > 2:
        break
print(raw_documents)
    
    



# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)

print("YOOOO")
print(documents)

# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

query = "What is the average ACT at UChicago?"

docs = vectorstore.similarity_search(query)

print("HEYY")
print(docs)

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)