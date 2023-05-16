"""This is the logic for ingesting data into LangChain."""
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import DirectoryLoader
#from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os, pickle, faiss, json

load_dotenv()

loader = DirectoryLoader('./.htmls/', glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
data = loader.load()

with open('./.htmls/file_url_dict.json', 'r') as f:
    file_url_map = json.load(f)

# Split the docs into smaller chunks if needed
text_splitter = CharacterTextSplitter(chunk_size=400, separator="\n")
docs = []
metadatas = []
for d in data:
    splits = text_splitter.split_text(d.page_content)
    docs.extend(splits)
    source = d.metadata['source'].removeprefix('.htmls/')
    print(f"{source}: {file_url_map[source]}")
    metadatas.extend([{"source": file_url_map[source]}] * len(splits))

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
