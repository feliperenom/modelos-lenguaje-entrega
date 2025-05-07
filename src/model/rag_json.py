import os
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

FAISS_INDEX_DIR = "faiss_index"
DATA_DIR = "C:/Users/Felipe/Documents/GitHub/modelos-lenguaje-entrega/src/data/web_scrapping_json"

def load_documents_from_json_folder(json_path):

    docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bloques = data.get("bloques", [])
    for i, bloque in enumerate(bloques):
        title = bloque.get("title", "").strip()
        text = bloque.get("text", "").strip()
        # print(f"--- Bloque {i} ---")
        # print("TITLE:", repr(title))
        # print("TEXT:", repr(text))
        # print("RAW BLOQUE:", bloque)
        # print("-----------------")
        if title and text:
            docs.append(f"{title}\n{text}")
        elif title:
            docs.append(title)
        elif text:
            docs.append(text)
    return docs

def docs_as_lc_documents(docs):
    return [Document(page_content=d) for d in docs]

def split_text_into_chunks(documents, chunk_size=5000, chunk_overlap=500):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks generados: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(repr(chunk.page_content))
        print("-----------------")
    return chunks

def store_in_faiss(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(FAISS_INDEX_DIR)
    return vectordb

def load_faiss_index():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectordb

def retrieve_documents(query, k=3):
    vectordb = load_faiss_index()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante en los documentos."
    # Devuelve solo el bloque más relevante
    return docs[0].page_content

def retrieve_documents_with_link(query, k=5):
    vectordb = load_faiss_index()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante en los documentos."
    
    # Tomamos el contenido del bloque más relevante
    content = docs[0].page_content.strip()

    # Buscamos el link correspondiente al bloque en los JSON
    folder_path = "C:/Users/Felipe/Documents/GitHub/modelos-lenguaje-entrega/src/data/json_docs"
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    found_link = None

    # Buscamos por coincidencia exacta del texto en los bloques de cada JSON
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            link = data.get("link")
            for bloque in data.get("bloques", []):
                title = bloque.get("title", "").strip()
                text = bloque.get("text", "").strip()
                bloque_full = f"{title}\n{text}".strip()
                if bloque_full == content:
                    found_link = link
                    break
            if found_link:
                break

    # Devuelve el bloque + link en markdown
    if found_link:
        return f"{content}\n\n[Ver documento legal completo]({found_link})"
    else:
        return content

import glob
import os

if __name__ == "__main__":
    # 1. Cargar documentos desde todos los archivos JSON en la carpeta
    folder_path = "C:/Users/Felipe/Documents/GitHub/modelos-lenguaje-entrega/src/data/json_docs"
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    docs = []
    for json_file in json_files:
        docs.extend(load_documents_from_json_folder(json_file))
    print(f"Se cargaron {len(docs)} bloques desde {len(json_files)} archivos JSON.")

    # 2. Convertir a Document
    docs = docs_as_lc_documents(docs)

    # 3. Sin chunking: cada bloque es un documento
    chunks = split_text_into_chunks(docs, chunk_size=5000, chunk_overlap=500)
    print(f"Se generaron {len(chunks)} documentos para la base vectorial.")

    # 4. Guardar en FAISS
    vectordb = store_in_faiss(chunks)
    print("Base de datos vectorial creada y persistida en FAISS.")

    # 5. Ejemplo de uso del retriever
    resultado = retrieve_documents_with_link("¿Qué dispone para el pago de jubilaciones, pensiones y retiros?")
    if not resultado:
        print("No se encontró información relevante en los documentos.")
    else:
        print(resultado)