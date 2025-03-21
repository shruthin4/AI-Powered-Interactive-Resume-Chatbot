
import os
import hashlib
import logging
import pdfplumber
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from nlp_utils import cleaning, extract_words  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chroma_db")

DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_collection"

logger.info("🔍 Loading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.info("✅ Embedding Model Loaded!")

chroma_client = chromadb.PersistentClient(path=DB_DIR)
vector_db = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)
logger.info(
    "🟢 ChromaDB Initialized. Existing Documents: %d",
    len(vector_db.get().get("ids", []))
)


def read_pdf(file_path):
    """Reads PDF text using pdfplumber only. Skips if no text is found."""
    text_chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

    return "\n".join(text_chunks)


def load_documents(main_folder):
    """
    Loads documents from subfolders. Each subfolder is 'category'.
    pdfplumber for PDF, direct read for .txt/.md. Skips empty text docs.
    """
    all_texts = []
    for root, _, files in os.walk(main_folder):
        category = os.path.basename(root)
        for file in files:
            file_path = os.path.join(root, file)
            logger.debug(f"Processing file: {file_path}")

            text_content = ""
            if file.lower().endswith(".pdf"):
                text_content = read_pdf(file_path)
            elif file.lower().endswith(".txt") or file.lower().endswith(".md"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path}: {e}")
                    text_content = ""
            else:
                logger.info(f"Skipping {file_path} (unsupported extension).")
                continue

            text_content = text_content.strip()
            if not text_content:
                logger.info(f"Skipping {file_path} - extracted text is empty.")
                continue

            all_texts.append({
                "text": text_content,
                "source": file,
                "category": category
            })
    return all_texts


def chunk_text(text, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def store_documents_in_chromadb(main_folder, reset_db=False, use_chunking=False):
    """Stores docs in ChromaDB, skipping duplicates. Optionally chunk large texts."""
    if reset_db:
        logger.info("🗑️ Resetting ChromaDB...")
        chroma_client.delete_collection(COLLECTION_NAME)
        global vector_db
        vector_db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model
        )

    documents = load_documents(main_folder)
    existing_ids = set(vector_db.get().get("ids", []))

    new_texts, new_metadatas, new_ids = [], [], []

    for doc in documents:
        if use_chunking:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(chunk.encode()).hexdigest()
                if doc_id in existing_ids:
                    logger.warning(f"⚠️ Skipping duplicate chunk: {doc['source']} chunk {i}")
                    continue
                new_texts.append(chunk)
                new_metadatas.append({
                    "source": f"{doc['source']} (chunk {i})",
                    "category": doc["category"]
                })
                new_ids.append(doc_id)
        else:
            doc_id = hashlib.md5(doc["text"].encode()).hexdigest()
            if doc_id in existing_ids:
                logger.warning(f"⚠️ Skipping duplicate doc {doc['source']}")
                continue
            new_texts.append(doc["text"])
            new_metadatas.append({
                "source": doc["source"],
                "category": doc["category"]
            })
            new_ids.append(doc_id)

    if new_texts:
        vector_db.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
        logger.info(f"✅ Stored {len(new_texts)} new documents (with{'out' if not use_chunking else ''} chunking).")
    else:
        logger.warning("⚠️ No new documents added.")


def list_chromadb_documents():
    """Lists doc IDs, sources, categories from the Chroma collection."""
    try:
        stored_docs = vector_db.get()
        if not stored_docs or not stored_docs.get("ids"):
            return []
        results = []
        for doc_id, meta in zip(stored_docs["ids"], stored_docs["metadatas"]):
            src = meta.get("source", "Unknown Document")
            cat = meta.get("category", "General")
            results.append({"id": doc_id, "source": src, "category": cat})
        return results
    except Exception as e:
        logger.error(f"Error retrieving stored docs: {str(e)}")
        return []


def detect_folder_category(user_message):
    doc_list = list_chromadb_documents()
    categories = set(d["category"].lower() for d in doc_list)
    msg = user_message.lower()

    matched = [cat for cat in categories if cat in msg]
    if len(matched) == 1:
        return matched[0]
    elif len(matched) > 1:
        return matched[0]
    else:
        return None


def query_chromadb(user_query, folder_category=None):
    logger.info(f"User Query: {user_query}, folder_category: {folder_category}")

    cleaned_q = cleaning(user_query)
    extracted = extract_words(user_query)
    enhanced_q = cleaned_q
    if extracted:
        enhanced_q += " " + " ".join(extracted.keys())

    logger.info(f"Enhanced Query: {enhanced_q}")
    total_docs = len(vector_db.get().get("ids", []))
    logger.info(f"Total docs in DB: {total_docs}")

    logger.info("🔍 Doing broad search with entire DB for best coverage.")
    all_results = vector_db.similarity_search_with_score(enhanced_q, k=total_docs)
    logger.info(f"🔎 Found {len(all_results)} results from broad search.")

    broad_set = []
    for doc, score in all_results:
        broad_set.append({
            "source": doc.metadata.get("source", "Unknown"),
            "category": doc.metadata.get("category", "General"),
            "text": doc.page_content
        })

    if folder_category:
        logger.info(f"🔎 Additionally retrieving folder_category: {folder_category}")
        cat_doc_ids = [
            d["id"] for d in list_chromadb_documents()
            if d["category"].lower() == folder_category.lower()
        ]
        cat_results = []
        if cat_doc_ids:
            stored_raw = vector_db.get(ids=cat_doc_ids)
            for idx, doc_id in enumerate(stored_raw["ids"]):
                meta = stored_raw["metadatas"][idx]
                page_content = stored_raw["documents"][idx]
                cat_results.append({
                    "source": meta.get("source", "Unknown"),
                    "category": meta.get("category", "General"),
                    "text": page_content
                })
            logger.info(f"Found {len(cat_results)} docs in {folder_category} folder.")
            unified = unify_docs(broad_set + cat_results)
            return unified

    return unify_docs(broad_set)


def unify_docs(doc_list):
    """
    Deduplicate entire doc_list line by line to reduce repeated lines
    especially in the same or multiple docs referencing the same bullet points.
    """
    unique_map = {}
    for d in doc_list:
        key = (d["source"], d["text"])
        if key not in unique_map:
            unique_map[key] = d
    return list(unique_map.values())


def combine_docs_text(documents):
    """
    Combine all doc texts into a single string, labeling source if needed,
    then deduplicate lines.
    """
    raw = []
    for doc in documents:
        content = f"[Source: {doc['source']} - Category: {doc['category']}]\n{doc['text']}"
        raw.append(content)

    combined = "\n".join(raw)

    lines = combined.split("\n")
    seen = set()
    deduped = []
    for line in lines:
        norm = line.strip()
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)
    final_text = "\n".join(deduped)
    return final_text


if __name__ == "__main__":

    store_documents_in_chromadb("./Documents", reset_db=True, use_chunking=False)






