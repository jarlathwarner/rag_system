import re

import chromadb
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from chunker import semantic_chunking

# load dataset & model
mitre_ds = load_dataset("Vyber07/cyber-security", data_files="malware/malware__mitre-attack-malware.json")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# create vectorDB
client = chromadb.Client()
collection = client.create_collection("docs_optimized")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# run the RAG pipeline on a sample batch
print("\n--- Running RAG pipeline ---")
chunk_count = 0
for idx, sample in enumerate(mitre_ds["train"]):
    text_content = str(sample)
    chunks = semantic_chunking(text_content)
    if not chunks:
        continue

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    for chunk_idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
        # check if chunk contains a MITRE ID pattern to save in metadata
        id_found = re.search(r'S\d{4}', chunk_text)
        extracted_id = id_found.group(0) if id_found else "none"

        # add the MITRE ID to metadata for later filtering
        collection.add(
            documents=[chunk_text],
            embeddings=[vector.tolist()],
            metadatas=[{"mitre_id": extracted_id, "doc_idx": idx}],
            ids=[f"doc_{idx}_chunk_{chunk_idx}"]
        )
        chunk_count += 1

print(f"Indexed {chunk_count} chunks.")

# get input from command line for what to query
query_text = ""

print("Enter q to exit.")
while query_text != "q":
    query_text = input(f"\n==> Enter your query: ")

    # perform filtering on the query for a MITRE ID
    query_id_match = re.search(r'S\d{4}', query_text, re.IGNORECASE)
    target_id = query_id_match.group(0).upper() if query_id_match else None

    # query chromaDB
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True)
    chroma_filter = {"mitre_id": target_id} if target_id else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        where=chroma_filter
    )

    candidates = results.get('documents', [[]])[0]
    if candidates:
        # rerank results
        pairs = [[query_text, doc] for doc in candidates]
        scores = rerank_model.predict(pairs)
        ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        top_context = ranked_results[0][0]
        generated_answer = f"Relevant match: {top_context[:200]}..."

        print("\n--- Final top result ---")
        print(f"Relevance score: {ranked_results[0][1]:.4f}")
        print(top_context)
        print("-" * 50)

        # get RAG triad evaluation
        # evaluate_rag_triad(query_text, top_context, generated_answer)
    else:
        print("No matching documents found.")