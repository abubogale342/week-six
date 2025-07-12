import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def test_faiss_loading():
    vector_store_path = "/data/faiss_consumer_vector_store"
    
    # Check if files exist
    faiss_path = os.path.join(vector_store_path, "faiss.index")
    metadata_path = os.path.join(vector_store_path, "metadata.pkl")
    
    print(f"FAISS index exists: {os.path.exists(faiss_path)}")
    print(f"Metadata file exists: {os.path.exists(metadata_path)}\n")
    
    # Load the index
    print("Loading FAISS index...")
    index = faiss.read_index(faiss_path)
    print(f"Index loaded. Dimensions: {index.d}, Number of vectors: {index.ntotal}\n")
    
    # Load metadata
    print("Loading metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Metadata loaded. Number of entries: {len(metadata)}")
    print("\nSample metadata entry:")
    print(metadata[0])
    
    # Test a query
    print("\nTesting a query...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "bank closed my account"
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    D, I = index.search(query_embedding, 3)  # Get top 3 results
    
    print("\nTop 3 results:")
    for i, idx in enumerate(I[0]):
        if idx < len(metadata):
            print(f"\nResult {i+1} (Score: {1-D[0][i]:.3f}):")
            print(f"ID: {metadata[idx].get('original_narrative_id', 'N/A')}")
            print(f"Product: {metadata[idx].get('Product', 'Unknown')}")
            print(f"Text: {metadata[idx].get('chunk_text', '')[:200]}...")

if __name__ == "__main__":
    test_faiss_loading()
