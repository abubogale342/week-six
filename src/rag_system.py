"""
RAG (Retrieval-Augmented Generation) System for Consumer Complaints Analysis
"""
import os
import faiss
import pickle
import numpy as np
import re
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class RAGSystem:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 vector_store_path: str = "/data/faiss_consumer_vector_store"):
        """
        Initialize the RAG system with embedding model and vector store.
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            vector_store_path: Path to the FAISS vector store directory
        """
        self.model = SentenceTransformer(model_name)
        self.vector_store_path = vector_store_path
        self.index = None
        self.metadata = None
        self.llm = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize FAISS index, metadata, and LLM components."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(self.vector_store_path, "faiss.index"))
        
        # Load metadata
        with open(os.path.join(self.vector_store_path, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Enhanced rule-based response patterns with better context handling
        self.response_patterns = [
            {
                'patterns': [
                    r'(?i)common.*issue|problem.*credit card|what.*wrong.*card',
                    r'(?i)frequent.*complaint|typical.*problem|issue.*encounter'
                ],
                'response': (
                    "Based on consumer complaints, common credit card issues include:\n"
                    "1. Unauthorized transactions and fraud\n"
                    "2. Billing disputes and statement errors\n"
                    "3. Account access and login problems\n"
                    "4. Poor customer service experiences\n"
                    "5. Issues with rewards and benefits"
                ),
                'context_integration': (
                    "\n\nRelated reports from our database:\n{context}"
                )
            },
            {
                'patterns': [
                    r'(?i)report.*unauthorized|fraud|stolen card|identity theft',
                    r'(?i)what.*do.*unauthorized|fraud.*detect'
                ],
                'response': (
                    "If you notice unauthorized transactions or suspect fraud:\n"
                    "1. Contact your bank/issuer immediately (24/7 hotline preferred)\n"
                    "2. Freeze your card through mobile/app/online banking\n"
                    "3. File an official fraud report with your bank\n"
                    "4. Update any saved payment methods\n"
                    "5. Monitor your credit report for suspicious activity"
                ),
                'context_integration': (
                    "\n\nRecent similar cases indicate:\n{context}"
                )
            },
            {
                'patterns': [
                    r'(?i)resolution.*time|how long.*resolve|time.*complaint',
                    r'(?i)how.*long.*investigat|when.*hear.*back'
                ],
                'response': (
                    "Standard resolution timeframes for complaints:\n"
                    "• Initial response: Within 15 business days\n"
                    "• Temporary credit for fraud claims: Typically 10 days\n"
                    "• Final resolution: Usually within 30-60 days\n"
                    "• Complex cases: May take up to 90 days"
                ),
                'context_integration': (
                    "\n\nOur database shows these resolution times for similar cases:\n{context}"
                )
            },
            {
                'patterns': [
                    r'(?i)information.*file.*complaint|what.*need.*complain',
                    r'(?i)document.*complain|how.*file.*grievance'
                ],
                'response': (
                    "To file an effective complaint, gather these details:\n"
                    "✓ Your full contact information\n"
                    "✓ Account/card numbers (last 4 digits only for security)\n"
                    "✓ Dates and amounts of disputed transactions\n"
                    "✓ Description of the issue (include relevant dates)\n"
                    "✓ Copies of statements or supporting documents\n"
                    "✓ Reference numbers from previous communications"
                ),
                'context_integration': (
                    "\n\nFrom similar cases, these additional details were helpful:\n{context}"
                )
            },
            {
                'patterns': [
                    r'(?i)how.*billing.*dispute|dispute.*charge|wrong.*amount',
                    r'(?i)refund.*request|charge.*shouldn\'t|incorrect.*charge'
                ],
                'response': (
                    "To dispute a billing error or incorrect charge:\n"
                    "1. Contact the merchant first to resolve directly\n"
                    "2. If unresolved, notify your card issuer in writing within 60 days\n"
                    "3. Include: Your information, charge details, reason for dispute\n"
                    "4. Keep records of all communications\n"
                    "5. The issuer must investigate within 30 days"
                ),
                'context_integration': (
                    "\n\nRecent dispute outcomes in similar cases:\n{context}"
                )
            }
        ]
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for the given query.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing the retrieved documents and their metadata
        """
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get the metadata for the retrieved documents
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata):  # Make sure the index is valid
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(1.0 - distances[0][i])  # Convert distance to similarity
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the retrieved context documents
        
        Args:
            query: The user's question
            context_docs: List of document dictionaries containing 'chunk_text' and other metadata
            
        Returns:
            Formatted response string
        """
        if not context_docs:
            return "I couldn't find any relevant information in our records."
        
        # Format the context for the response
        response_parts = ["Here's what I found in our records:"]
        
        for i, doc in enumerate(context_docs[:3]):  # Use top 3 contexts
            text = doc.get('chunk_text', '')
            product = doc.get('Product', 'Unknown')
            score = doc.get('similarity_score', 0.0)
            
            # Truncate long text
            truncated_text = text[:300] + ('...' if len(text) > 300 else '')
            
            response_parts.append(
                f"\n{i+1}. [Product: {product}, Confidence: {score:.2f}]\n"
                f"   {truncated_text}"
            )
        
        response_parts.append(
            "\nWould you like me to help you with anything specific from this information?"
        )
        
        return "\n".join(response_parts)
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query by retrieving relevant documents and generating an answer.
        
        Args:
            question: The user's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the generated answer and retrieved documents
        """
        try:
            # Retrieve relevant documents with metadata
            retrieved_docs = self.retrieve(question, k)
            
            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in our records.",
                    'retrieved_documents': []
                }
            
            # Generate an answer using the retrieved documents with metadata
            answer = self.generate(question, retrieved_docs)
            
            # Prepare the full response
            response = {
                'question': question,
                'answer': answer,
                'retrieved_documents': []
            }
            
            # Include full document details in the response
            for doc in retrieved_docs:
                response['retrieved_documents'].append({
                    'text': doc.get('chunk_text', ''),
                    'product': doc.get('Product', 'Unknown'),
                    'narrative_id': doc.get('original_narrative_id', 'N/A'),
                    'score': doc.get('similarity_score', 0.0)
                })
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                'question': question,
                'answer': "An error occurred while processing your request.",
                'error': error_msg,
                'retrieved_documents': []
            }

def evaluate_rag_system(rag_system: RAGSystem, test_questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Evaluate the RAG system on a set of test questions.
    
    Args:
        rag_system: The RAG system to evaluate
        test_questions: List of dictionaries containing 'question' and 'expected_answer' keys
        
    Returns:
        List of evaluation results for each question
    """
    results = []
    
    for test_case in tqdm(test_questions, desc="Evaluating RAG system"):
        question = test_case['question']
        
        try:
            # Get the system's response
            response = rag_system.query(question)
            
            # Format the retrieved documents for display
            retrieved_docs = []
            for doc in response.get('retrieved_documents', [])[:2]:  # Show top 2 sources
                doc_text = doc.get('text', '')
                retrieved_docs.append({
                    'text': doc_text[:200] + '...' if len(doc_text) > 200 else doc_text,
                    'score': doc.get('score', 0.0),
                    'product': doc.get('product', 'Unknown'),
                    'narrative_id': doc.get('narrative_id', 'N/A')
                })
            
            # Prepare result with evaluation metrics
            result = {
                'question': question,
                'generated_answer': response.get('answer', 'No answer generated'),
                'retrieved_sources': retrieved_docs,
                'quality_score': 0,  # Placeholder for manual evaluation
                'analysis': ""  # Can be filled in manually
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error evaluating question '{question}': {str(e)}")
            results.append({
                'question': question,
                'error': str(e),
                'generated_answer': 'Error processing question',
                'retrieved_sources': [],
                'quality_score': 0,
                'analysis': f"Error: {str(e)}"
            })
    
    return results

def print_evaluation_results(results: List[Dict[str, Any]]):
    """Print the evaluation results in a formatted table."""
    print("\n" + "="*100)
    print("RAG SYSTEM EVALUATION RESULTS")
    print("="*100)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Question: {result['question']}")
        print(f"   Generated Answer: {result['generated_answer']}")
        
        print("\n   Retrieved Sources:")
        for j, source in enumerate(result.get('retrieved_sources', []), 1):
            print(f"   {j}. {source.get('text', 'No text available')}")
            print(f"      Score: {source.get('score', 0.0):.4f}")
            print(f"      Product: {source.get('product', 'Unknown')}")
            print(f"      Narrative ID: {source.get('narrative_id', 'N/A')}")
        
        if 'error' in result:
            print(f"\n   Error: {result['error']}")
            
        print(f"\n   Quality Score: {result.get('quality_score', 0)}/5")
        if result.get('analysis'):
            print(f"   Analysis: {result['analysis']}")
        
        print("\n" + "-"*100)

if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Example test questions
    test_questions = [
        {
            'question': "What are the most common issues with credit card payments?",
            'expected_answer': "billing disputes late payments unauthorized transactions"
        },
        {
            'question': "How do customers report unauthorized transactions?",
            'expected_answer': "call customer service file dispute online banking"
        },
        {
            'question': "What are the typical resolution times for fraud claims?",
            'expected_answer': "10 business days investigation temporary credit"
        },
        {
            'question': "What information is needed to file a complaint?",
            'expected_answer': "account details transaction date amount description"
        },
        {
            'question': "How are billing disputes typically resolved?",
            'expected_answer': "investigation merchant contact refund if valid"
        }
    ]
    
    # Run evaluation
    results = evaluate_rag_system(rag, test_questions)
    
    # Print results
    print_evaluation_results(results)
