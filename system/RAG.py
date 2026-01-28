from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os
import requests
import argparse
import math

class RAGSystem:
    def __init__(self, model_name, directory_path: str = "./documents", verbose: bool = False):
        self.directory_path = directory_path
        self.vectorstore = None
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.verbose = verbose
        self.last_query = None
        self.last_results = None
        # New keyword-related attributes
        self.keyword_weights = {}  # Store IDF weights for keywords
        self.document_keywords = {}  # Store keywords per document
        self.document_contents = {}  # Store the raw content of each chunk
        self.total_documents = 0

    def _custom_text_splitter(self, text: str, separator: str = "\n\n") -> list:
        """Split text at specified separator and clean the resulting chunks."""
        chunks = text.lower().split(separator)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _extract_keywords(self, text: str) -> list:
        """Extract keywords between | | markers"""
        parts = text.split('|')
        if len(parts) >= 2:
            # Get the content between first set of | |
            keywords = [k.strip().lower() for k in parts[1].split(',')]
            return [k for k in keywords if k]
        return []

    def _calculate_keyword_weights(self):
        """Calculate IDF weights for keywords using document content"""
        keyword_doc_freq = {}
        
        # Count document frequency for each keyword with content matching
        for doc_id, keywords in self.document_keywords.items():
            seen_keywords = set()  # Track matched keywords per document
            # Get the actual document content without the prefix                    
            doc_content = self.document_contents[doc_id].lower()
            doc_content = doc_content.replace('search_document:', '').strip()
            for keyword in keywords:
                keyword = keyword.lower()
                # Check if keyword appears in document content
                if keyword in doc_content and keyword not in seen_keywords:
                    keyword_doc_freq[keyword] = keyword_doc_freq.get(keyword, 0) + 1
                    seen_keywords.add(keyword)
                    
                    if self.verbose:
                        print(f"Found keyword '{keyword}' in document {doc_id}")
        
        if self.verbose:
            print("\nKeyword frequencies in content:")
            for keyword, freq in keyword_doc_freq.items():
                print(f"{keyword}: {freq}")
        
        # Calculate IDF weights based on actual document appearances
        for doc_id, keywords in self.document_keywords.items():
            weights = {}
            doc_content = self.document_contents[doc_id].lower()
            doc_content = doc_content.replace('search_document:', '').strip()
            
            for keyword in keywords:
                keyword = keyword.lower()
                # Calculate IDF weight
                idf = math.log((self.total_documents + 1) / (keyword_doc_freq[keyword] + 1))
                # Add term frequency component
                tf = doc_content.count(keyword)
                weights[keyword] = (1 + math.log(tf + 1)) * idf  # Add 1 to handle zero counts
            
            self.keyword_weights[doc_id] = weights

    def _keyword_similarity(self, query: str, doc_id: str) -> float:
        """Calculate keyword-based distance reduction"""
        if doc_id not in self.keyword_weights or doc_id not in self.document_contents:
            return 1.0
        
        query_lower = query.lower()
        doc_weights = self.keyword_weights[doc_id]
        doc_content = self.document_contents[doc_id].lower()
        
        if self.verbose:
            print(f"\nMatching keywords for doc {doc_id}:")
            print(f"Document keywords: {list(doc_weights.keys())}")
            print(f"Query: {query_lower}")
        
        score = 0.0
        matches = []
        max_possible_score = sum(doc_weights.values())
        
        if max_possible_score == 0:
            return 1.0
        
        # Check for keyword matches in query and document content
        for keyword, weight in doc_weights.items():
            if keyword in query_lower and keyword in doc_content:
                score += weight
                matches.append(keyword)
            # Check for substring matches
            elif any(term in keyword for term in query_lower.split() if len(term) >= 3):
                score += weight * 0.75  # Half score for partial matches
                matches.append(f"{keyword}*")
        
        normalized_distance = 1 - (score / max_possible_score)
        
        if self.verbose and matches:
            print(f"Matched terms: {matches}")
            print(f"Score: {score:.4f}")
            print(f"Distance modifier: {normalized_distance:.4f}")
        
        return normalized_distance

    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            requests.get("http://localhost:11434/api/embeddings")
            return True
        except requests.ConnectionError:
            print("Connection Error")
            return False

    def initialize(self) -> bool:
        """Initialize the RAG system"""
        if not self.check_ollama_connection():
            return False

        if not os.path.exists(self.directory_path):
            return False
        
        txt_files = [f for f in os.listdir(self.directory_path) if f.endswith('.txt')]
        if not txt_files:
            return False

        try:
            self.vectorstore = self._setup_vectorstore()
            return True
        except Exception:
            return False

    def _setup_vectorstore(self):
        """Set up vector store in memory"""
        return self._build_new_vectorstore()

    def _build_new_vectorstore(self):
        """Build new vector store from documents with keyword extraction"""
        loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True 
        )
        
        documents = loader.load()
        self.total_documents = len(documents)
        
        # Reset keyword tracking
        self.document_keywords.clear()
        self.keyword_weights.clear()
        
        # Create splits and process keywords
        custom_splits = []
        all_keywords = set()
        
        print("\nProcessing documents and extracting keywords:")
        print("-" * 50)
        
        for doc_idx, doc in enumerate(documents):
            chunks = self._custom_text_splitter(doc.page_content)
            for chunk_idx, chunk in enumerate(chunks):
                if chunk:
                    doc_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                    keywords = self._extract_keywords(chunk)
                    
                    # Store original chunk content without search_document prefix
                    self.document_contents[doc_id] = chunk
                    
                    if keywords:
                        self.document_keywords[doc_id] = keywords
                        all_keywords.update(keywords)
                        if self.verbose:
                            print(f"\nChunk {doc_idx}.{chunk_idx}:")
                            print(f"Keywords: {', '.join(keywords)}")
                    
                    # Add search_document prefix only for vectorstore
                    custom_splits.append(Document(
                        page_content="search_document:" + chunk,
                        metadata={
                            **doc.metadata,
                            'doc_id': doc_id,
                            'keywords': keywords
                        }
                    ))
        
        # Calculate keyword weights after processing all documents
        self._calculate_keyword_weights()
        
        # Create vector store with L2 normalization (in memory only)
        return FAISS.from_documents(
            custom_splits, 
            self.embeddings,
            normalize_L2=True
        )

    def search(self, query: str, k: int = 3, semantic_weight: float = 0.65, max_dissimilarity: float = 0.8):
        """Hybrid search combining Euclidean distance with keyword-based distance reduction"""
        if not self.vectorstore:
            return []
        query = query.lower()
        try:
            print(f"\nProcessing query: '{query}'")
            print("-" * 50)
            
            # Stage 1: Get more candidates using vector similarity (k * 3)
            initial_k = k * 3
            results = self.vectorstore.similarity_search_with_score(
                "search_query: " + query, 
                k=initial_k,
                fetch_k= 20,
                normalize_L2=True
            )
            
            print(f"Retrieved {initial_k} initial candidates")
            
            # Stage 2: Rerank using hybrid scoring
            hybrid_results = []
            for doc, vector_score in results:
                doc_id = doc.metadata.get('doc_id')
                if doc_id:
                    # Get keyword distance modifier (1.0 = no effect, 0.0 = maximum reduction)
                    keyword_modifier = self._keyword_similarity(query, doc_id)
                    # Combine scores - reduce vector distance based on keyword matches

                    combined_score = semantic_weight*vector_score + ((1 - semantic_weight) * keyword_modifier)
                    hybrid_results.append((doc, {
                        'combined': combined_score,
                        'vector': vector_score,
                        'keyword': keyword_modifier
                    }))
                else:
                    hybrid_results.append((doc, {
                        'combined': vector_score,
                        'vector': vector_score,
                        'keyword': 1.0
                    }))                    
            
            # Sort by combined score (lower is better for distances)
            hybrid_results.sort(key=lambda x: x[1]['combined'])
            final_results = hybrid_results[:k]
            
            print(f"\nTop {k} results after reranking:")
            for doc, scores in final_results:
                print(f"\nDocument: {doc.metadata.get('doc_id')}")
                print(f"Keywords: {', '.join(doc.metadata.get('keywords', []))}")
                print(f"Vector score: {scores['vector']:.4f}")
                print(f"Keyword score: {scores['keyword']:.4f}")
                print(f"Combined score: {scores['combined']:.4f}")
                print(f"Content preview: {doc.page_content[:200]}...")
            
            self.last_query = query
            self.last_results = final_results
            
            return [(doc, score['combined']) for doc, score in final_results if score['combined'] < max_dissimilarity]
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

def main():
    parser = argparse.ArgumentParser(description='RAG System')
    args = parser.parse_args()
    
    print("Initializing RAG system...")
    rag = RAGSystem(verbose=True)
    if not rag.initialize():
        print("Failed to initialize RAG system")
        return

    print("\nRAG system ready! Enter your queries (type 'quit' to exit)")
    print("-" * 50)

    while True:
        query = input("\nEnter query: ")
        if query.lower() == 'quit':
            break
        results = rag.search(query)

if __name__ == "__main__":
    main()