# services/vector_store.py
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import pickle
import json
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

# For Pinecone integration (optional)
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from data_ingestion.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document class for vector store"""
    page_content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

class FAISSVectorStore:
    """FAISS-based vector store implementation"""
    
    def __init__(self, embedding_dim: int = 384, index_path: str = "data/vector_store"):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for similarity
        self.index_to_docstore_id = {}
        self.docstore = {}
        self.metadata_store = {}
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        try:
            doc_ids = []
            embeddings = []
            
            for i, doc in enumerate(documents):
                # Generate document ID
                doc_id = self._generate_doc_id(doc.page_content)
                doc_ids.append(doc_id)
                
                # Store document and metadata
                self.docstore[doc_id] = doc.page_content
                self.metadata_store[doc_id] = doc.metadata
                
                # Prepare embedding
                if doc.embedding:
                    embeddings.append(doc.embedding)
                else:
                    # This would require embedding service - placeholder for now
                    embeddings.append(np.random.random(self.embedding_dim).tolist())
                
                # Update index mapping
                current_index = self.index.ntotal + i
                self.index_to_docstore_id[current_index] = doc_id
            
            # Add embeddings to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            
            # Save index
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Convert query to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Retrieve documents
            documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    doc_id = self.index_to_docstore_id.get(idx)
                    if doc_id:
                        content = self.docstore.get(doc_id, "")
                        metadata = self.metadata_store.get(doc_id, {})
                        
                        doc = Document(
                            page_content=content,
                            metadata=metadata,
                            score=float(score)
                        )
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {e}")
            return []
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
            
            # Save metadata
            with open(self.index_path / "docstore.pkl", "wb") as f:
                pickle.dump(self.docstore, f)
            
            with open(self.index_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata_store, f)
            
            with open(self.index_path / "index_mapping.pkl", "wb") as f:
                pickle.dump(self.index_to_docstore_id, f)
                
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            index_file = self.index_path / "faiss.index"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(self.index_path / "docstore.pkl", "rb") as f:
                    self.docstore = pickle.load(f)
                
                with open(self.index_path / "metadata.pkl", "rb") as f:
                    self.metadata_store = pickle.load(f)
                
                with open(self.index_path / "index_mapping.pkl", "rb") as f:
                    self.index_to_docstore_id = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} documents")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")

class PineconeVectorStore:
    """Pinecone-based vector store implementation"""
    
    def __init__(self, index_name: str = "finance-assistant", dimension: int = 384):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        pinecone.init(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"))
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone"""
        try:
            vectors = []
            doc_ids = []
            
            for doc in documents:
                doc_id = self._generate_doc_id(doc.page_content)
                doc_ids.append(doc_id)
                
                # Prepare vector for Pinecone
                vector_data = {
                    "id": doc_id,
                    "values": doc.embedding or np.random.random(self.dimension).tolist(),
                    "metadata": {
                        **doc.metadata,
                        "content": doc.page_content[:1000]  # Pinecone metadata limit
                    }
                }
                vectors.append(vector_data)
            
            # Upsert to Pinecone
            self.index.upsert(vectors)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], k: int = 5, filters: Dict = None) -> List[Document]:
        """Search Pinecone for similar documents"""
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filters
            )
            
            # Convert to Document objects
            documents = []
            for match in results.matches:
                metadata = dict(match.metadata)
                content = metadata.pop("content", "")
                
                doc = Document(
                    page_content=content,
                    metadata=metadata,
                    score=match.score
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in Pinecone similarity search: {e}")
            return []
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(content.encode()).hexdigest()

class VectorStoreService:
    """Main vector store service with multiple backend support"""
    
    def __init__(self, backend: str = "faiss", **kwargs):
        self.backend = backend
        self.embedding_service = EmbeddingService()
        
        # Initialize vector store backend
        if backend == "faiss":
            self.store = FAISSVectorStore(**kwargs)
        elif backend == "pinecone":
            self.store = PineconeVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"Initialized VectorStoreService with {backend} backend")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100) -> Dict[str, Any]:
        """Add documents to vector store with embeddings"""
        try:
            # Convert dict documents to Document objects with embeddings
            doc_objects = []
            
            for doc_dict in documents:
                content = doc_dict.get("content", "")
                metadata = doc_dict.get("metadata", {})
                
                # Generate embedding
                embedding = self.embedding_service.embed_text(content)
                
                doc = Document(
                    page_content=content,
                    metadata=metadata,
                    embedding=embedding
                )
                doc_objects.append(doc)
            
            # Add in batches
            all_doc_ids = []
            for i in range(0, len(doc_objects), batch_size):
                batch = doc_objects[i:i + batch_size]
                doc_ids = self.store.add_documents(batch)
                all_doc_ids.extend(doc_ids)
            
            return {
                "success": True,
                "documents_added": len(documents),
                "document_ids": all_doc_ids,
                "backend": self.backend
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "error": str(e)}
    
    def similarity_search(self, query: str, k: int = 5, filters: Dict = None) -> List[Document]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search vector store
            if self.backend == "faiss":
                return self.store.similarity_search(query_embedding, k)
            elif self.backend == "pinecone":
                return self.store.similarity_search(query_embedding, k, filters)
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def keyword_search(self, keyword: str, k: int = 5) -> List[Document]:
        """Simple keyword-based search (fallback)"""
        try:
            # For now, use embedding search with keyword
            # In production, you might want a separate keyword index
            return self.similarity_search(keyword, k)
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, keywords: List[str] = None, k: int = 5) -> List[Document]:
        """Combine semantic and keyword search"""
        try:
            # Get semantic results
            semantic_results = self.similarity_search(query, k)
            
            # Get keyword results
            keyword_results = []
            if keywords:
                for keyword in keywords:
                    kw_results = self.keyword_search(keyword, k//2)
                    keyword_results.extend(kw_results)
            
            # Combine and deduplicate
            all_results = semantic_results + keyword_results
            seen_ids = set()
            unique_results = []
            
            for doc in all_results:
                content_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_id not in seen_ids:
                    seen_ids.add(content_id)
                    unique_results.append(doc)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if self.backend == "faiss":
                return {
                    "backend": "faiss",
                    "total_documents": self.store.index.ntotal,
                    "embedding_dimension": self.store.embedding_dim,
                    "index_path": str(self.store.index_path)
                }
            elif self.backend == "pinecone":
                stats = self.store.index.describe_index_stats()
                return {
                    "backend": "pinecone",
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_name": self.store.index_name
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Delete documents from vector store"""
        try:
            if self.backend == "pinecone":
                self.store.index.delete(ids=doc_ids)
                return {"success": True, "deleted_count": len(doc_ids)}
            else:
                # FAISS doesn't support direct deletion
                return {"success": False, "error": "FAISS backend doesn't support deletion"}
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {"success": False, "error": str(e)}
    
    def update_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
        """Recompute embeddings for existing documents"""
        try:
            # This would require rebuilding the entire index
            # Implementation depends on specific requirements
            return {"success": False, "error": "Not implemented"}
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            return {"success": False, "error": str(e)}

# Factory function for easy initialization
def get_vector_store(backend: str = None, **kwargs) -> VectorStoreService:
    """Factory function to create vector store service"""
    backend = backend or os.getenv("VECTOR_STORE_BACKEND", "faiss")
    return VectorStoreService(backend=backend, **kwargs)