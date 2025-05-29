# data_ingestion/embeddings.py
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import hashlib
import json
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# OpenAI embeddings (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Hugging Face transformers (alternative)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Simple file-based cache for embeddings"""
    
    def __init__(self, cache_dir: str = "data/embedding_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        self.metadata = self._load_metadata()
    
    def _generate_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination"""
        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired"""
        key = self._generate_key(text, model_name)
        
        if key in self.cache:
            # Check if not expired
            if key in self.metadata:
                created_time = datetime.fromisoformat(self.metadata[key]['created'])
                if datetime.now() - created_time < timedelta(hours=self.ttl_hours):
                    return self.cache[key]
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.metadata[key]
        
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]):
        """Store embedding in cache"""
        key = self._generate_key(text, model_name)
        self.cache[key] = embedding
        self.metadata[key] = {
            'created': datetime.now().isoformat(),
            'model': model_name,
            'text_length': len(text)
        }
        
        # Periodic cleanup and save
        if len(self.cache) % 100 == 0:
            self._cleanup_expired()
            self._save_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
        return {}
    
    def _load_metadata(self) -> Dict:
        """Load metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache metadata: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save embedding cache: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, meta in self.metadata.items():
            created_time = datetime.fromisoformat(meta['created'])
            if current_time - created_time >= timedelta(hours=self.ttl_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.metadata.pop(key, None)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired embeddings")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_embeddings": len(self.cache),
            "cache_size_mb": len(pickle.dumps(self.cache)) / (1024 * 1024),
            "models_used": list(set(meta.get('model', 'unknown') for meta in self.metadata.values())),
            "oldest_entry": min(self.metadata.values(), key=lambda x: x['created'])['created'] if self.metadata else None,
            "newest_entry": max(self.metadata.values(), key=lambda x: x['created'])['created'] if self.metadata else None
        }

class SentenceTransformerEmbedder:
    """Sentence Transformers based embedding service"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self.model:
            return 384  # Default for MiniLM
        return self.model.get_sentence_embedding_dimension()

class OpenAIEmbedder:
    """OpenAI API based embedding service"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        logger.info(f"Initialized OpenAI embedder with model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"OpenAI batch embedding error: {e}")
                raise
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        # Known dimensions for OpenAI models
        dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimensions.get(self.model_name, 1536)

class HuggingFaceEmbedder:
    """Hugging Face transformers based embedding service"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"Loaded HuggingFace model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()[0].tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            except Exception as e:
                logger.error(f"HuggingFace batch embedding error: {e}")
                raise
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.config.hidden_size
        return 384  # Default

class EmbeddingService:
    """Main embedding service with multiple backend support"""
    
    def __init__(self, 
                 backend: str = "sentence_transformers",
                 model_name: str = None,
                 use_cache: bool = True,
                 cache_ttl_hours: int = 24,
                 **kwargs):
        
        self.backend = backend
        self.use_cache = use_cache
        self.cache = EmbeddingCache(ttl_hours=cache_ttl_hours) if use_cache else None
        
        # Initialize embedder based on backend
        if backend == "sentence_transformers":
            model_name = model_name or "all-MiniLM-L6-v2"
            self.embedder = SentenceTransformerEmbedder(model_name)
        elif backend == "openai":
            model_name = model_name or "text-embedding-ada-002"
            self.embedder = OpenAIEmbedder(model_name, **kwargs)
        elif backend == "huggingface":
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.embedder = HuggingFaceEmbedder(model_name)
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")
        
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized EmbeddingService with {backend} backend, model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text with caching"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        if self.use_cache:
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding
        embedding = self.embedder.embed_text(text)
        
        # Cache the result
        if self.use_cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        # Check cache for individual texts
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        if self.use_cache:
            for i, text in valid_texts:
                cached_embedding = self.cache.get(text, self.model_name)
                if cached_embedding:
                    embeddings[i] = cached_embedding
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = [text for _, text in valid_texts]
            indices_to_embed = [i for i, _ in valid_texts]
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            batch_size = batch_size or (32 if self.backend == "sentence_transformers" else 8)
            new_embeddings = self.embedder.embed_batch(texts_to_embed, batch_size)
            
            # Store results and cache
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                if self.use_cache:
                    self.cache.set(texts[idx], self.model_name, embedding)
        
        # Fill None values with zero embeddings (for empty texts)
        dimension = self.get_dimension()
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = [0.0] * dimension
        
        return embeddings
    
    async def embed_text_async(self, text: str) -> List[float]:
        """Async version of embed_text"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.embed_text, text)
    
    async def embed_batch_async(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Async version of embed_batch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.embed_batch, texts, batch_size)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedder.get_dimension()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        stats = {
            "backend": self.backend,
            "model_name": self.model_name,
            "embedding_dimension": self.get_dimension(),
            "cache_enabled": self.use_cache
        }
        
        if self.use_cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.use_cache:
            self.cache.cache.clear()
            self.cache.metadata.clear()
            self.cache._save_cache()
            logger.info("Embedding cache cleared")
    
    def precompute_embeddings(self, texts: List[str], batch_size: int = None) -> Dict[str, Any]:
        """Precompute and cache embeddings for a list of texts"""
        try:
            start_time = datetime.now()
            embeddings = self.embed_batch(texts, batch_size)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "texts_processed": len(texts),
                "embeddings_generated": len([e for e in embeddings if e]),
                "processing_time_seconds": processing_time,
                "embeddings_per_second": len(texts) / processing_time if processing_time > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {e}")
            return {"success": False, "error": str(e)}

# Factory function for easy initialization
def get_embedding_service(backend: str = None, **kwargs) -> EmbeddingService:
    """Factory function to create embedding service"""
    backend = backend or os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
    return EmbeddingService(backend=backend, **kwargs)

# Utility functions
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two embeddings"""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.linalg.norm(a_np - b_np)

def find_most_similar(query_embedding: List[float], 
                     candidate_embeddings: List[List[float]], 
                     top_k: int = 5) -> List[int]:
    """Find indices of most similar embeddings"""
    similarities = [
        cosine_similarity(query_embedding, candidate) 
        for candidate in candidate_embeddings
    ]
    
    # Get top-k indices
    top_indices = sorted(range(len(similarities)), 
                        key=lambda i: similarities[i], 
                        reverse=True)[:top_k]
    
    return top_indices