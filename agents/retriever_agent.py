# agents/retriever_agent.py
import logging
from typing import Dict, List, Any, Optional
from crewai import Agent, Task
from crewai_tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from services.vector_store import VectorStoreService
from data_ingestion.embeddings import EmbeddingService
import json

logger = logging.getLogger(__name__)

class DocumentSearchTool(BaseTool):
    """Tool for searching documents in the vector store"""
    name: str = "document_search"
    description: str = "Search for relevant documents using semantic similarity"
    
    def __init__(self, vector_service: VectorStoreService, **kwargs):
        super().__init__(**kwargs)
        self._vector_service = vector_service
    
    def _run(self, query: str, k: int = 5, filters: Dict = None) -> str:
        """Search for documents based on query"""
        try:
            if not self._vector_service:
                return json.dumps({"error": "Vector service not initialized", "results": []})
                
            results = self._vector_service.similarity_search(query, k=k, filters=filters)
            
            # Format results for agent consumption
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', 0.0)
                })
            
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return json.dumps({"error": str(e), "results": []})
        
class ContextRetrievalTool(BaseTool):
    """Tool for retrieving contextual information for RAG"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "context_retrieval"
    description: str = "Retrieve contextual information from documents"
    vector_service: VectorStoreService = Field(default=None)
    
    def __init__(self, vector_service: VectorStoreService, **kwargs):
        super().__init__(**kwargs)
        self.vector_service = vector_service
    
    def _run(self, query: str, context_type: str = "general", k: int = 3) -> str:
        """Retrieve context documents for RAG"""
        try:
            if not self.vector_service:
                return json.dumps({"error": "Vector service not initialized", "context": []})
                
            # Enhanced query based on context type
            enhanced_queries = {
                "financial": f"financial analysis market data {query}",
                "earnings": f"earnings results quarterly reports {query}",
                "risk": f"risk assessment portfolio exposure {query}",
                "tech": f"technology sector analysis {query}",
                "asia": f"Asia Pacific market trends {query}",
                "general": query
            }
            
            search_query = enhanced_queries.get(context_type, query)
            
            # Search with context-specific filters
            filters = {"category": context_type} if context_type != "general" else None
            results = self.vector_service.similarity_search(search_query, k=k, filters=filters)
            
            # Format for RAG consumption
            context_docs = []
            for doc in results:
                context_docs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "category": doc.metadata.get("category", "general")
                })
            
            return json.dumps(context_docs, indent=2)
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return json.dumps({"error": str(e), "context": []})

class HybridSearchTool(BaseTool):
    """Tool for hybrid semantic + keyword search"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "hybrid_search"
    description: str = "Perform hybrid search combining semantic similarity and keyword matching"
    vector_service: VectorStoreService = Field(default=None)
    
    def __init__(self, vector_service: VectorStoreService, **kwargs):
        super().__init__(**kwargs)
        self.vector_service = vector_service
    
    def _run(self, query: str, keywords: List[str] = None, k: int = 5) -> str:
        """Perform hybrid search"""
        try:
            if not self.vector_service:
                return json.dumps({"error": "Vector service not initialized", "results": []})
                
            # Get semantic results
            semantic_results = self.vector_service.similarity_search(query, k=k)
            
            # Get keyword results if keywords provided
            keyword_results = []
            if keywords:
                for keyword in keywords:
                    kw_results = self.vector_service.keyword_search(keyword, k=k//2)
                    keyword_results.extend(kw_results)
            
            # Combine and deduplicate results
            all_results = semantic_results + keyword_results
            seen_content = set()
            unique_results = []
            
            for doc in all_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(doc)
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(unique_results[:k]):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "search_type": "hybrid"
                })
            
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return json.dumps({"error": str(e), "results": []})

class RetrieverAgent:
    """Enhanced Retriever Agent for RAG operations"""
    
    def __init__(self, vector_service: VectorStoreService = None):
        try:
            self.vector_service = vector_service or VectorStoreService()
            self.embedding_service = EmbeddingService()
            
            # Initialize tools with proper vector service
            self.document_search_tool = DocumentSearchTool(self.vector_service)
            self.context_retrieval_tool = ContextRetrievalTool(self.vector_service)
            self.hybrid_search_tool = HybridSearchTool(self.vector_service)
            
            # Create CrewAI agent
            self.agent = Agent(
                role="Document Retrieval Specialist",
                goal="Retrieve the most relevant documents and context for financial queries",
                backstory="""You are an expert at finding and retrieving relevant financial documents, 
                market data, and contextual information. You specialize in semantic search, 
                document ranking, and providing precise context for financial analysis.""",
                tools=[
                    self.document_search_tool,
                    self.context_retrieval_tool,
                    self.hybrid_search_tool
                ],
                verbose=True,
                allow_delegation=False
            )
            
            logger.info("RetrieverAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RetrieverAgent: {e}")
            # Initialize with minimal functionality as fallback
            self.vector_service = None
            self.embedding_service = None
            self.agent = None
    
    def search_documents(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Direct document search method with error handling"""
        try:
            if not self.vector_service:
                logger.warning("Vector service not available, returning empty results")
                return []
                
            results = self.vector_service.similarity_search(query, k=k, filters=filters)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0.0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Direct search error: {e}")
            return []
    
    def get_rag_context(self, query: str, context_type: str = "general", k: int = 3) -> Dict[str, Any]:
        """Get context for RAG operations with fallback"""
        try:
            if not self.context_retrieval_tool:
                return {"error": "Context retrieval not available", "context": []}
                
            # Use context retrieval tool
            context_result = self.context_retrieval_tool._run(query, context_type, k)
            context_data = json.loads(context_result)
            
            if isinstance(context_data, dict) and "error" in context_data:
                return {"error": context_data["error"], "context": []}
            
            return {
                "query": query,
                "context_type": context_type,
                "context_documents": context_data,
                "document_count": len(context_data) if isinstance(context_data, list) else 0
            }
            
        except Exception as e:
            logger.error(f"RAG context error: {e}")
            return {"error": str(e), "context": []}
    
    def create_retrieval_task(self, query: str, context: Dict[str, Any] = None) -> Task:
        """Create a CrewAI task for document retrieval"""
        if not self.agent:
            raise ValueError("RetrieverAgent not properly initialized")
            
        context = context or {}
        context_type = context.get("category", "general")
        k = context.get("k", 5)
        
        task_description = f"""
        Search for and retrieve the most relevant documents for this query: "{query}"
        
        Requirements:
        - Use appropriate search strategy (semantic, hybrid, or contextual)
        - Context type: {context_type}
        - Return top {k} most relevant documents
        - Include metadata and relevance scores
        - Format results for easy consumption by other agents
        
        Additional context: {json.dumps(context, indent=2)}
        """
        
        return Task(
            description=task_description,
            agent=self.agent,
            expected_output="JSON formatted list of relevant documents with metadata and scores"
        )
    
    def perform_multi_strategy_search(self, query: str, strategies: List[str] = None) -> Dict[str, Any]:
        """Perform search using multiple strategies and combine results"""
        strategies = strategies or ["semantic", "hybrid", "contextual"]
        all_results = {}
        
        try:
            # Semantic search
            if "semantic" in strategies:
                semantic_results = self.search_documents(query, k=5)
                all_results["semantic"] = semantic_results
            
            # Hybrid search
            if "hybrid" in strategies and self.hybrid_search_tool:
                # Extract potential keywords from query
                keywords = query.split()[:3]  # Simple keyword extraction
                hybrid_result = self.hybrid_search_tool._run(query, keywords, k=5)
                try:
                    all_results["hybrid"] = json.loads(hybrid_result)
                except json.JSONDecodeError:
                    all_results["hybrid"] = []
            
            # Contextual search
            if "contextual" in strategies:
                context_result = self.get_rag_context(query, "financial", k=3)
                all_results["contextual"] = context_result.get("context_documents", [])
            
            # Combine and rank results
            combined_results = self._combine_search_results(all_results)
            
            return {
                "query": query,
                "strategies_used": strategies,
                "individual_results": all_results,
                "combined_results": combined_results,
                "total_documents": len(combined_results)
            }
            
        except Exception as e:
            logger.error(f"Multi-strategy search error: {e}")
            return {"error": str(e), "results": []}
    
    def _combine_search_results(self, strategy_results: Dict[str, List]) -> List[Dict]:
        """Combine results from multiple search strategies"""
        try:
            all_docs = []
            seen_content = set()
            
            # Weight different strategies
            strategy_weights = {
                "semantic": 1.0,
                "hybrid": 0.8,
                "contextual": 0.9
            }
            
            for strategy, results in strategy_results.items():
                weight = strategy_weights.get(strategy, 0.5)
                
                if isinstance(results, list):
                    for doc in results:
                        if isinstance(doc, dict):
                            content = doc.get("content", "")
                            content_hash = hash(content[:100]) if content else hash(str(doc))
                            
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                doc["strategy"] = strategy
                                doc["weighted_score"] = doc.get("score", 0.5) * weight
                                all_docs.append(doc)
            
            # Sort by weighted score
            all_docs.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
            return all_docs[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Result combination error: {e}")
            return []
    
    def update_document_index(self, documents: List[Dict], batch_size: int = 100) -> Dict[str, Any]:
        """Update the vector store with new documents"""
        try:
            if not self.vector_service:
                return {"success": False, "error": "Vector service not available"}
                
            results = self.vector_service.add_documents(documents, batch_size=batch_size)
            return {
                "success": True,
                "documents_added": len(documents),
                "index_stats": results
            }
        except Exception as e:
            logger.error(f"Index update error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store index"""
        try:
            if not self.vector_service:
                return {"error": "Vector service not available"}
                
            return self.vector_service.get_stats()
        except Exception as e:
            logger.error(f"Index stats error: {e}")
            return {"error": str(e)}

# Factory function for safe initialization
def get_retriever_agent(vector_service: VectorStoreService = None) -> RetrieverAgent:
    """Factory function to safely create RetrieverAgent"""
    try:
        return RetrieverAgent(vector_service)
    except Exception as e:
        logger.error(f"Failed to create RetrieverAgent: {e}")
        # Return a basic agent that won't crash the system
        return RetrieverAgent(None)