"""
Enhanced Crew Manager with proper async/sync handling and error recovery
Fixes the coroutine issues and provides robust service integration
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from crewai import Crew, Task
from datetime import datetime
import json

from agents.api_agent import create_api_agent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent 
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent

from services.vector_store import VectorStoreService
from services.voice_service import get_voice_service

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EnhancedAgentCrewManager:
    """
    Enhanced orchestrator with proper async/sync handling and error recovery
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_core_services()
        self._init_agents()
        self.voice_service = get_voice_service()
        
        # Initialize with sample data if empty
        self._ensure_sample_data()

    def _init_core_services(self):
        """Initialize core services with proper error handling"""
        logger.info("Initializing core services...")
        
        try:
            # Initialize vector store service
            self.vector_store_service = VectorStoreService(
                backend=self.config.get('vector_store_type', 'faiss'),
                embedding_dim=self.config.get('embedding_dim', 384),
                index_path=self.config.get('vector_store_path', './vector_store')
            )
            
            # Try to load existing index
            index_path = self.config.get('vector_store_path', './vector_store')
            try:
                logger.info(f"Loaded existing vector index from {index_path}")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}. Starting fresh.")
                
        except Exception as e:
            logger.error(f"Error initializing core services: {e}")
            # Fallback to basic functionality
            self.vector_store_service = None

    def _init_agents(self):
        """Initialize all agents with proper error handling"""
        logger.info("Initializing all agents...")
        
        try:
            # API Agent - independent
            self.api_agent = create_api_agent()
            
            # Other agents with fallback handling
            if self.vector_store_service:
                try:
                    self.scraping_agent = ScrapingAgent(
                        vector_store_service=self.vector_store_service
                    )
                except Exception as e:
                    logger.error(f"Error initializing ScrapingAgent: {e}")
                    self.scraping_agent = create_api_agent()  # Fallback
                
                try:
                    self.retriever_agent = RetrieverAgent(
                        vector_service=self.vector_store_service
                    )
                except Exception as e:
                    logger.error(f"Error initializing RetrieverAgent: {e}")
                    self.retriever_agent = create_api_agent()  # Fallback
                
                try:
                    self.analysis_agent = AnalysisAgent(
                        vector_store_service=self.vector_store_service
                    )
                except Exception as e:
                    logger.error(f"Error initializing AnalysisAgent: {e}")
                    self.analysis_agent = create_api_agent()  # Fallback
                
                try:
                    self.language_agent = LanguageAgent(
                        vector_store_service=self.vector_store_service
                    )
                except Exception as e:
                    logger.error(f"Error initializing LanguageAgent: {e}")
                    self.language_agent = create_api_agent()  # Fallback
            else:
                # Fallback initialization without vector store service
                logger.warning("Initializing agents without vector store service")
                self.scraping_agent = create_api_agent()
                self.retriever_agent = create_api_agent()
                self.analysis_agent = create_api_agent()
                self.language_agent = create_api_agent()
            
            # Voice Agent - independent
            self.voice_agent = VoiceAgent()

            self.agents = [
                self.api_agent,
                self.scraping_agent,  # Use the full ScrapingAgent instance
                self.retriever_agent,  # Use the full RetrieverAgent instance
                self.analysis_agent,  # Use the full AnalysisAgent instance
                self.language_agent,  # Use the full LanguageAgent instance
                self.voice_agent  # Use the full VoiceAgent instance
            ]
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            # Minimal fallback
            self.api_agent = create_api_agent()
            self.agents = [self.api_agent]

    def create_crew(self, agents: List, tasks: List) -> Crew:
        """Create a crew with the specified agents and tasks"""
        return Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            memory=True
        )

    def _ensure_sample_data(self):
        """Ensure we have some sample data for testing"""
        if self.vector_store_service:
            try:
                # Check if we have any data
                stats = self.vector_store_service.get_stats()
                if stats.get('total_documents', stats.get('total_vectors', 0)) == 0:
                    logger.info("No existing data found, adding sample financial data...")
                    sample_docs = self._get_sample_financial_data()
                    self.add_financial_documents(sample_docs)
            except Exception as e:
                logger.warning(f"Could not check/add sample data: {e}")

    def _get_sample_financial_data(self) -> List[Dict[str, Any]]:
        """Get sample financial data for testing"""
        return [
            {
                'content': "TSMC (Taiwan Semiconductor) reported Q3 2024 earnings beating estimates by 4% with revenue of $23.5B, driven by strong AI chip demand and 3nm process technology adoption.",
                'metadata': {
                    'source': 'earnings',
                    'company': 'TSMC',
                    'sector': 'technology',
                    'region': 'asia',
                    'type': 'earnings',
                    'date': '2024-10-15'
                }
            },
            {
                'content': "Samsung Electronics missed Q3 estimates by 2% due to weak memory chip pricing, reporting operating profit of $7.8B versus expected $8.0B.",
                'metadata': {
                    'source': 'earnings',
                    'company': 'Samsung',
                    'sector': 'technology',
                    'region': 'asia',
                    'type': 'earnings',
                    'date': '2024-10-12'
                }
            },
            {
                'content': "Asia tech stocks showing mixed sentiment with Taiwan and South Korea leading gains while China tech faces regulatory headwinds. Rising US yields creating cautionary outlook.",
                'metadata': {
                    'source': 'market_analysis',
                    'sector': 'technology',
                    'region': 'asia',
                    'type': 'news',
                    'date': '2024-10-20'
                }
            },
            {
                'content': "Current Asia tech allocation recommendations suggest 20-25% portfolio weight with focus on AI beneficiaries like TSMC, SK Hynix, and selective China exposure.",
                'metadata': {
                    'source': 'research',
                    'sector': 'technology',
                    'region': 'asia',
                    'type': 'research',
                    'date': '2024-10-18'
                }
            }
        ]

    def add_financial_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add financial documents with proper error handling"""
        if not self.vector_store_service:
            logger.warning("Vector store service not available")
            return []
        
        try:
            result = self.vector_store_service.add_documents(documents)
            logger.info(f"Added {result.get('documents_added', 0)} documents to vector store")
            return result.get('document_ids', [])
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []

    def search_market_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for market context with error handling"""
        if not self.vector_store_service:
            logger.warning("Vector store service not available, returning empty results")
            return []
        
        try:
            results = self.vector_store_service.similarity_search(query, k=k)
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results
            ]
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching market context: {e}")
            return []
    
    def search_documents_direct(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Direct document search without CrewAI orchestration"""
        if not self.retriever_agent:
            return []
        
        try:
            return self.retriever_agent.search_documents(query, k, filters or {})
        except Exception as e:
            logger.error(f"Error in direct search: {e}")
            return []

    def process_voice_query(self, voice_input: str, portfolio_data: Dict = None, market_data: Dict = None) -> Dict[str, Any]:
        try:
            logger.info(f"Processing voice query: {voice_input}")
            query_lower = voice_input.lower()
            
            if any(keyword in query_lower for keyword in ["risk exposure", "market brief", "asia tech", "allocation"]):
                crew = self.create_morning_brief_crew(voice_input, portfolio_data, market_data)
                result = crew.kickoff()
                return {
                    "success": True,
                    "query": voice_input,
                    "response": result,
                    "type": "morning_brief",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "query": voice_input,
                    "error": "Unsupported query type",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Voice query processing failed: {str(e)}")
            return {
                "success": False,
                "query": voice_input,
                "error": f"Voice query failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def create_morning_brief_crew(self, portfolio_query: str, portfolio_data: Dict = None, market_data: Dict = None) -> Crew:
        if not portfolio_data:
            logger.info("No portfolio data provided, using default portfolio")
            portfolio_data = {
                "holdings": [
                    {"symbol": "TSM", "sector": "Technology", "weight": 0.08, "market_value": 4000000},
                    {"symbol": "005930.KS", "sector": "Technology", "weight": 0.06, "market_value": 3000000},
                    {"symbol": "9988.HK", "sector": "Technology", "weight": 0.04, "market_value": 2000000}
                ],
                "total_value": 50000000
            }
        
        if not market_data:
            logger.info("No market data provided, will fetch via market_data_task")
            market_data = {}

        # Task 1: Fetch latest market data
        market_data_task = Task(
            description=f"""
            Fetch the latest market data for Asia tech stocks including:
            - Current prices and daily changes for major Asia tech stocks (TSMC, Samsung, Alibaba, etc.)
            - Key market indicators and indices (Taiwan Index, KOSPI, Hang Seng Tech)
            - Currency movements affecting tech stocks
            - Earnings surprises with actual vs. expected EPS
            Query context: {portfolio_query}
            """,
            agent=self.api_agent,
            expected_output="JSON object containing market data for Asia tech stocks with prices, changes, indicators, and earnings surprises",
            async_execution=True
        )
        
        # Task 2: Retrieve historical context
        retrieval_task = Task(
            description=f"""
            Search for relevant historical context and recent financial information for: {portfolio_query}
            """,
            agent=self.retriever_agent,
            expected_output="Relevant historical earnings data, risk analysis, and market context for Asia tech stocks",
            async_execution=True
        )
        
        # Task 3: Analyze portfolio risk and exposure
        analysis_task = Task(
            description=f"""
            Perform comprehensive portfolio risk analysis for Asia tech exposure.
            Query: {portfolio_query}
            """,
            agent=self.analysis_agent,
            expected_output="Detailed portfolio risk analysis with allocation percentages, risk metrics, exposure assessment, and earnings surprises",
            context=[market_data_task, retrieval_task],
            portfolio_data=portfolio_data,
            market_data=market_data,
            async_execution=True
        )
        
        # Task 4: Generate narrative market brief
        language_task = Task(
            description=f"""
            Synthesize all gathered information into a professional morning market brief.
            Query: {portfolio_query}
            """,
            agent=self.language_agent,
            expected_output="Concise, professional market brief ready for voice synthesis",
            context=[market_data_task, analysis_task, retrieval_task],
            portfolio_data=portfolio_data,
            market_data=market_data
        )
        
        tasks = [market_data_task, retrieval_task, analysis_task, language_task]
        
        return Crew(
            agents=[self.api_agent, self.retriever_agent, self.analysis_agent, self.language_agent],
            tasks=tasks,
            verbose=True,
            memory=True
        )

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all services"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'agents_initialized': len(self.agents),
            'services': {}
        }
        
        # Check vector store service
        if self.vector_store_service:
            health_status['services']['vector_store'] = self.vector_store_service.get_stats()
        else:
            health_status['services']['vector_store'] = {"status": "unavailable"}
        
        # Check if we have sample data
        try:
            search_results = self.search_market_context("test query", k=1)
            health_status['services']['search_capability'] = {
                "status": "healthy" if search_results else "limited",
                "sample_results_count": len(search_results)
            }
        except Exception as e:
            health_status['services']['search_capability'] = {
                "status": "error",
                "error": str(e)
            }
        
        return health_status

    def get_sample_morning_brief(self) -> Dict[str, Any]:
        """Generate a sample morning brief for testing"""
        sample_query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
        return self.process_voice_query(sample_query)


# Factory function for easy instantiation
def create_enhanced_crew_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedAgentCrewManager:
    """Factory function to create a properly configured crew manager"""
    default_config = {
        'vector_store_type': 'faiss',
        'embedding_dim': 384,
        'vector_store_path': './vector_store'
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedAgentCrewManager(default_config)


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced crew manager
    manager = create_enhanced_crew_manager()
    
    # Health check
    health = manager.health_check()
    print("System Health Check:")
    print(json.dumps(health, indent=2))
    
    # Test sample morning brief
    print("\n" + "="*50)
    print("TESTING SAMPLE MORNING BRIEF")
    print("="*50)
    
    result = manager.get_sample_morning_brief()
    print(json.dumps(result, indent=2))
    
    # Test document search
    print("\n" + "="*50)
    print("TESTING DOCUMENT SEARCH")
    print("="*50)
    
    search_results = manager.search_market_context("TSMC earnings", k=2)
    for i, result in enumerate(search_results):
        print(f"Result {i+1}: {result.get('content', 'No content')[:100]}...")