# orchestrator/task_router.py

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from orchestrator.crew_manager import EnhancedAgentCrewManager, create_enhanced_crew_manager
from services.voice_service import get_voice_service

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TaskRouter:
    """Routes tasks to appropriate agent workflows with proper async/sync handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the task router with proper configuration"""
        try:
            self.crew_manager = create_enhanced_crew_manager(config)
            self.voice_service = get_voice_service()
            self._loop = None
            logger.info("Task router initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize task router: {e}")
            # Initialize with minimal functionality
            self.crew_manager = None
            self.voice_service = get_voice_service()

    def _get_event_loop(self):
        """Get or create event loop for async operations"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def route(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point to route a request to the appropriate handler.
        
        Args:
            request: Dict with keys like 'input_type', 'query', 'audio', etc.
        
        Returns:
            Dict with the response from the appropriate handler.
        """
        try:
            input_type = request.get("input_type", "text")
            logger.info(f"Routing input of type: {input_type}")

            if input_type == "voice":
                return self._handle_voice_input_sync(request)
            elif input_type == "text":
                return self._handle_text_input(request)
            else:
                return {
                    "error": f"Unsupported input_type: {input_type}",
                    "supported_types": ["text", "voice"],
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Task routing failed: {e}")
            return {
                "error": f"Routing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _handle_voice_input_sync(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle speech input synchronously by running async code in event loop"""
        try:
            # Get audio input
            audio_input = request.get("audio")
            if not audio_input:
                return {
                    "error": "No audio input provided",
                    "timestamp": datetime.now().isoformat()
                }

            # Handle voice processing in sync context
            loop = self._get_event_loop()
            
            # If we're already in a running loop, handle differently
            try:
                if loop.is_running():
                    # Create a new loop in a thread for this operation
                    import concurrent.futures
                    import threading
                    
                    def run_async_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self._handle_voice_input_async(request)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        result = future.result(timeout=30)  # 30 second timeout
                        return result
                else:
                    # Run in current loop
                    return loop.run_until_complete(
                        self._handle_voice_input_async(request)
                    )
            except Exception as loop_error:
                logger.error(f"Event loop error: {loop_error}")
                # Fallback: try direct text processing if we can extract transcript
                audio_text = request.get("transcript") or request.get("query")
                if audio_text:
                    return self._handle_text_input({"query": audio_text})
                else:
                    return {
                        "error": "Voice processing failed and no transcript available",
                        "details": str(loop_error),
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            logger.error(f"Voice input handling failed: {e}")
            return {
                "error": f"Voice processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_voice_input_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice input asynchronously"""
        try:
            audio_input = request.get("audio")
            
            if self.voice_service:
                # Use voice service for STT
                result = await self.voice_service.process_voice_query(audio_input)
                if result.get("transcript"):
                    # Process the transcript as text
                    text_request = {"query": result["transcript"]}
                    text_result = self._handle_text_input(text_request)
                    
                    # Combine voice and text results
                    return {
                        "input_type": "voice",
                        "transcript": result["transcript"],
                        "text_response": text_result,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "error": "Speech-to-text conversion failed",
                        "details": result,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "error": "Voice service not available",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Async voice processing failed: {e}")
            return {
                "error": f"Voice processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _handle_text_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text-based query and route to appropriate workflow."""
        try:
            query = request.get("query", "").strip()
            
            if not query:
                return {
                    "error": "Empty query provided",
                    "timestamp": datetime.now().isoformat()
                }

            logger.info(f"Processing text query: {query}")
            
            # Check if crew manager is available
            if not self.crew_manager:
                return {
                    "error": "Crew manager not available",
                    "fallback_response": "System is experiencing technical difficulties. Please try again later.",
                    "timestamp": datetime.now().isoformat()
                }

            # Determine query type and route appropriately
            query_lower = query.lower()
            
            # Morning brief queries
            if any(keyword in query_lower for keyword in [
                "risk exposure", "market brief", "asia tech", "allocation", 
                "earnings surprise", "portfolio", "morning brief"
            ]):
                return self._handle_morning_brief(query)
            
            # Market data queries
            elif any(keyword in query_lower for keyword in [
                "price", "market data", "stock price", "quote", "ticker"
            ]):
                return self._handle_market_data_query(query)
            
            # General financial queries
            elif any(keyword in query_lower for keyword in [
                "earnings", "financial", "analysis", "report", "news"
            ]):
                return self._handle_general_financial_query(query)
            
            # Default case
            else:
                return self._handle_general_query(query)

        except Exception as e:
            logger.error(f"Text input handling failed: {e}")
            return {
                "error": f"Text processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _handle_morning_brief(self, query: str) -> Dict[str, Any]:
        """Handle morning market brief queries"""
        try:
            logger.info(f"Processing morning brief query: {query}")
            
            # Use the crew manager's voice query processing
            result = self.crew_manager.process_voice_query(query)
            
            if result.get("success"):
                return {
                    "query_type": "morning_brief",
                    "query": query,
                    "response": result.get("response"),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "query_type": "morning_brief",
                    "query": query,
                    "error": result.get("error", "Morning brief generation failed"),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Morning brief handling failed: {e}")
            return {
                "query_type": "morning_brief",
                "query": query,
                "error": f"Morning brief failed: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def _handle_market_data_query(self, query: str) -> Dict[str, Any]:
        """Handle market data queries"""
        try:
            logger.info(f"Processing market data query: {query}")
            
            # For now, use the general crew processing
            result = self.crew_manager.process_voice_query(query)
            
            return {
                "query_type": "market_data",
                "query": query,
                "response": result.get("response", "Market data query processed"),
                "success": result.get("success", True),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Market data query handling failed: {e}")
            return {
                "query_type": "market_data",
                "query": query,
                "error": f"Market data query failed: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def _handle_general_financial_query(self, query: str) -> Dict[str, Any]:
        """Handle general financial queries"""
        try:
            logger.info(f"Processing general financial query: {query}")
            
            # Search for relevant context first
            context_results = self.crew_manager.search_market_context(query, k=3)
            
            if context_results:
                # Use context to provide response
                context_summary = "\n".join([
                    r.get('content', '')[:200] + "..." if len(r.get('content', '')) > 200 
                    else r.get('content', '') 
                    for r in context_results[:2]
                ])
                
                response = f"Based on available financial data:\n\n{context_summary}"
                
                return {
                    "query_type": "general_financial",
                    "query": query,
                    "response": response,
                    "context_sources": len(context_results),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "query_type": "general_financial",
                    "query": query,
                    "response": "I don't have specific information about that query. Please provide more details or ask about Asia tech stock risk exposure.",
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"General financial query handling failed: {e}")
            return {
                "query_type": "general_financial",
                "query": query,
                "error": f"Financial query failed: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries that don't fit specific categories"""
        try:
            return {
                "query_type": "general",
                "query": query,
                "response": "I'm a financial assistant focused on market analysis and portfolio management. Please ask me about:\n- Risk exposure in Asia tech stocks\n- Market briefings and analysis\n- Earnings surprises and financial data\n- Portfolio allocation recommendations",
                "suggestions": [
                    "What's our risk exposure in Asia tech stocks today?",
                    "Give me a morning market brief",
                    "Highlight any earnings surprises",
                    "Analyze our portfolio allocation"
                ],
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"General query handling failed: {e}")
            return {
                "query_type": "general",
                "query": query,
                "error": f"Query processing failed: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the task router and its components"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "router_status": "healthy",
            "components": {}
        }

        # Check crew manager
        if self.crew_manager:
            try:
                crew_health = self.crew_manager.health_check()
                health_status["components"]["crew_manager"] = {
                    "status": "healthy",
                    "details": crew_health
                }
            except Exception as e:
                health_status["components"]["crew_manager"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health_status["components"]["crew_manager"] = {
                "status": "unavailable"
            }

        # Check voice service
        if self.voice_service:
            health_status["components"]["voice_service"] = {
                "status": "available"
            }
        else:
            health_status["components"]["voice_service"] = {
                "status": "unavailable"
            }

        # Overall health assessment
        component_statuses = [
            comp.get("status") for comp in health_status["components"].values()
        ]
        
        if "unhealthy" in component_statuses:
            health_status["router_status"] = "degraded"
        elif "unavailable" in component_statuses:
            health_status["router_status"] = "limited"

        return health_status

    def get_supported_queries(self) -> Dict[str, Any]:
        """Get information about supported query types"""
        return {
            "supported_query_types": {
                "morning_brief": {
                    "description": "Morning market briefings and portfolio analysis",
                    "examples": [
                        "What's our risk exposure in Asia tech stocks today?",
                        "Give me a morning market brief",
                        "Highlight any earnings surprises"
                    ]
                },
                "market_data": {
                    "description": "Real-time market data and stock prices",
                    "examples": [
                        "What's the current price of TSMC?",
                        "Show me market data for Samsung"
                    ]
                },
                "financial_analysis": {
                    "description": "Financial analysis and research",
                    "examples": [
                        "Analyze earnings for tech companies",
                        "What's the latest financial news?"
                    ]
                }
            },
            "input_types": ["text", "voice"],
            "timestamp": datetime.now().isoformat()
        }


# Factory function for easy instantiation
def create_task_router(config: Optional[Dict[str, Any]] = None) -> TaskRouter:
    """Factory function to create a properly configured task router"""
    return TaskRouter(config)


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Test the task router
    router = create_task_router()
    
    # Health check
    print("=== HEALTH CHECK ===")
    health = router.health_check()
    print(json.dumps(health, indent=2))
    
    # Test text queries
    print("\n=== TEXT QUERY TESTS ===")
    
    test_queries = [
        "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?",
        "Give me the current price of TSMC",
        "What are the latest earnings for Samsung?",
        "Hello, how are you?"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        result = router.route({"input_type": "text", "query": query})
        print(f"Result: {result.get('response', result.get('error', 'No response'))[:100]}...")
    
    # Test supported queries
    print("\n=== SUPPORTED QUERIES ===")
    supported = router.get_supported_queries()
    print(json.dumps(supported, indent=2))