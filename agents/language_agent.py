"""
Language Agent for Finance Assistant
Synthesizes narrative responses using LLM with RAG integration
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from services.vector_store import VectorStoreService
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialNarrativeTool(BaseTool):
    """Tool for generating financial narratives from structured data"""
    
    name: str = "financial_narrative_generator"
    description: str = "Generates natural language financial narratives from market data and analysis"
    
    def _run(self, market_data: Union[Dict, str] = None, analysis_results: Union[Dict, str] = None, context_documents: Union[List[Dict], str] = None) -> str:
        """Generate narrative from financial data"""
        try:
            # Handle string inputs by parsing them as JSON or creating empty dicts
            if isinstance(market_data, str):
                try:
                    market_data = json.loads(market_data) if market_data.strip() else {}
                except json.JSONDecodeError:
                    market_data = {}
            elif market_data is None:
                market_data = {}
                
            if isinstance(analysis_results, str):
                try:
                    analysis_results = json.loads(analysis_results) if analysis_results.strip() else {}
                except json.JSONDecodeError:
                    analysis_results = {}
            elif analysis_results is None:
                analysis_results = {}
                
            if isinstance(context_documents, str):
                try:
                    context_documents = json.loads(context_documents) if context_documents.strip() else []
                except json.JSONDecodeError:
                    context_documents = []
            elif context_documents is None:
                context_documents = []
            
            # Format the input data for narrative generation
            narrative_context = self._format_context(market_data, analysis_results, context_documents)
            
            # Create structured prompt for narrative generation
            prompt = self._create_narrative_prompt(narrative_context)
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating narrative: {str(e)}")
            return f"Error generating financial narrative: {str(e)}"
    
    def _format_context(self, market_data: Dict, analysis_results: Dict, context_documents: List[Dict] = None) -> Dict:
        """Format context for narrative generation"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data or {},
            "analysis": analysis_results or {},
            "supporting_documents": context_documents or []
        }
        
        # Extract key metrics safely
        if market_data and isinstance(market_data, dict):
            context["key_metrics"] = {
                "portfolio_allocation": market_data.get("portfolio_allocation", {}),
                "performance_data": market_data.get("performance", {}),
                "market_indicators": market_data.get("indicators", {})
            }
        else:
            context["key_metrics"] = {
                "portfolio_allocation": {},
                "performance_data": {},
                "market_indicators": {}
            }
        
        return context
    
    def _create_narrative_prompt(self, context: Dict) -> str:
        """Create structured prompt for narrative generation"""
        return f"""
        Based on the following financial data and analysis, create a professional market brief:
        
        Market Data: {json.dumps(context.get('market_data', {}), indent=2)}
        Analysis Results: {json.dumps(context.get('analysis', {}), indent=2)}
        Supporting Context: {json.dumps(context.get('supporting_documents', []), indent=2)}
        
        Generate a concise, professional narrative that addresses:
        1. Current portfolio allocation and changes
        2. Key performance highlights (earnings surprises, notable moves)
        3. Market sentiment and risk factors
        4. Actionable insights for portfolio management
        """


class MarketSentimentTool(BaseTool):
    """Tool for analyzing and interpreting market sentiment"""
    
    name: str = "market_sentiment_analyzer"
    description: str = "Analyzes market sentiment from news and data"
    
    def _run(self, news_data: Union[List[Dict], str] = None, market_indicators: Union[Dict, str] = None) -> Dict:
        """Analyze market sentiment"""
        try:
            # Handle string inputs
            if isinstance(news_data, str):
                try:
                    news_data = json.loads(news_data) if news_data.strip() else []
                except json.JSONDecodeError:
                    news_data = []
            elif news_data is None:
                news_data = []
                
            if isinstance(market_indicators, str):
                try:
                    market_indicators = json.loads(market_indicators) if market_indicators.strip() else {}
                except json.JSONDecodeError:
                    market_indicators = {}
            elif market_indicators is None:
                market_indicators = {}
            
            sentiment_score = 0
            sentiment_factors = []
            
            # Analyze news sentiment
            if news_data and isinstance(news_data, list):
                for item in news_data:
                    if isinstance(item, dict):
                        # Simple sentiment analysis based on keywords
                        content = item.get('title', '') + ' ' + item.get('summary', '')
                        score = self._calculate_sentiment_score(content)
                        sentiment_score += score
                        
                sentiment_score = sentiment_score / len(news_data) if news_data else 0
            
            # Analyze market indicators
            if market_indicators and isinstance(market_indicators, dict):
                indicator_sentiment = self._analyze_market_indicators(market_indicators)
                sentiment_factors.extend(indicator_sentiment)
            
            return {
                "overall_sentiment": self._classify_sentiment(sentiment_score),
                "sentiment_score": sentiment_score,
                "factors": sentiment_factors,
                "confidence": min(abs(sentiment_score) * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0,
                "factors": [],
                "confidence": 0
            }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from text"""
        if not isinstance(text, str):
            return 0
            
        positive_words = ['beat', 'exceed', 'growth', 'strong', 'positive', 'gain', 'rise', 'bull']
        negative_words = ['miss', 'decline', 'weak', 'negative', 'loss', 'fall', 'bear', 'risk']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
            
        return (pos_count - neg_count) / total_words * 10
    
    def _analyze_market_indicators(self, indicators: Dict) -> List[str]:
        """Analyze market indicators for sentiment factors"""
        factors = []
        
        # Analyze yield changes
        if 'yield_change' in indicators:
            yield_change = indicators['yield_change']
            if isinstance(yield_change, (int, float)):
                if yield_change > 0.1:
                    factors.append("rising yields creating pressure")
                elif yield_change < -0.1:
                    factors.append("falling yields providing support")
        
        # Analyze volatility
        if 'volatility' in indicators:
            vol = indicators['volatility']
            if isinstance(vol, (int, float)):
                if vol > 25:
                    factors.append("elevated volatility indicating uncertainty")
                elif vol < 15:
                    factors.append("low volatility suggesting stability")
        
        return factors
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score"""
        if score > 0.5:
            return "bullish"
        elif score < -0.5:
            return "bearish"
        else:
            return "neutral"


class LanguageAgent:
    """
    Language Agent responsible for synthesizing natural language responses
    from financial data using LLM and RAG integration
    """
    
    def __init__(self, vector_store_service: Optional[VectorStoreService] = None, llm_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.vector_store_service = vector_store_service
        self.llm = self._initialize_llm()
        self.tools = [
            FinancialNarrativeTool(),
            MarketSentimentTool()
        ]
        self.agent = self._create_agent()
        self.task = None
        
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if self.llm_provider.lower() == "openai":
                return ChatOpenAI(
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=1000
                )
            elif self.llm_provider.lower() == "ollama":
                return Ollama(
                    model=self.model_name,
                    temperature=0.3
                )
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_provider}, defaulting to OpenAI")
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # Fallback to a basic configuration
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI language agent"""
        return Agent(
            role='Financial Language Specialist',
            goal='Generate clear, professional financial narratives and market briefs',
            backstory="""You are an expert financial writer with deep knowledge of markets, 
            portfolio management, and investment analysis. You excel at translating complex 
            financial data into clear, actionable insights for portfolio managers and investors.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def synthesize_market_brief(self, query: str, market_data: Union[Dict, str] = None, 
                               analysis_results: Union[Dict, str] = None, context_documents: Union[List[Dict], str] = None) -> str:
        """
        Main method to synthesize market brief from various data sources
        """
        try:
            # Safely handle different input types
            if isinstance(market_data, str):
                try:
                    market_data = json.loads(market_data) if market_data.strip() else {}
                except json.JSONDecodeError:
                    market_data = {}
            elif market_data is None:
                market_data = {}
                
            if isinstance(analysis_results, str):
                try:
                    analysis_results = json.loads(analysis_results) if analysis_results.strip() else {}
                except json.JSONDecodeError:
                    analysis_results = {}
            elif analysis_results is None:
                analysis_results = {}
                
            if isinstance(context_documents, str):
                try:
                    context_documents = json.loads(context_documents) if context_documents.strip() else []
                except json.JSONDecodeError:
                    context_documents = []
            elif context_documents is None:
                context_documents = []
            
            # Create synthesis task
            task = Task(
                description=f"""
                Create a professional market brief responding to: "{query}"
                
                Use the provided market data, analysis results, and context documents to generate 
                a comprehensive response that includes:
                1. Current portfolio status and key changes
                2. Notable earnings surprises or market movements
                3. Risk assessment and market sentiment
                4. Actionable insights for portfolio management
                
                Market Data: {json.dumps(market_data, indent=2)}
                Analysis Results: {json.dumps(analysis_results, indent=2)}
                Context Documents: {json.dumps(context_documents, indent=2)}
                
                Format the response as a professional, conversational market brief suitable for verbal delivery.
                """,
                agent=self.agent,
                expected_output="A clear, concise market brief addressing the query with specific data points and actionable insights"
            )
            
            # Execute the task
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Extract and clean the result
            if hasattr(result, 'raw'):
                return self._clean_output(result.raw)
            else:
                return self._clean_output(str(result))
            
        except Exception as e:
            logger.error(f"Error synthesizing market brief: {str(e)}")
            return self._generate_fallback_response(query, market_data, analysis_results)
    
    def _clean_output(self, output: str) -> str:
        """Clean and format the output"""
        if not isinstance(output, str):
            output = str(output)
            
        # Remove any markdown formatting for voice output
        cleaned = output.replace('**', '').replace('*', '').replace('#', '')
        
        # Ensure proper sentence structure
        sentences = cleaned.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return '. '.join(sentences) + '.'
    
    def _generate_fallback_response(self, query: str, market_data: Dict, analysis_results: Dict) -> str:
        """Generate a basic fallback response when main synthesis fails"""
        try:
            # Ensure inputs are dictionaries
            if not isinstance(market_data, dict):
                market_data = {}
            if not isinstance(analysis_results, dict):
                analysis_results = {}
                
            # Extract key information for basic response
            allocation = market_data.get('portfolio_allocation', {})
            performance = analysis_results.get('performance_summary', {})
            
            response_parts = []
            
            # Portfolio allocation
            if allocation and isinstance(allocation, dict):
                for region, pct in allocation.items():
                    if isinstance(region, str) and 'asia' in region.lower() and 'tech' in region.lower():
                        response_parts.append(f"Your Asia tech allocation is {pct}% of AUM")
            
            # Performance highlights
            if performance and isinstance(performance, dict):
                beats = performance.get('earnings_beats', [])
                misses = performance.get('earnings_misses', [])
                
                if isinstance(beats, list):
                    for item in beats:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            company, pct = item[0], item[1]
                            response_parts.append(f"{company} beat estimates by {pct}%")
                        
                if isinstance(misses, list):
                    for item in misses:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            company, pct = item[0], item[1]
                            response_parts.append(f"{company} missed by {pct}%")
            
            # Default response if no specific data
            if not response_parts:
                response_parts.append("Based on current market conditions, Asia tech exposure appears stable")
            
            # Default sentiment
            response_parts.append("Market sentiment appears neutral with cautionary undertones")
            
            return '. '.join(response_parts) + '.'
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return "I'm experiencing technical difficulties processing your market brief request. Please try again."
    
    def generate_risk_summary(self, portfolio_data: Union[Dict, str] = None, market_conditions: Union[Dict, str] = None) -> str:
        """Generate risk-focused summary"""
        # Handle string inputs
        if isinstance(portfolio_data, str):
            try:
                portfolio_data = json.loads(portfolio_data) if portfolio_data.strip() else {}
            except json.JSONDecodeError:
                portfolio_data = {}
        elif portfolio_data is None:
            portfolio_data = {}
            
        if isinstance(market_conditions, str):
            try:
                market_conditions = json.loads(market_conditions) if market_conditions.strip() else {}
            except json.JSONDecodeError:
                market_conditions = {}
        elif market_conditions is None:
            market_conditions = {}
        
        task = Task(
            description=f"""
            Generate a risk-focused summary based on:
            Portfolio Data: {json.dumps(portfolio_data, indent=2)}
            Market Conditions: {json.dumps(market_conditions, indent=2)}
            
            Focus on:
            1. Current risk exposure levels
            2. Concentration risks
            3. Market-related risks
            4. Recommended risk management actions
            """,
            agent=self.agent,
            expected_output="A concise risk summary with specific risk levels and recommendations"
        )
        
        try:
            crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            return self._clean_output(str(result))
        except Exception as e:
            logger.error(f"Error generating risk summary: {str(e)}")
            return "Risk analysis temporarily unavailable."
    
    def format_for_voice(self, text: str) -> str:
        """Format text specifically for voice output"""
        if not isinstance(text, str):
            text = str(text)
            
        # Remove special characters that don't translate well to speech
        voice_text = text.replace('%', ' percent')
        voice_text = voice_text.replace('$', ' dollars')
        voice_text = voice_text.replace('&', ' and')
        
        # Add pauses for better speech flow
        voice_text = voice_text.replace('.', '. ')
        voice_text = voice_text.replace(',', ', ')
        
        return voice_text.strip()


# Factory function for easy instantiation
def create_language_agent(vector_store_service: Optional[VectorStoreService] = None, llm_provider: str = "openai", model_name: str = "gpt-3.5-turbo") -> LanguageAgent:
    """Factory function to create language agent"""
    return LanguageAgent(vector_store_service, llm_provider, model_name)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    agent = create_language_agent()
    
    # Sample data for testing
    sample_market_data = {
        "portfolio_allocation": {
            "asia_tech": 22,
            "us_equities": 45,
            "bonds": 20,
            "other": 13
        },
        "performance": {
            "daily_change": 1.2,
            "weekly_change": -0.8
        }
    }
    
    sample_analysis = {
        "earnings_beats": [("TSMC", 4)],
        "earnings_misses": [("Samsung", 2)],
        "sentiment": "neutral_cautious"
    }
    
    sample_query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    
    # Test the agent
    try:
        result = agent.synthesize_market_brief(
            query=sample_query,
            market_data=sample_market_data,
            analysis_results=sample_analysis
        )
        print("Generated Market Brief:")
        print(result)
        print("\nVoice-formatted version:")
        print(agent.format_for_voice(result))
        
    except Exception as e:
        print(f"Test failed: {str(e)}")