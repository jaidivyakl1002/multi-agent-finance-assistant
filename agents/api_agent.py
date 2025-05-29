import os
import requests
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any, ClassVar
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from pydantic import BaseModel
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataTool(BaseTool):
    """Custom tool for fetching market data from multiple sources"""

    name: str = "market_data_fetcher"
    description: str = "Fetches real-time and historical market data from AlphaVantage and Yahoo Finance"
    alpha_vantage_key: Optional[str] = None

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        super().__init__()
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
    def _run(self, symbols: str, data_type: str = "quote", period: str = "1d") -> Dict[str, Any]:
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            results = {}
            
            if data_type == "quote":
                results = self._fetch_quotes(symbol_list)
            elif data_type == "historical":
                results = self._fetch_historical(symbol_list, period)
            elif data_type == "news":
                results = self._fetch_news(symbol_list)
            
            output = {
                "data_type": data_type,
                "symbols": symbol_list,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Market data fetched: {output}")
            return output
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _fetch_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        import yfinance as yf
        quotes = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                earnings = ticker.earnings_dates.head(1) if hasattr(ticker, 'earnings_dates') else None
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    
                    quote = {
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_pct, 2),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'market_cap': info.get('marketCap', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A')
                    }
                    
                    if earnings is not None and not earnings.empty:
                        actual_eps = earnings.get('Reported EPS', [None])[0]
                        expected_eps = earnings.get('EPS Estimate', [None])[0]
                        if actual_eps and expected_eps:
                            quote['earnings'] = {
                                'actual_eps': float(actual_eps),
                                'expected_eps': float(expected_eps),
                                'surprise': float(actual_eps - expected_eps),
                                'surprise_percent': ((actual_eps - expected_eps) / expected_eps * 100) if expected_eps != 0 else 0
                            }
                    
                    quotes[symbol] = quote
            except Exception as e:
                logger.warning(f"Failed to fetch quote for {symbol}: {str(e)}")
                quotes[symbol] = {'error': str(e)}
        return quotes
    
    def _fetch_historical(self, symbols: List[str], period: str) -> Dict[str, Any]:
        """Fetch historical data using Yahoo Finance"""
        historical = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    historical[symbol] = {
                        'data': hist.to_dict('records'),
                        'summary': {
                            'high': round(hist['High'].max(), 2),
                            'low': round(hist['Low'].min(), 2),
                            'avg_volume': int(hist['Volume'].mean()),
                            'volatility': round(hist['Close'].pct_change().std() * 100, 2)
                        }
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to fetch historical data for {symbol}: {str(e)}")
                historical[symbol] = {'error': str(e)}
                
        return historical
    
    def _fetch_news(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch news using Yahoo Finance"""
        news_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                if news:
                    news_data[symbol] = []
                    for item in news[:5]:  # Limit to 5 recent news items
                        news_data[symbol].append({
                            'title': item.get('title', ''),
                            'publisher': item.get('publisher', ''),
                            'publish_time': datetime.fromtimestamp(
                                item.get('providerPublishTime', 0)
                            ).strftime('%Y-%m-%d %H:%M:%S'),
                            'link': item.get('link', '')
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to fetch news for {symbol}: {str(e)}")
                news_data[symbol] = {'error': str(e)}
                
        return news_data
    
    def _format_results(self, results: Dict[str, Any], data_type: str) -> str:
        """Format results for agent consumption"""
        if not results:
            return "No data retrieved"
            
        formatted = f"Market Data ({data_type.upper()}):\n\n"
        
        for symbol, data in results.items():
            if 'error' in data:
                formatted += f"{symbol}: Error - {data['error']}\n"
                continue
                
            formatted += f"{symbol}:\n"
            
            if data_type == "quote":
                formatted += f"  Price: ${data['price']} ({data['change']:+.2f}, {data['change_percent']:+.2f}%)\n"
                formatted += f"  Volume: {data['volume']:,}\n"
                formatted += f"  Sector: {data['sector']}\n"
                formatted += f"  Market Cap: {data['market_cap']}\n"
                
            elif data_type == "historical":
                summary = data['summary']
                formatted += f"  High: ${summary['high']}, Low: ${summary['low']}\n"
                formatted += f"  Avg Volume: {summary['avg_volume']:,}\n"
                formatted += f"  Volatility: {summary['volatility']:.2f}%\n"
                
            elif data_type == "news":
                for news_item in data[:3]:  # Show top 3 news items
                    formatted += f"  • {news_item['title']} ({news_item['publisher']})\n"
                    
            formatted += "\n"
            
        return formatted

class AsiaFocusedMarketTool(BaseTool):
    """Specialized tool for Asia-Pacific market data"""
    
    name: str = "asia_market_data"
    description: str = "Fetches Asia-Pacific market data with focus on tech stocks"
    
    ASIA_TECH_SYMBOLS: ClassVar[List[str]] = [
        'TSM', '005930.KS', 'ASML', 'NVDA', '2330.TW', '6758.T', '000660.KS'
    ]
    
    def _run(self, query: str = "") -> Dict[str, Any]:
        try:
            market_tool = MarketDataTool()
            symbols_str = ','.join(self.ASIA_TECH_SYMBOLS)
            quotes_result = market_tool._run(symbols_str, "quote")
            
            sentiment = self._analyze_regional_sentiment()
            
            return {
                "overview": "Asia-Pacific Tech Market Overview",
                "quotes": quotes_result,
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching Asia market data: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _analyze_regional_sentiment(self) -> str:
        """Simple sentiment analysis based on price movements"""
        try:
            positive_moves = 0
            total_moves = 0
            
            for symbol in self.ASIA_TECH_SYMBOLS[:4]:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if len(hist) >= 2:
                        change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                        if change > 0:
                            positive_moves += 1
                        total_moves += 1
                except:
                    continue
            
            if total_moves > 0:
                sentiment_score = positive_moves / total_moves
                if sentiment_score >= 0.7:
                    sentiment = "POSITIVE"
                elif sentiment_score >= 0.4:
                    sentiment = "NEUTRAL"
                else:
                    sentiment = "NEGATIVE"
                    
                return f"\nREGIONAL SENTIMENT: {sentiment} ({positive_moves}/{total_moves} stocks positive)\n"
            
        except Exception as e:
            logger.warning(f"Could not analyze sentiment: {str(e)}")
            
        return "\nREGIONAL SENTIMENT: NEUTRAL (insufficient data)\n"

# ============================================================================
# AGENT CREATION AND TASK DEFINITIONS
# ============================================================================

def create_market_data_agent() -> Agent:
    """Create the Market Data Agent with specialized tools"""
    
    market_tool = MarketDataTool()
    asia_tool = AsiaFocusedMarketTool()
    
    agent = Agent(
        role='Senior Market Data Analyst',
        goal='Provide comprehensive, accurate, and timely market data analysis with expertise in Asia-Pacific tech sector',
        backstory=(
            "You are a seasoned market data analyst with 15+ years of experience "
            "covering global financial markets, with specialized expertise in Asia-Pacific "
            "technology stocks. You have deep knowledge of semiconductor industry dynamics, "
            "regional market correlations, and can quickly identify significant market "
            "movements, earnings surprises, and risk factors that impact portfolio allocations."
        ),
        tools=[market_tool, asia_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        memory=True
    )
    
    return agent

def create_market_analysis_task(agent: Agent, query: str, context: Optional[Dict] = None) -> Task:
    """Create a comprehensive market analysis task"""
    
    context = context or {}
    symbols = context.get('symbols', 'TSM,005930.KS,ASML')
    data_type = context.get('data_type', 'quote')
    period = context.get('period', '1d')
    
    description = f"""
    Conduct comprehensive market analysis for the following request: {query}
    
    Your analysis should include:
    
    1. REAL-TIME MARKET DATA:
       - Fetch current quotes for specified symbols: {symbols}
       - Include price changes, volume, and market cap information
       - Focus on Asia-Pacific tech stocks if mentioned in query
    
    2. CONTEXT ANALYSIS:
       - Identify any earnings surprises (beats/misses with specific percentages)
       - Analyze regional market sentiment and trends
       - Highlight significant price movements or volume anomalies
    
    3. RISK ASSESSMENT:
       - Evaluate current risk exposure for mentioned sectors/regions
       - Identify key risk factors affecting the portfolio
       - Assess market volatility and correlation patterns
    
    4. PORTFOLIO RELEVANCE:
       - Calculate or estimate portfolio allocation percentages
       - Identify changes from previous trading sessions
       - Highlight impact on overall risk profile
    
    Query Context: {json.dumps(context, indent=2)}
    
    Provide specific numbers, percentages, and concrete data points.
    """
    
    task = Task(
        description=description,
        agent=agent,
        expected_output=(
            "Comprehensive market analysis report with specific data points including: "
            "current prices and changes, portfolio allocation percentages, earnings surprises "
            "with exact percentages, regional sentiment assessment, and key risk factors. "
            "All data should be current and include specific numbers."
        )
    )
    
    return task

def create_asia_tech_overview_task(agent: Agent, query: str) -> Task:
    """Create specialized task for Asia tech market overview"""
    
    task = Task(
        description=f"""
        Provide a comprehensive Asia-Pacific technology sector overview for: {query}
        
        Focus areas:
        1. Major Asia tech stock performance (TSMC, Samsung, SK Hynix, etc.)
        2. Semiconductor industry trends and earnings results
        3. Regional market sentiment and cross-correlations
        4. Currency impacts on tech stock valuations
        5. Supply chain and geopolitical factors affecting the sector
        
        Use the Asia-focused market tool for specialized regional analysis.
        Include specific allocation percentages and risk metrics.
        """,
        agent=agent,
        expected_output=(
            "Detailed Asia tech sector overview with current performance metrics, "
            "earnings analysis, sentiment assessment, and portfolio allocation recommendations"
        )
    )
    
    return task

def create_earnings_analysis_task(agent: Agent, query: str, companies: List[str]) -> Task:
    """Create specialized task for earnings analysis"""
    
    companies_str = ', '.join(companies)
    
    task = Task(
        description=f"""
        Conduct detailed earnings analysis for: {query}
        
        Companies to analyze: {companies_str}
        
        For each company, determine:
        1. Latest earnings results vs. estimates (exact percentage beats/misses)
        2. Revenue performance and guidance updates
        3. Key business segment performance
        4. Management commentary on future outlook
        5. Market reaction and stock price impact
        
        Highlight any significant earnings surprises with specific percentages.
        Assess the overall impact on sector sentiment and portfolio allocations.
        """,
        agent=agent,
        expected_output=(
            "Detailed earnings analysis showing specific beat/miss percentages, "
            "revenue performance, and market impact assessment for each company"
        )
    )
    
    return task

# ============================================================================
# CREW ORCHESTRATION PATTERNS
# ============================================================================

class MarketDataCrew:
    """Orchestrates market data analysis using CrewAI framework"""
    
    def __init__(self):
        self.agent = create_market_data_agent()
        logger.info("Market Data Crew initialized")
    
    def analyze_market_query(self, query: str, context: Optional[Dict] = None) -> str:
        """Analyze a general market query"""
        task = create_market_analysis_task(self.agent, query, context)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
            memory=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def get_asia_tech_overview(self, query: str = "Asia tech market overview") -> str:
        """Get Asia tech sector overview"""
        task = create_asia_tech_overview_task(self.agent, query)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
            memory=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def analyze_earnings(self, query: str, companies: List[str]) -> str:
        """Analyze earnings for specific companies"""
        task = create_earnings_analysis_task(self.agent, query, companies)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
            memory=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def process_morning_brief_query(self, query: str) -> Dict[str, Any]:
        """Process the specific morning brief use case"""
        
        # Extract context from query for targeted analysis
        context = self._extract_query_context(query)
        
        # Create targeted task based on query analysis
        if "asia tech" in query.lower() and "risk exposure" in query.lower():
            task = create_asia_tech_overview_task(self.agent, query)
        elif "earnings" in query.lower():
            companies = self._extract_companies_from_query(query)
            task = create_earnings_analysis_task(self.agent, query, companies)
        else:
            task = create_market_analysis_task(self.agent, query, context)
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
            memory=True
        )
        
        result = crew.kickoff()
        
        return {
            "query": query,
            "context": context,
            "analysis": str(result),
            "timestamp": datetime.now().isoformat(),
            "agent_type": "market_data_specialist"
        }
    
    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract context from query for targeted analysis"""
        query_lower = query.lower()
        
        context = {
            "symbols": "TSM,005930.KS,ASML,NVDA",  # Default Asia tech symbols
            "data_type": "quote",
            "period": "1d",
            "focus_areas": []
        }
        
        # Detect focus areas
        if "risk exposure" in query_lower:
            context["focus_areas"].append("risk_analysis")
        if "earnings" in query_lower:
            context["focus_areas"].append("earnings_analysis")
        if "asia" in query_lower or "asian" in query_lower:
            context["focus_areas"].append("asia_focus")
        if "tech" in query_lower or "technology" in query_lower:
            context["focus_areas"].append("tech_sector")
        
        return context
    
    def _extract_companies_from_query(self, query: str) -> List[str]:
        """Extract company names from query"""
        companies = []
        query_upper = query.upper()
        
        # Common company mappings
        company_mapping = {
            "TSMC": "TSM",
            "SAMSUNG": "005930.KS", 
            "TAIWAN SEMICONDUCTOR": "TSM",
            "SK HYNIX": "000660.KS",
            "NVIDIA": "NVDA",
            "ASML": "ASML"
        }
        
        for company_name, symbol in company_mapping.items():
            if company_name in query_upper:
                companies.append(symbol)
        
        return companies if companies else ["TSM", "005930.KS"]  # Default

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_api_agent() -> Agent:
    """Factory function - maintains backward compatibility"""
    return create_market_data_agent()

def get_market_data_crew() -> MarketDataCrew:
    """Factory function to get market data crew instance"""
    return MarketDataCrew()

# Example usage for testing
if __name__ == "__main__":
    # Test the enhanced structure
    crew = MarketDataCrew()
    
    # Test morning brief query
    test_query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    result = crew.process_morning_brief_query(test_query)
    
    print("Morning Brief Result:")
    print(json.dumps(result, indent=2))
    
    # Test Asia tech overview
    asia_result = crew.get_asia_tech_overview()
    print("\nAsia Tech Overview:")
    print(asia_result)