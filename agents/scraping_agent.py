import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path
from services.vector_store import VectorStoreService
from crewai import Agent, Task
from crewai_tools import BaseTool
from pydantic import PrivateAttr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Structured result from scraping operations"""
    source: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class SECFilingsTool(BaseTool):
    name: str = "sec_filings_scraper"
    description: str = "Scrapes SEC filings, 10-K, 10-Q, and earnings reports for specified companies"

    _base_url: str = PrivateAttr(default="https://www.sec.gov")
    _headers: Dict[str, str] = PrivateAttr(default_factory=lambda: {
        'User-Agent': 'Finance Assistant Bot (educational@example.com)',
        'Accept': 'application/json, text/html',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    })
    _session: requests.Session = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session = requests.Session()
        self._session.headers.update(self._headers)

        
    def _run(self, ticker: str, filing_types: List[str] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Scrape SEC filings for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to search for ['10-K', '10-Q', '8-K']
            limit: Maximum number of filings to retrieve
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']
            
        try:
            # Get company CIK
            cik = self._get_company_cik(ticker)
            if not cik:
                return {"error": f"Could not find CIK for ticker {ticker}"}
            
            # Search for filings
            filings = self._search_filings(cik, filing_types, limit)
            
            # Extract content from filings
            results = []
            for filing in filings[:limit]:
                content = self._extract_filing_content(filing)
                if content:
                    results.append({
                        "ticker": ticker,
                        "filing_type": filing.get("form"),
                        "filing_date": filing.get("filingDate"),
                        "content": content,
                        "url": filing.get("url"),
                        "metadata": {
                            "accessionNumber": filing.get("accessionNumber"),
                            "reportDate": filing.get("reportDate")
                        }
                    })
            
            return {
                "success": True,
                "ticker": ticker,
                "filings_count": len(results),
                "filings": results
            }
            
        except Exception as e:
            logger.error(f"Error scraping SEC filings for {ticker}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    def _get_company_cik(self, ticker: str) -> Optional[str]:
        """Get company CIK from ticker symbol"""
        try:
            url = f"{self._base_url}/files/company_tickers.json"
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            
            companies = response.json()
            for company in companies.values():
                if company.get('ticker', '').upper() == ticker.upper():
                    return str(company.get('cik_str', '')).zfill(10)
            
            return None
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {str(e)}")
            return None
    
    def _search_filings(self, cik: str, filing_types: List[str], limit: int) -> List[Dict]:
        """Search for filings using SEC EDGAR API"""
        try:
            url = f"{self._base_url}/Archives/edgar/data/{cik.lstrip('0')}/company_concept.json"
            response = self._session.get(url, timeout=15)
            
            # Fallback to submissions endpoint
            submissions_url = f"{self._base_url}/Archives/edgar/data/{cik}/submissions.json"
            response = self._session.get(submissions_url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})
            
            filings = []
            forms = recent_filings.get('form', [])
            dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            
            for i, form in enumerate(forms):
                if form in filing_types and len(filings) < limit * 2:  # Get extra to filter
                    filings.append({
                        'form': form,
                        'filingDate': dates[i] if i < len(dates) else None,
                        'accessionNumber': accession_numbers[i] if i < len(accession_numbers) else None,
                        'url': f"{self._base_url}/Archives/edgar/data/{cik.lstrip('0')}/{accession_numbers[i].replace('-', '')}/{accession_numbers[i]}-index.htm"
                    })
            
            return sorted(filings, key=lambda x: x.get('filingDate', ''), reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error searching filings for CIK {cik}: {str(e)}")
            return []
    
    def _extract_filing_content(self, filing: Dict) -> Optional[str]:
        """Extract text content from SEC filing"""
        try:
            response = self._session.get(filing['url'], timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content size (first 5000 characters for summary)
            return text[:5000] if len(text) > 5000 else text
            
        except Exception as e:
            logger.error(f"Error extracting filing content: {str(e)}")
            return None

class FinancialNewsTool(BaseTool):
    name: str = "financial_news_scraper"
    description: str = "Scrapes financial news, earnings reports, and market sentiment from various sources"

    _news_sources: Dict[str, str] = PrivateAttr(default_factory=lambda: {
        'yahoo_finance': 'https://finance.yahoo.com',
        'marketwatch': 'https://www.marketwatch.com',
        'reuters': 'https://www.reuters.com/business/finance'
    })
    _headers: Dict[str, str] = PrivateAttr(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    _session: requests.Session = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session = requests.Session()
        self._session.headers.update(self._headers)
    
    def _run(self, ticker: str, news_type: str = "earnings", limit: int = 10) -> Dict[str, Any]:
        """
        Scrape financial news for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            news_type: Type of news to scrape ('earnings', 'general', 'analyst')
            limit: Maximum number of articles to retrieve
        """
        try:
            results = []
            
            if news_type == "earnings":
                yahoo_earnings = self._scrape_yahoo_earnings(ticker, limit//2)
                results.extend(yahoo_earnings)
                
            elif news_type == "general":
                yahoo_news = self._scrape_yahoo_news(ticker, limit//2)
                results.extend(yahoo_news)
                
            elif news_type == "analyst":
                analyst_news = self._scrape_analyst_ratings(ticker, limit//2)
                results.extend(analyst_news)
            
            return {
                "success": True,
                "ticker": ticker,
                "news_type": news_type,
                "articles_count": len(results),
                "articles": results[:limit]
            }
            
        except Exception as e:
            logger.error(f"Error scraping news for {ticker}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    def _scrape_yahoo_earnings(self, ticker: str, limit: int) -> List[Dict]:
        """Scrape earnings information from Yahoo Finance"""
        try:
            url = f"{self._news_sources['yahoo_finance']}/quote/{ticker}/analysis"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract earnings estimates and analyst data
            earnings_data = []
            
            # Look for earnings table
            tables = soup.find_all('table')
            for table in tables:
                if 'earnings' in str(table).lower():
                    rows = table.find_all('tr')
                    for row in rows[1:limit]:  # Skip header
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            earnings_data.append({
                                "source": "Yahoo Finance - Earnings",
                                "ticker": ticker,
                                "content": ' '.join([cell.get_text().strip() for cell in cells]),
                                "url": url,
                                "scraped_at": datetime.now().isoformat()
                            })
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo earnings for {ticker}: {str(e)}")
            return []
    
    def _scrape_yahoo_news(self, ticker: str, limit: int) -> List[Dict]:
        """Scrape general news from Yahoo Finance"""
        try:
            url = f"{self._news_sources['yahoo_finance']}/quote/{ticker}/news"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_articles = []
            
            # Look for news articles
            articles = soup.find_all(['h3', 'h4'], class_=lambda x: x and 'headline' in x.lower())
            
            for article in articles[:limit]:
                title = article.get_text().strip()
                link_elem = article.find('a')
                link = link_elem.get('href') if link_elem else None
                
                if link and not link.startswith('http'):
                    link = urljoin(self.news_sources['yahoo_finance'], link)
                
                news_articles.append({
                    "source": "Yahoo Finance - News",
                    "ticker": ticker,
                    "title": title,
                    "content": title,  # Could expand to get full article content
                    "url": link,
                    "scraped_at": datetime.now().isoformat()
                })
            
            return news_articles
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo news for {ticker}: {str(e)}")
            return []
    
    def _scrape_analyst_ratings(self, ticker: str, limit: int) -> List[Dict]:
        """Scrape analyst ratings and recommendations"""
        try:
            url = f"{self._news_sources['yahoo_finance']}/quote/{ticker}/analysis"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            analyst_data = []
            
            # Look for analyst recommendations
            sections = soup.find_all('section')
            for section in sections:
                if 'recommendation' in str(section).lower():
                    text_content = section.get_text().strip()
                    if text_content:
                        analyst_data.append({
                            "source": "Yahoo Finance - Analyst Ratings",
                            "ticker": ticker,
                            "content": text_content,
                            "url": url,
                            "scraped_at": datetime.now().isoformat()
                        })
            
            return analyst_data[:limit]
            
        except Exception as e:
            logger.error(f"Error scraping analyst ratings for {ticker}: {str(e)}")
            return []

class ScrapingAgent:
    """CrewAI Scraping Agent for Financial Data"""
    
    def __init__(self, vector_store_service=None):
        self.sec_tool = SECFilingsTool()
        self.news_tool = FinancialNewsTool()
        self.vector_store_service = vector_store_service
        
        # Initialize CrewAI agent
        self.agent = Agent(
            role='Financial Document Scraper',
            goal='Extract and analyze financial documents, SEC filings, and market news',
            backstory="""You are an expert financial document analyst with deep knowledge 
                        of SEC filings, earnings reports, and financial news sources. 
                        You excel at extracting relevant information from various financial 
                        documents and presenting it in a structured, actionable format.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.sec_tool, self.news_tool]
        )
    
    def create_scraping_task(self, ticker: str, scraping_type: str = "comprehensive") -> Task:
        """Create a scraping task for the agent"""
        
        if scraping_type == "sec_filings":
            description = f"""
            Scrape and analyze SEC filings for {ticker}. Focus on:
            1. Recent 10-K and 10-Q filings
            2. Any recent 8-K filings with material information
            3. Extract key financial metrics and risk factors
            4. Summarize any significant changes or updates
            """
        elif scraping_type == "earnings":
            description = f"""
            Gather earnings-related information for {ticker}. Focus on:
            1. Recent earnings reports and estimates
            2. Analyst ratings and recommendations
            3. Earnings surprises (beats/misses)
            4. Forward-looking guidance if available
            """
        else:  # comprehensive
            description = f"""
            Perform comprehensive financial document scraping for {ticker}. Include:
            1. Recent SEC filings (10-K, 10-Q, 8-K)
            2. Latest earnings reports and analyst estimates
            3. Recent financial news and market sentiment
            4. Any material changes or significant events
            
            Provide a structured summary with key insights and risk factors.
            """
        
        return Task(
            description=description,
            agent=self.agent,
            expected_output=f"Structured financial document analysis for {ticker} with key insights and data points"
        )
    
    async def scrape_multiple_tickers(self, tickers: List[str], scraping_type: str = "earnings") -> Dict[str, Any]:
        """Asynchronously scrape multiple tickers"""
        
        async def scrape_single_ticker(ticker: str) -> Dict[str, Any]:
            try:
                if scraping_type == "sec_filings":
                    result = self.sec_tool._run(ticker)
                elif scraping_type in ["earnings", "general", "analyst"]:
                    result = self.news_tool._run(ticker, news_type=scraping_type)
                else:
                    # Comprehensive - combine both
                    sec_result = self.sec_tool._run(ticker, limit=3)
                    news_result = self.news_tool._run(ticker, news_type="earnings", limit=5)
                    
                    result = {
                        "ticker": ticker,
                        "success": sec_result.get("success", False) or news_result.get("success", False),
                        "sec_filings": sec_result,
                        "news_data": news_result,
                        "combined_insights": self._combine_insights(sec_result, news_result)
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Error scraping {ticker}: {str(e)}")
                return {
                    "ticker": ticker,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute scraping tasks concurrently
        tasks = [scrape_single_ticker(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "scraping_type": scraping_type,
            "tickers_processed": len(tickers),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _combine_insights(self, sec_data: Dict, news_data: Dict) -> Dict[str, Any]:
        """Combine insights from SEC filings and news data"""
        insights = {
            "data_sources": [],
            "key_findings": [],
            "risk_factors": [],
            "recent_developments": []
        }
        
        # Process SEC filing insights
        if sec_data.get("success") and sec_data.get("filings"):
            insights["data_sources"].append("SEC Filings")
            for filing in sec_data["filings"]:
                insights["key_findings"].append({
                    "source": f"SEC {filing['filing_type']}",
                    "date": filing["filing_date"],
                    "summary": filing["content"][:200] + "..." if len(filing["content"]) > 200 else filing["content"]
                })
        
        # Process news insights
        if news_data.get("success") and news_data.get("articles"):
            insights["data_sources"].append("Financial News")
            for article in news_data["articles"]:
                insights["recent_developments"].append({
                    "source": article["source"],
                    "title": article.get("title", ""),
                    "content": article["content"][:150] + "..." if len(article["content"]) > 150 else article["content"]
                })
        
        return insights
    
    def get_agent(self) -> Agent:
        """Return the CrewAI agent instance"""
        return self.agent

# Utility functions for integration
def create_scraping_crew_member() -> ScrapingAgent:
    """Factory function to create a scraping agent for CrewAI crews"""
    return ScrapingAgent()

def test_scraping_tools():
    """Test function for scraping tools"""
    scraper = ScrapingAgent()
    
    # Test SEC filings
    print("Testing SEC filings scraper...")
    sec_result = scraper.sec_tool._run("AAPL", limit=2)
    print(f"SEC Result: {json.dumps(sec_result, indent=2, default=str)}")
    
    # Test news scraper
    print("\nTesting news scraper...")
    news_result = scraper.news_tool._run("AAPL", news_type="earnings", limit=3)
    print(f"News Result: {json.dumps(news_result, indent=2, default=str)}")

if __name__ == "__main__":
    # Test the scraping tools
    test_scraping_tools()