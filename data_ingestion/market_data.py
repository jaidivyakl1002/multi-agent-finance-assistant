import os
import logging
import requests
import yfinance as yf
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Structure for market data points"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None

class NetworkUtils:
    """Utility class for network diagnostics and health checks"""
    
    @staticmethod
    def check_internet_connection() -> bool:
        """Check if internet connection is available"""
        try:
            # Try to connect to Google's DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    @staticmethod
    def check_host_resolution(host: str) -> bool:
        """Check if a specific host can be resolved"""
        try:
            socket.gethostbyname(host)
            return True
        except socket.gaierror:
            return False
    
    @staticmethod
    def test_api_endpoints() -> Dict[str, bool]:
        """Test connectivity to various financial data API endpoints"""
        endpoints = {
            'yahoo_finance': 'query1.finance.yahoo.com',
            'alpha_vantage': 'www.alphavantage.co',
            'google_dns': '8.8.8.8'
        }
        
        results = {}
        for name, host in endpoints.items():
            if name == 'google_dns':
                results[name] = NetworkUtils.check_internet_connection()
            else:
                results[name] = NetworkUtils.check_host_resolution(host)
        
        return results

class MarketDataProvider:
    def __init__(self, alpha_vantage_key: Optional[str] = None, timeout: int = 10):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.timeout = timeout
        self.network_utils = NetworkUtils()
        
    def check_network_health(self) -> Dict[str, Any]:
        """Comprehensive network health check"""
        health_report = {
            'timestamp': datetime.now(),
            'internet_available': self.network_utils.check_internet_connection(),
            'endpoints': self.network_utils.test_api_endpoints(),
            'recommendations': []
        }
        
        if not health_report['internet_available']:
            health_report['recommendations'].append("No internet connection detected")
        
        if not health_report['endpoints'].get('yahoo_finance', False):
            health_report['recommendations'].append("Yahoo Finance API unreachable - consider using Alpha Vantage")
        
        if not health_report['endpoints'].get('alpha_vantage', False):
            health_report['recommendations'].append("Alpha Vantage API unreachable")
            
        return health_report

    def get_quote_yfinance(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch real-time quote data using yfinance with enhanced error handling."""
        # Pre-flight network check
        if not self.network_utils.check_host_resolution('query1.finance.yahoo.com'):
            logger.error("Yahoo Finance host unreachable. Check network connectivity.")
            return {symbol: {"error": "Network connectivity issue - Yahoo Finance unreachable"} for symbol in symbols}
        
        data = {}
        for symbol in symbols:
            try:
                # Set timeout and session for better reliability
                ticker = yf.Ticker(symbol)
                ticker.session.timeout = self.timeout
                
                # Try to get basic info first
                info = ticker.info
                if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                    # Fall back to history if info fails
                    hist = ticker.history(period="2d")
                    if len(hist) < 2:
                        raise ValueError(f"Insufficient historical data for {symbol}")
                    
                    price = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    change = price - prev
                    volume = hist["Volume"].iloc[-1]
                else:
                    # Use info data when available
                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    change = info.get('regularMarketChange', 0)
                    volume = info.get('regularMarketVolume', 0)
                    
                    # Calculate previous price for percentage
                    prev = price - change if change != 0 else price

                data[symbol] = {
                    "price": round(float(price), 2),
                    "change": round(float(change), 2),
                    "change_percent": round((change / prev) * 100, 2) if prev != 0 else 0,
                    "volume": int(volume) if volume else 0,
                    "market_cap": info.get("marketCap", "N/A") if 'info' in locals() else "N/A",
                    "sector": info.get("sector", "N/A") if 'info' in locals() else "N/A",
                    "industry": info.get("industry", "N/A") if 'info' in locals() else "N/A",
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"[yfinance] Error for {symbol}: {e}")
                data[symbol] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
        return data

    def get_historical_yfinance(self, symbols: List[str], period: str = "1mo") -> Dict[str, Any]:
        """Fetch historical price data for symbols with better error handling."""
        if not self.network_utils.check_host_resolution('query1.finance.yahoo.com'):
            logger.error("Yahoo Finance host unreachable for historical data.")
            return {symbol: {"error": "Network connectivity issue"} for symbol in symbols}
            
        result = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                ticker.session.timeout = self.timeout
                hist = ticker.history(period=period)
                
                if hist.empty:
                    result[symbol] = {"error": f"No historical data available for {symbol}"}
                else:
                    result[symbol] = hist.reset_index().to_dict("records")
            except Exception as e:
                logger.warning(f"[yfinance] Historical error for {symbol}: {e}")
                result[symbol] = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        return result

    def get_alpha_vantage_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from Alpha Vantage with better error handling."""
        if not self.alpha_vantage_key:
            return {"error": "Alpha Vantage API key not configured"}
            
        if not self.network_utils.check_host_resolution('www.alphavantage.co'):
            return {"error": "Alpha Vantage host unreachable"}
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            json_data = response.json()
            
            # Check for API error messages
            if "Error Message" in json_data:
                return {"error": json_data["Error Message"]}
            
            if "Note" in json_data:
                return {"error": f"API limit reached: {json_data['Note']}"}
                
            global_quote = json_data.get("Global Quote", {})
            if not global_quote:
                return {"error": "No data returned from Alpha Vantage"}
                
            return {
                "symbol": global_quote.get("01. symbol", symbol),
                "price": float(global_quote.get("05. price", 0)),
                "volume": int(global_quote.get("06. volume", 0)),
                "change": float(global_quote.get("09. change", 0)),
                "change_percent": global_quote.get("10. change percent", "0%"),
                "last_updated": datetime.now().isoformat()
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"[Alpha Vantage] Network error for {symbol}: {e}")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"[Alpha Vantage] Unexpected error for {symbol}: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    def fetch_market_data(
        self, symbols: List[str], data_type: str = "quote", period: str = "1mo"
    ) -> Dict[str, Any]:
        """Unified entry point to fetch market data with health checks."""
        # Perform network health check
        health = self.check_network_health()
        
        result = {
            "network_health": health,
            "data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to fetch data based on available endpoints
        if data_type == "quote":
            if health['endpoints'].get('yahoo_finance', False):
                result["data"] = self.get_quote_yfinance(symbols)
                result["primary_source"] = "yahoo_finance"
            elif health['endpoints'].get('alpha_vantage', False) and self.alpha_vantage_key:
                # Fall back to Alpha Vantage for individual quotes
                result["data"] = {}
                for symbol in symbols:
                    result["data"][symbol] = self.get_alpha_vantage_quote(symbol)
                    time.sleep(0.2)  # Rate limiting
                result["primary_source"] = "alpha_vantage"
            else:
                result["data"] = {symbol: {"error": "No available data sources"} for symbol in symbols}
                result["primary_source"] = "none"
                
        elif data_type == "historical":
            if health['endpoints'].get('yahoo_finance', False):
                result["data"] = self.get_historical_yfinance(symbols, period)
                result["primary_source"] = "yahoo_finance"
            else:
                result["data"] = {symbol: {"error": "Historical data requires Yahoo Finance"} for symbol in symbols}
                result["primary_source"] = "none"
        else:
            result["data"] = {"error": f"Unsupported data_type: {data_type}"}
            
        return result


class AlphaVantageConnector:
    """Enhanced Alpha Vantage API connector with better error handling"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found. Some features may be limited.")
    
    def get_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote for a symbol with enhanced error handling"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
            
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check for various error conditions
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
                return None
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return MarketDataPoint(
                    symbol=symbol,
                    price=float(quote.get('05. price', 0)),
                    change=float(quote.get('09. change', 0)),
                    change_percent=float(quote.get('10. change percent', '0%').replace('%', '')),
                    volume=int(quote.get('06. volume', 0)),
                    timestamp=datetime.now(),
                    source='alpha_vantage',
                    metadata={'raw_data': quote}
                )
            else:
                logger.error(f"Unexpected Alpha Vantage response for {symbol}: {data}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching Alpha Vantage data for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching Alpha Vantage data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Alpha Vantage data for {symbol}: {e}")
            return None
        finally:
            time.sleep(self.rate_limit_delay)  # Rate limiting

class YahooFinanceConnector:
    """Enhanced Yahoo Finance connector with better error handling"""
    
    def __init__(self, timeout: int = 10):
        self.source = 'yahoo_finance'
        self.timeout = timeout
        self.network_utils = NetworkUtils()
    
    def get_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote using yfinance with enhanced error handling"""
        if not self.network_utils.check_host_resolution('query1.finance.yahoo.com'):
            logger.error("Yahoo Finance host unreachable")
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Set timeout for the session
            if hasattr(ticker, 'session'):
                ticker.session.timeout = self.timeout
            
            # Try multiple approaches to get data
            info = ticker.info
            hist = ticker.history(period="2d")
            
            # Determine the best data source
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
                volume = hist['Volume'].iloc[-1]
            elif info and ('currentPrice' in info or 'regularMarketPrice' in info):
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                change = info.get('regularMarketChange', 0)
                change_percent = info.get('regularMarketChangePercent', 0)
                volume = info.get('regularMarketVolume', 0)
            else:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            return MarketDataPoint(
                symbol=symbol,
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(volume) if volume else 0,
                timestamp=datetime.now(),
                source=self.source,
                metadata={
                    'sector': info.get('sector') if info else None,
                    'industry': info.get('industry') if info else None
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None

class MarketDataManager:
    """Enhanced main manager with better error handling and diagnostics"""
    
    def __init__(self, preferred_source: str = 'yahoo', alpha_vantage_key: Optional[str] = None, timeout: int = 10):
        self.alpha_vantage = AlphaVantageConnector(alpha_vantage_key, timeout)
        self.yahoo_finance = YahooFinanceConnector(timeout)
        self.preferred_source = preferred_source
        self.network_utils = NetworkUtils()
        
        # Asia tech stocks symbols
        self.asia_tech_symbols = [
            'TSM',      # Taiwan Semiconductor (TSMC)
            '005930.KS', # Samsung Electronics
            'ASML',     # ASML (Netherlands, but Asian exposure)
            'BABA',     # Alibaba
            'TCEHY',    # Tencent Holdings
            '6758.T',   # Sony Group Corp
            'UMC',      # United Microelectronics
            'ASX'       # Advanced Semiconductor Engineering
        ]
    
    def diagnose_connectivity(self) -> Dict[str, Any]:
        """Comprehensive connectivity diagnostics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'internet_connection': self.network_utils.check_internet_connection(),
            'api_endpoints': self.network_utils.test_api_endpoints(),
            'preferred_source': self.preferred_source,
            'alpha_vantage_configured': bool(self.alpha_vantage.api_key),
            'recommendations': self._get_connectivity_recommendations()
        }
    
    def _get_connectivity_recommendations(self) -> List[str]:
        """Get recommendations based on current connectivity status"""
        recommendations = []
        endpoints = self.network_utils.test_api_endpoints()
        
        if not endpoints.get('google_dns', False):
            recommendations.append("No internet connection - check network settings")
        elif not endpoints.get('yahoo_finance', False):
            recommendations.append("Yahoo Finance unreachable - try Alpha Vantage or check firewall/proxy")
        elif not endpoints.get('alpha_vantage', False):
            recommendations.append("Alpha Vantage unreachable - Yahoo Finance should work")
        
        if not self.alpha_vantage.api_key:
            recommendations.append("Consider setting ALPHA_VANTAGE_API_KEY for backup data source")
            
        return recommendations
    
    def get_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get quote from preferred source with intelligent fallback"""
        # First check connectivity
        connectivity = self.diagnose_connectivity()
        
        if not connectivity['internet_connection']:
            logger.error("No internet connection available")
            return None
        
        if self.preferred_source == 'alpha_vantage' and connectivity['alpha_vantage_configured']:
            data = self.alpha_vantage.get_quote(symbol)
            if data is None and connectivity['api_endpoints'].get('yahoo_finance', False):
                logger.info(f"Alpha Vantage failed for {symbol}, falling back to Yahoo Finance")
                data = self.yahoo_finance.get_quote(symbol)
        else:
            if connectivity['api_endpoints'].get('yahoo_finance', False):
                data = self.yahoo_finance.get_quote(symbol)
            elif connectivity['alpha_vantage_configured']:
                logger.info(f"Yahoo Finance unavailable for {symbol}, trying Alpha Vantage")
                data = self.alpha_vantage.get_quote(symbol)
            else:
                logger.error("No available data sources")
                data = None
                
        return data

# Factory function for easy initialization
def create_market_data_manager(source: str = 'yahoo', alpha_vantage_key: Optional[str] = None, timeout: int = 10) -> MarketDataManager:
    """Create a market data manager with specified configuration"""
    return MarketDataManager(preferred_source=source, alpha_vantage_key=alpha_vantage_key, timeout=timeout)

# Diagnostic function
def run_diagnostics():
    """Run comprehensive diagnostics"""
    print("=== Market Data Module Diagnostics ===")
    manager = create_market_data_manager()
    
    connectivity = manager.diagnose_connectivity()
    print(f"\nConnectivity Status ({connectivity['timestamp']}):")
    print(f"Internet Connection: {'✓' if connectivity['internet_connection'] else '✗'}")
    print(f"Yahoo Finance: {'✓' if connectivity['api_endpoints']['yahoo_finance'] else '✗'}")
    print(f"Alpha Vantage: {'✓' if connectivity['api_endpoints']['alpha_vantage'] else '✗'}")
    print(f"Alpha Vantage API Key: {'✓' if connectivity['alpha_vantage_configured'] else '✗'}")
    
    if connectivity['recommendations']:
        print("\nRecommendations:")
        for rec in connectivity['recommendations']:
            print(f"- {rec}")
    
    # Test a simple quote
    print(f"\nTesting quote for AAPL...")
    quote = manager.get_quote('AAPL')
    if quote:
        print(f"✓ Successfully retrieved: ${quote.price} ({quote.change:+.2f}, {quote.change_percent:+.2f}%)")
    else:
        print("✗ Failed to retrieve quote")

if __name__ == "__main__":
    run_diagnostics()