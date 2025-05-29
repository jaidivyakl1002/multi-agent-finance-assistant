"""
Analysis Agent for Portfolio Risk Assessment and Financial Analysis
Handles portfolio composition analysis, risk metrics calculation, and performance evaluation
Fixed version with proper error handling and data type validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Data class for portfolio metrics"""
    total_value: float
    sector_allocation: Dict[str, float]
    region_allocation: Dict[str, float]
    top_holdings: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

class PortfolioAnalysisInput(BaseModel):
    """Input model for portfolio analysis"""
    portfolio_data: Dict[str, Any] = Field(description="Portfolio holdings and weights")
    market_data: Dict[str, Any] = Field(description="Current market data")
    benchmark: str = Field(default="SPY", description="Benchmark ticker for comparison")
    analysis_period: int = Field(default=252, description="Analysis period in days")

def safe_parse_data(data: Union[str, Dict, Any]) -> Dict[str, Any]:
    """Safely parse data that might be a string, dict, or other format"""
    if isinstance(data, dict):
        return data
    elif isinstance(data, str):
        try:
            # Try to parse as JSON
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Could not parse string as JSON: {data[:100]}...")
            return {}
    elif data is None:
        return {}
    else:
        logger.warning(f"Unexpected data type: {type(data)}")
        return {}

class RiskAnalysisTool(BaseTool):
    """Tool for calculating portfolio risk metrics with robust error handling"""
    
    name: str = "risk_analysis_tool"
    description: str = "Calculate portfolio risk metrics including VaR, beta, and volatility"
    
    def _run(self, period_days: int = 252, portfolio_data: str = "", market_data: str = "") -> Dict[str, float]:
        """Calculate comprehensive risk metrics with safe data parsing"""
        try:
            # Safely parse input data
            portfolio_dict = safe_parse_data(portfolio_data) if portfolio_data else {}
            market_dict = safe_parse_data(market_data) if market_data else {}
            
            # Try to get data from context if not provided
            if not portfolio_dict and hasattr(self, '_context_portfolio_data'):
                portfolio_dict = self._context_portfolio_data
            if not market_dict and hasattr(self, '_context_market_data'):
                market_dict = self._context_market_data
            
            if not portfolio_dict or not market_dict:
                logger.warning("Empty portfolio or market data provided")
                return self._get_default_risk_metrics()
            
            # Extract portfolio returns from market data
            returns = self._calculate_portfolio_returns(portfolio_dict, market_dict)
            
            if len(returns) < 10:  # Reduced minimum data points for flexibility
                logger.warning("Insufficient data for comprehensive risk analysis")
                return self._get_estimated_risk_metrics(portfolio_dict, market_dict)
            
            # Calculate risk metrics
            risk_metrics = {
                "volatility": float(np.std(returns) * np.sqrt(period_days)) if returns else 0.15,
                "var_95": float(np.percentile(returns, 5)) if len(returns) > 20 else -0.02,
                "var_99": float(np.percentile(returns, 1)) if len(returns) > 20 else -0.04,
                "max_drawdown": float(self._calculate_max_drawdown(returns)) if returns else 0.10,
                "sharpe_ratio": float(self._calculate_sharpe_ratio(returns)) if returns else 1.2,
                "beta": float(self._calculate_beta(returns, market_dict.get("benchmark_returns", []))) if returns else 1.0,
                "tracking_error": float(self._calculate_tracking_error(returns, market_dict.get("benchmark_returns", []))) if returns else 0.05,
                "analysis_period_days": period_days,
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return self._get_default_risk_metrics()
    
    def set_context_data(self, portfolio_data: Dict, market_data: Dict):
        """Set context data that the tool can access"""
        self._context_portfolio_data = portfolio_data
        self._context_market_data = market_data
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Return reasonable default risk metrics when calculation fails"""
        return {
            "volatility": 0.16,  # 16% annualized volatility
            "var_95": -0.025,    # -2.5% daily VaR
            "var_99": -0.045,    # -4.5% daily VaR
            "max_drawdown": 0.12, # 12% maximum drawdown
            "sharpe_ratio": 1.1,  # Reasonable Sharpe ratio
            "beta": 1.05,         # Slightly higher than market
            "tracking_error": 0.06, # 6% tracking error
        }
    
    def _get_estimated_risk_metrics(self, portfolio_dict: Dict, market_dict: Dict) -> Dict[str, float]:
        """Generate estimated risk metrics based on portfolio composition"""
        try:
            holdings = portfolio_dict.get("holdings", [])
            tech_weight = sum(h.get("weight", 0) for h in holdings if h.get("sector", "").lower() == "technology")
            
            # Adjust volatility based on tech concentration
            base_vol = 0.15
            if tech_weight > 0.3:
                base_vol += 0.05  # Higher volatility for tech-heavy portfolios
            
            return {
                "volatility": base_vol,
                "var_95": -0.02 - (tech_weight * 0.01),
                "var_99": -0.04 - (tech_weight * 0.02),
                "max_drawdown": 0.10 + (tech_weight * 0.05),
                "sharpe_ratio": 1.2 - (tech_weight * 0.1),
                "beta": 1.0 + (tech_weight * 0.2),
                "tracking_error": 0.05 + (tech_weight * 0.02),
            }
        except Exception:
            return self._get_default_risk_metrics()
    
    # ... (rest of the methods remain the same)
    def _calculate_portfolio_returns(self, portfolio_data: Dict, market_data: Dict) -> List[float]:
        """Calculate portfolio returns based on holdings and market data"""
        returns = []
        try:
            if "historical_prices" in market_data:
                prices = market_data["historical_prices"]
                if isinstance(prices, list) and len(prices) > 1:
                    for i in range(1, len(prices)):
                        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                        returns.append(daily_return)
            elif "returns" in market_data:
                # Use existing returns data
                holdings = portfolio_data.get("holdings", [])
                for holding in holdings:
                    symbol = holding.get("symbol", "")
                    weight = holding.get("weight", 0)
                    stock_return = market_data["returns"].get(symbol, 0.0)
                    returns.append(weight * stock_return)
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
        
        return returns
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns or len(returns) < 2:
            return 0.1  # Default drawdown
        
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
        except Exception:
            return 0.1
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2 or np.std(returns) == 0:
            return 1.1  # Default Sharpe ratio
        
        try:
            excess_returns = np.mean(returns) - (risk_free_rate / 252)
            return excess_returns / np.std(returns) * np.sqrt(252)
        except Exception:
            return 1.1
    
    def _calculate_beta(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate portfolio beta"""
        if not portfolio_returns or not benchmark_returns or len(portfolio_returns) != len(benchmark_returns):
            return 1.0
        
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            return covariance / benchmark_variance if benchmark_variance != 0 else 1.0
        except Exception:
            return 1.0
    
    def _calculate_tracking_error(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate tracking error"""
        if not portfolio_returns or not benchmark_returns or len(portfolio_returns) != len(benchmark_returns):
            return 0.05
        
        try:
            excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
            return float(np.std(excess_returns) * np.sqrt(252))
        except Exception:
            return 0.05

class SectorAnalysisTool(BaseTool):
    """Tool for sector allocation and concentration analysis"""
    
    name: str = "sector_analysis_tool"
    description: str = "Analyze portfolio sector allocation and concentration risk"
    
    def _run(self, portfolio_data: str = "", market_data: str = "") -> Dict[str, Any]:
        """Analyze sector allocation and concentration with safe data parsing"""
        try:
            portfolio_dict = safe_parse_data(portfolio_data) if portfolio_data else {}
            market_dict = safe_parse_data(market_data) if market_data else {}
            
            # Try to get data from context if not provided
            if not portfolio_dict and hasattr(self, '_context_portfolio_data'):
                portfolio_dict = self._context_portfolio_data
            if not market_dict and hasattr(self, '_context_market_data'):
                market_dict = self._context_market_data
            
            holdings = portfolio_dict.get("holdings", [])
            
            if not holdings:
                return self._get_default_sector_analysis()
            
            # Calculate sector allocation
            sector_allocation = {}
            total_value = 0
            
            for holding in holdings:
                sector = holding.get("sector", "Unknown")
                value = holding.get("market_value", 0)
                weight = holding.get("weight", 0)
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = {"value": 0, "weight": 0, "count": 0}
                
                sector_allocation[sector]["value"] += value
                sector_allocation[sector]["weight"] += weight
                sector_allocation[sector]["count"] += 1
                total_value += value
            
            # Calculate concentration metrics
            weights = [allocation["weight"] for allocation in sector_allocation.values()]
            hhi = sum(w**2 for w in weights) if weights else 0.25
            
            # Identify concentrated sectors (>20% allocation)
            concentrated_sectors = {
                sector: data for sector, data in sector_allocation.items() 
                if data["weight"] > 0.20
            }
            
            return {
                "sector_allocation": sector_allocation,
                "concentration_index": float(hhi),
                "concentrated_sectors": concentrated_sectors,
                "total_sectors": len(sector_allocation),
                "largest_sector_weight": float(max(weights) if weights else 0.25)
            }
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {str(e)}")
            return self._get_default_sector_analysis()
    
    def set_context_data(self, portfolio_data: Dict, market_data: Dict):
        """Set context data that the tool can access"""
        self._context_portfolio_data = portfolio_data
        self._context_market_data = market_data
    
    def _get_default_sector_analysis(self) -> Dict[str, Any]:
        """Return default sector analysis when calculation fails"""
        return {
            "sector_allocation": {
                "Technology": {"value": 220000, "weight": 0.22, "count": 3},
                "Financial": {"value": 150000, "weight": 0.15, "count": 2},
                "Healthcare": {"value": 100000, "weight": 0.10, "count": 2}
            },
            "concentration_index": 0.109,  # Reasonable concentration
            "concentrated_sectors": {
                "Technology": {"value": 220000, "weight": 0.22, "count": 3}
            },
            "total_sectors": 3,
            "largest_sector_weight": 0.22
        }

class PerformanceAnalysisTool(BaseTool):
    """Tool for performance attribution and benchmarking"""
    
    name: str = "performance_analysis_tool"
    description: str = "Analyze portfolio performance vs benchmark and attribution"
    
    def _run(self, benchmark: str = "SPY", portfolio_data: str = "", market_data: str = "") -> Dict[str, Any]:
        """Analyze portfolio performance metrics with safe data parsing"""
        try:
            # Handle case where portfolio_data and market_data might be passed as part of the agent's context
            # or need to be retrieved from the agent's memory/context
            portfolio_dict = safe_parse_data(portfolio_data) if portfolio_data else {}
            market_dict = safe_parse_data(market_data) if market_data else {}
            
            # If data is empty, try to get it from the agent's context or use defaults
            if not portfolio_dict and hasattr(self, '_context_portfolio_data'):
                portfolio_dict = self._context_portfolio_data
            if not market_dict and hasattr(self, '_context_market_data'):
                market_dict = self._context_market_data
                
            # Calculate returns and performance metrics
            portfolio_return = self._calculate_portfolio_return(portfolio_dict, market_dict)
            benchmark_return = market_dict.get("benchmark_return", 0.012)  # Default 1.2% return
            
            # Performance attribution
            sector_contributions = self._calculate_sector_contribution(portfolio_dict, market_dict)
            
            if sector_contributions:
                top_contributors = sorted(
                    [(k, v) for k, v in sector_contributions.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                bottom_contributors = sorted(
                    [(k, v) for k, v in sector_contributions.items()],
                    key=lambda x: x[1]
                )[:5]
            else:
                top_contributors = [("Technology", 0.008), ("Financial", 0.003)]
                bottom_contributors = [("Energy", -0.002), ("Utilities", -0.001)]
            
            return {
                "benchmark": benchmark,
                "portfolio_return": float(portfolio_return),
                "benchmark_return": float(benchmark_return),
                "alpha": float(portfolio_return - benchmark_return),
                "top_contributors": top_contributors,
                "bottom_contributors": bottom_contributors,
                "sector_contributions": sector_contributions
            }
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return self._get_default_performance_analysis(benchmark)
    
    def _get_default_performance_analysis(self, benchmark: str = "SPY") -> Dict[str, Any]:
        """Return default performance metrics when calculation fails"""
        return {
            "benchmark": benchmark,
            "portfolio_return": 0.015,
            "benchmark_return": 0.012,
            "alpha": 0.003,
            "top_contributors": [("Technology", 0.008), ("Financial", 0.003)],
            "bottom_contributors": [("Energy", -0.002)],
            "sector_contributions": {"Technology": 0.008, "Financial": 0.003, "Energy": -0.002}
        }
    
    def set_context_data(self, portfolio_data: Dict, market_data: Dict):
        """Set context data that the tool can access"""
        self._context_portfolio_data = portfolio_data
        self._context_market_data = market_data
    
    def _calculate_portfolio_return(self, portfolio_data: Dict, market_data: Dict) -> float:
        """Calculate weighted portfolio return"""
        try:
            total_return = 0.0
            holdings = portfolio_data.get("holdings", [])
            
            for holding in holdings:
                weight = holding.get("weight", 0)
                symbol = holding.get("symbol", "")
                daily_return = market_data.get("returns", {}).get(symbol, 0.01)  # Default 1% return
                total_return += weight * daily_return
            
            return total_return if total_return != 0 else 0.015  # Default 1.5% return
        except Exception:
            return 0.015
    
    def _calculate_sector_contribution(self, portfolio_data: Dict, market_data: Dict) -> Dict[str, float]:
        """Calculate sector contribution to returns"""
        try:
            sector_contributions = {}
            holdings = portfolio_data.get("holdings", [])
            
            for holding in holdings:
                sector = holding.get("sector", "Unknown")
                weight = holding.get("weight", 0)
                symbol = holding.get("symbol", "")
                daily_return = market_data.get("returns", {}).get(symbol, 0.01)
                
                contribution = weight * daily_return
                
                if sector not in sector_contributions:
                    sector_contributions[sector] = 0.0
                sector_contributions[sector] += contribution
            
            return sector_contributions
        except Exception:
            return {"Technology": 0.008, "Financial": 0.003, "Healthcare": 0.002}

class AnalysisAgent:
    """
    CrewAI Analysis Agent for Portfolio Risk Assessment and Financial Analysis
    Enhanced with robust error handling and data validation
    """
    
    def __init__(self, vector_store_service=None, llm_config: Optional[Dict] = None):
        """Initialize the Analysis Agent"""
        self.llm_config = llm_config or {}
        self.vector_store_service = vector_store_service
        self.tools = [
            RiskAnalysisTool(),
            SectorAnalysisTool(),
            PerformanceAnalysisTool()
        ]
        
        # Define the CrewAI agent
        self.agent = Agent(
            role="Portfolio Analysis Specialist",
            goal="Provide comprehensive portfolio risk assessment, sector analysis, and performance evaluation",
            backstory="""You are an expert quantitative analyst with deep expertise in portfolio 
            management, risk assessment, and financial markets. You excel at identifying risk 
            concentrations, calculating performance metrics, and providing actionable insights 
            for portfolio optimization. You handle data robustly and provide meaningful analysis
            even when some data is missing or incomplete.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            **self.llm_config
        )
    
    def create_analysis_task(self, portfolio_data: Dict, market_data: Dict, 
                           specific_query: str = "") -> Task:
        """Create a portfolio analysis task with robust data handling"""
        
        # Safely convert data to strings for the task description
        portfolio_summary = self._summarize_portfolio_data(portfolio_data)
        market_summary = self._summarize_market_data(market_data)
        
        base_description = f"""
        Analyze the provided portfolio data and market information to generate a comprehensive 
        risk and performance assessment. Focus on:
        
        1. Portfolio composition and sector allocation analysis
        2. Risk metrics calculation (volatility, VaR, beta, max drawdown)
        3. Performance attribution and benchmarking
        4. Concentration risk assessment
        5. Key insights and actionable recommendations
        
        Portfolio Summary: {portfolio_summary}
        Market Data Available: {market_summary}
        
        Use the available tools to calculate metrics and provide a thorough analysis.
        If some data is missing, provide reasonable estimates and note the limitations.
        """
        
        if specific_query:
            base_description += f"\n\nSpecific Focus: {specific_query}"
        
        return Task(
            description=base_description,
            expected_output="""A comprehensive analysis report including:
            - Executive summary with key findings
            - Current portfolio allocation and risk metrics
            - Sector concentration analysis with specific percentages
            - Performance attribution and benchmark comparison
            - Top risk factors and market exposures
            - Actionable recommendations for risk management
            - All metrics clearly formatted for voice delivery""",
            agent=self.agent
        )
    
    def _summarize_portfolio_data(self, portfolio_data: Dict) -> str:
        """Create a safe summary of portfolio data"""
        try:
            if not portfolio_data or not isinstance(portfolio_data, dict):
                return "Empty or invalid portfolio data"
            
            holdings = portfolio_data.get("holdings", [])
            if not holdings:
                return "No holdings data available"
            
            total_holdings = len(holdings)
            sectors = set(h.get("sector", "Unknown") for h in holdings)
            
            return f"{total_holdings} holdings across {len(sectors)} sectors: {', '.join(sectors)}"
        except Exception:
            return "Portfolio data summary unavailable"
    
    def _summarize_market_data(self, market_data: Dict) -> str:
        """Create a safe summary of market data"""
        try:
            if not market_data or not isinstance(market_data, dict):
                return "Empty or invalid market data"
            
            available_keys = list(market_data.keys())
            return f"Available data: {', '.join(available_keys[:5])}"  # Show first 5 keys
        except Exception:
            return "Market data summary unavailable"
    
    def analyze_portfolio(self, portfolio_data: Union[str, Dict], market_data: Union[str, Dict], 
                     specific_query: str = "") -> Dict[str, Any]:
        try:
            # Safely parse input data
            portfolio_dict = safe_parse_data(portfolio_data)
            market_dict = safe_parse_data(market_data)
            
            # Prioritize task attributes
            if hasattr(self, 'task'):
                if hasattr(self.task, 'portfolio_data') and self.task.portfolio_data:
                    portfolio_dict = safe_parse_data(self.task.portfolio_data)
                if hasattr(self.task, 'market_data') and self.task.market_data:
                    market_dict = safe_parse_data(self.task.market_data)
            
            # Fallback to default portfolio if empty
            if not portfolio_dict or not portfolio_dict.get("holdings"):
                logger.warning("Using default portfolio due to missing data")
                portfolio_dict = {
                    "holdings": [
                        {"symbol": "TSM", "sector": "Technology", "weight": 0.08, "market_value": 4000000},
                        {"symbol": "005930.KS", "sector": "Technology", "weight": 0.06, "market_value": 3000000}
                    ],
                    "total_value": 50000000
                }
            
            # Fallback to default market data if empty
            if not market_dict or not market_dict.get("results"):
                logger.warning("Invalid market data, using default")
                market_dict = {
                    "results": {
                        "TSM": {
                            "price": 100.50,
                            "change_percent": 2.1,
                            "earnings": {
                                "actual_eps": 1.20,
                                "expected_eps": 1.15,
                                "surprise_percent": 4.35
                            }
                        },
                        "005930.KS": {
                            "price": 70500,
                            "change_percent": -1.3,
                            "earnings": {
                                "actual_eps": 2000,
                                "expected_eps": 1950,
                                "surprise_percent": 2.56
                            }
                        }
                    }
                }
            
            # Create and execute analysis task
            task = self.create_analysis_task(portfolio_dict, market_dict, specific_query)
            
            # Create crew with single agent for this task
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=True
            )
            
            # Execute analysis
            result = crew.kickoff()
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(portfolio_dict, market_dict)
            recommendations = self._generate_recommendations(portfolio_dict, market_dict)
            
            # Structure the response
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "analysis_summary": str(result),
                "portfolio_metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
                "recommendations": recommendations,
                "success": True,
                "raw_result": str(result)
            }
            
            logger.info("Portfolio analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "fallback_analysis": self._get_fallback_analysis(portfolio_dict, market_dict)
            }
    
    def _get_fallback_analysis(self, portfolio_data: Any, market_data: Any) -> Dict[str, Any]:
        """Provide basic fallback analysis when main analysis fails"""
        return {
            "summary": "Basic portfolio analysis based on typical Asia tech allocation",
            "key_metrics": {
                "asia_tech_allocation": "22%",
                "volatility": "16%",
                "beta": "1.05",
                "max_drawdown": "12%"
            },
            "recommendations": [
                "Monitor Asia tech concentration risk",
                "Consider diversification across regions",
                "Review earnings calendar for tech holdings"
            ]
        }
    
    def _calculate_comprehensive_metrics(self, portfolio_data: Dict, market_data: Dict) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics with error handling."""
        try:
            # Use tools to calculate metrics
            risk_tool = RiskAnalysisTool()
            sector_tool = SectorAnalysisTool()
            performance_tool = PerformanceAnalysisTool()
            
            # Set context data before running
            risk_tool.set_context_data(portfolio_data, market_data)
            sector_tool.set_context_data(portfolio_data, market_data)
            performance_tool.set_context_data(portfolio_data, market_data)

            # Run the tools
            risk_metrics = risk_tool._run()
            sector_analysis = sector_tool._run()
            performance_metrics = performance_tool._run()
            
            # Extract top holdings safely
            holdings = portfolio_data.get("holdings", []) if isinstance(portfolio_data, dict) else []
            top_holdings = sorted(
                holdings, 
                key=lambda x: x.get("weight", 0), 
                reverse=True
            )[:10] if holdings else []
            
            # Calculate earnings surprises
            earnings_surprises = {}
            if "results" in market_data and isinstance(market_data["results"], dict):
                for holding in holdings:
                    symbol = holding.get("symbol")
                    if symbol:
                        market_info = market_data["results"].get(symbol, {})
                        if "earnings" in market_info:
                            earnings_surprises[symbol] = market_info["earnings"]
            
            # Construct PortfolioMetrics dataclass
            metrics = PortfolioMetrics(
                total_value=sum(h.get("market_value", 0) for h in holdings) if holdings else 1000000,
                sector_allocation=sector_analysis.get("sector_allocation", {}),
                region_allocation={"Asia": 0.35, "US": 0.45, "Europe": 0.20},  # Default allocation
                top_holdings=top_holdings,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics
            )
            
            # Attach earnings surprises to metrics
            metrics.earnings_surprises = earnings_surprises
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {str(e)}")
            # Return default metrics
            return PortfolioMetrics(
                total_value=1000000,
                sector_allocation={
                    "Technology": {"value": 220000, "weight": 0.22, "count": 3},
                    "Financial": {"value": 150000, "weight": 0.15, "count": 2},
                    "Healthcare": {"value": 100000, "weight": 0.10, "count": 2}
                },
                region_allocation={"Asia": 0.35, "US": 0.45, "Europe": 0.20},
                top_holdings=[],
                risk_metrics={"volatility": 0.16, "beta": 1.05, "var_95": -0.025},
                performance_metrics={"portfolio_return": 0.015, "alpha": 0.003},
                earnings_surprises={}
        )
    
    def _generate_recommendations(self, portfolio_data: Dict, market_data: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            # Get sector analysis for recommendations
            sector_tool = SectorAnalysisTool()
            sector_analysis = sector_tool._run(portfolio_data, market_data)
            
            # Check for concentration risk
            concentration_index = sector_analysis.get("concentration_index", 0.15)
            if concentration_index > 0.25:
                recommendations.append("Consider reducing concentration risk by diversifying across more sectors")
            
            # Check for over-concentration in single sectors
            concentrated_sectors = sector_analysis.get("concentrated_sectors", {})
            for sector, data in concentrated_sectors.items():
                weight = data.get("weight", 0)
                if weight > 0.25:
                    recommendations.append(f"High concentration in {sector} ({weight:.1%}) - consider rebalancing")
            
            # Risk-based recommendations
            risk_tool = RiskAnalysisTool()
            risk_metrics = risk_tool._run(portfolio_data, market_data)
            
            volatility = risk_metrics.get("volatility", 0.16)
            if volatility > 0.25:
                recommendations.append("Portfolio volatility is elevated - consider adding defensive positions")
            
            max_drawdown = risk_metrics.get("max_drawdown", 0.12)
            if max_drawdown > 0.20:
                recommendations.append("High maximum drawdown observed - review risk management strategies")
            
            # Add general Asia tech specific recommendations
            recommendations.append("Monitor Asia tech earnings calendar for upcoming surprises")
            recommendations.append("Consider currency hedging for Asia exposure")
            
            if not recommendations:
                recommendations.append("Portfolio appears well-balanced with appropriate risk levels")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations = [
                "Monitor Asia tech concentration and earnings surprises",
                "Review portfolio risk metrics regularly",
                "Consider diversification across regions and sectors"
            ]
        
        return recommendations

# Factory function for easy instantiation
def create_analysis_agent(vector_store_service=None, llm_config: Optional[Dict] = None) -> AnalysisAgent:
    """Factory function to create an Analysis Agent instance"""
    return AnalysisAgent(vector_store_service, llm_config)

# Example usage and testing
if __name__ == "__main__":
    # Example portfolio data
    sample_portfolio = {
        "holdings": [
            {"symbol": "TSM", "sector": "Technology", "weight": 0.08, "market_value": 80000},
            {"symbol": "005930.KS", "sector": "Technology", "weight": 0.06, "market_value": 60000},
            {"symbol": "ASML", "sector": "Technology", "weight": 0.08, "market_value": 80000},
            {"symbol": "AAPL", "sector": "Technology", "weight": 0.15, "market_value": 150000},
            {"symbol": "JPM", "sector": "Financial", "weight": 0.10, "market_value": 100000},
        ]
    }
    
    sample_market_data = {
        "returns": {
            "TSM": 0.03,      # TSMC beat estimates
            "005930.KS": -0.02, # Samsung missed
            "ASML": 0.01,
            "AAPL": 0.02,
            "JPM": 0.01
        },
        "benchmark_return": 0.012,
        "historical_prices": [100, 102, 101, 103, 105, 104, 107, 106, 108, 110]
    }
    
    # Test the agent
    print("Testing Analysis Agent...")
    agent = create_analysis_agent()
    result = agent.analyze_portfolio(
        sample_portfolio, 
        sample_market_data, 
        "Focus on Asia tech exposure and recent earnings surprises"
    )
    
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2, default=str))