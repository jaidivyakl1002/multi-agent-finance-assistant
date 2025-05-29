# utils/data_conversion.py
import logging
from typing import Any, Union, Optional

logger = logging.getLogger(__name__)

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float with proper error handling
    
    Args:
        value: The value to convert
        default: Default value to return if conversion fails
        
    Returns:
        float: Converted value or default
    """
    if value is None:
        return default
    
    # If already a float
    if isinstance(value, float):
        return value
    
    # If it's an integer
    if isinstance(value, int):
        return float(value)
    
    # If it's a string, try to convert
    if isinstance(value, str):
        # Remove common formatting characters
        cleaned_value = value.strip().replace(',', '').replace('%', '')
        
        # Handle empty strings
        if not cleaned_value:
            return default
        
        # Handle special cases
        if cleaned_value.lower() in ['n/a', 'na', 'null', 'none', '-']:
            return default
            
        try:
            return float(cleaned_value)
        except ValueError:
            logger.warning(f"Could not convert '{value}' to float, using default {default}")
            return default
    
    # If it's a dictionary or other complex type
    if isinstance(value, (dict, list)):
        logger.warning(f"Cannot convert {type(value)} to float, using default {default}")
        return default
    
    # Last resort: try direct conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' of type {type(value)} to float, using default {default}")
        return default

def safe_percentage_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a percentage value to float
    
    Args:
        value: The percentage value to convert (e.g., "5.2%", 5.2, "5.2")
        default: Default value to return if conversion fails
        
    Returns:
        float: Converted percentage as decimal (5.2% -> 5.2)
    """
    if value is None:
        return default
    
    # Convert to string for processing
    str_value = str(value).strip()
    
    # Remove percentage sign if present
    if str_value.endswith('%'):
        str_value = str_value[:-1]
    
    # Use safe float conversion
    return safe_float_conversion(str_value, default)

def extract_numeric_value(data: dict, key: str, default: float = 0.0) -> float:
    """
    Safely extract and convert a numeric value from a dictionary
    
    Args:
        data: Dictionary containing the data
        key: Key to extract
        default: Default value if extraction fails
        
    Returns:
        float: Extracted and converted value
    """
    try:
        value = data.get(key)
        return safe_float_conversion(value, default)
    except Exception as e:
        logger.warning(f"Error extracting {key} from data: {e}")
        return default

def calculate_portfolio_metrics(market_data: dict, symbols: list = None) -> dict:
    """
    Calculate portfolio metrics with safe data handling
    
    Args:
        market_data: Dictionary of market data by symbol
        symbols: List of symbols to process
        
    Returns:
        dict: Portfolio metrics
    """
    if not market_data:
        return {
            "total_value": 0.0,
            "total_change": 0.0,
            "asia_tech_performance": 0.0,
            "risk_exposure": 0.0,
            "sector_breakdown": {},
            "errors": ["No market data available"]
        }
    
    symbols = symbols or list(market_data.keys())
    
    total_value = 0.0
    total_change = 0.0
    asia_tech_performance = 0.0
    errors = []
    sector_breakdown = {}
    
    for symbol in symbols:
        try:
            data = market_data.get(symbol, {})
            
            if not data or not isinstance(data, dict):
                errors.append(f"No valid data for {symbol}")
                continue
            
            # Extract values safely
            price = extract_numeric_value(data, "price", 0.0)
            change = extract_numeric_value(data, "change", 0.0)
            change_percent = extract_numeric_value(data, "change_percent", 0.0)
            market_cap = extract_numeric_value(data, "market_cap", 0.0)
            
            # Update totals
            total_value += price
            total_change += change
            
            # Asia tech symbols (customize based on your requirements)
            asia_tech_symbols = ["TSM", "005930.KS", "ASML", "2330.TW"]
            if symbol in asia_tech_symbols:
                asia_tech_performance += change_percent
            
            # Sector breakdown
            sector = data.get("sector", "Unknown")
            if sector not in sector_breakdown:
                sector_breakdown[sector] = {"count": 0, "total_value": 0.0, "avg_change": 0.0}
            
            sector_breakdown[sector]["count"] += 1
            sector_breakdown[sector]["total_value"] += price
            sector_breakdown[sector]["avg_change"] += change_percent
            
        except Exception as e:
            error_msg = f"Error processing {symbol}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Calculate averages for sectors
    for sector_data in sector_breakdown.values():
        if sector_data["count"] > 0:
            sector_data["avg_change"] /= sector_data["count"]
    
    # Calculate risk exposure (simplified)
    risk_exposure = abs(total_change / max(total_value, 1.0)) * 100
    
    return {
        "total_value": total_value,
        "total_change": total_change,
        "asia_tech_performance": asia_tech_performance,
        "risk_exposure": risk_exposure,
        "sector_breakdown": sector_breakdown,
        "processed_symbols": len([s for s in symbols if s in market_data]),
        "errors": errors
    }

def validate_market_data(market_data: dict) -> dict:
    """
    Validate and clean market data
    
    Args:
        market_data: Raw market data dictionary
        
    Returns:
        dict: Validated and cleaned market data
    """
    if not isinstance(market_data, dict):
        return {}
    
    cleaned_data = {}
    
    for symbol, data in market_data.items():
        if not isinstance(data, dict):
            continue
            
        cleaned_entry = {}
        
        # Standard numeric fields
        numeric_fields = [
            "price", "change", "change_percent", "volume", 
            "market_cap", "pe_ratio", "dividend_yield"
        ]
        
        for field in numeric_fields:
            if field in data:
                cleaned_entry[field] = safe_float_conversion(data[field])
        
        # String fields
        string_fields = ["sector", "industry", "name", "currency"]
        for field in string_fields:
            if field in data and data[field] is not None:
                cleaned_entry[field] = str(data[field])
        
        # Only add if we have some valid data
        if cleaned_entry:
            cleaned_data[symbol] = cleaned_entry
    
    return cleaned_data

# Example usage in your main.py _build_dynamic_portfolio_data method:
def build_dynamic_portfolio_data_safe(query_context: dict, market_response: dict) -> dict:
    """
    Safe version of _build_dynamic_portfolio_data with proper error handling
    """
    try:
        # Validate and clean market data
        market_data = validate_market_data(market_response.get("data", {}))
        
        if not market_data:
            return {
                "error": "No valid market data available",
                "portfolio_summary": {
                    "total_positions": 0,
                    "total_value": 0.0,
                    "asia_tech_performance": 0.0,
                    "risk_metrics": {"exposure": 0.0}
                }
            }
        
        # Calculate metrics safely
        metrics = calculate_portfolio_metrics(market_data)
        
        # Build portfolio data structure
        portfolio_data = {
            "portfolio_summary": {
                "total_positions": len(market_data),
                "total_value": metrics["total_value"],
                "asia_tech_performance": metrics["asia_tech_performance"],
                "risk_metrics": {
                    "exposure": metrics["risk_exposure"],
                    "sectors": metrics["sector_breakdown"]
                }
            },
            "holdings": []
        }
        
        # Add individual holdings
        for symbol, data in market_data.items():
            holding = {
                "symbol": symbol,
                "current_price": data.get("price", 0.0),
                "change": data.get("change", 0.0),
                "change_percent": data.get("change_percent", 0.0),
                "sector": data.get("sector", "Unknown"),
                "market_cap": data.get("market_cap", 0.0)
            }
            portfolio_data["holdings"].append(holding)
        
        # Add any processing errors
        if metrics["errors"]:
            portfolio_data["processing_errors"] = metrics["errors"]
        
        return portfolio_data
        
    except Exception as e:
        logger.error(f"Error in build_dynamic_portfolio_data_safe: {e}")
        return {
            "error": str(e),
            "portfolio_summary": {
                "total_positions": 0,
                "total_value": 0.0,
                "asia_tech_performance": 0.0,
                "risk_metrics": {"exposure": 0.0}
            }
        }