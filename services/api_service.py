# services/agent_routes.py
from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any
from orchestrator.crew_manager import EnhancedAgentCrewManager
import logging
import json
from data_ingestion.market_data import MarketDataProvider
from agents.api_agent import MarketDataCrew, get_market_data_crew
from services.voice_service import get_voice_service
from fastapi import UploadFile, File
import datetime

market_provider = MarketDataProvider()

voice_service = get_voice_service()

router = APIRouter()
logger = logging.getLogger(__name__)
crew_manager = EnhancedAgentCrewManager()
market_data_crew = get_market_data_crew()

@router.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return await voice_service.transcribe_upload(file)

@router.post("/voice/synthesize")
async def synthesize_speech(payload: dict):
    text = payload.get("text")
    engine = payload.get("engine", "local")
    return voice_service.generate_speech(text, engine)

@router.post("/voice/query")
async def process_voice_query(file: UploadFile = File(...)):
    return await voice_service.process_voice_query(file)

@router.post("/agent/api/analyze")
async def analyze_market_data(payload: Dict[str, Any]):
    """
    Enhanced market data analysis using CrewAI agent crew
    Replaces direct tool usage with proper agent orchestration
    """
    try:
        query = payload.get("query", "Analyze current market conditions")
        context = payload.get("context", {})
        
        # Build analysis context from payload
        analysis_context = {
            "symbols": payload.get("symbols", "TSM,005930.KS,ASML"),
            "data_type": payload.get("data_type", "quote"),
            "period": payload.get("period", "1d"),
            "focus_areas": payload.get("focus_areas", []),
            "include_sentiment": payload.get("include_sentiment", True),
            "include_risk_metrics": payload.get("include_risk_metrics", True)
        }
        analysis_context.update(context)
        
        # Use market data crew for comprehensive analysis
        result = market_data_crew.analyze_market_query(query, analysis_context)
        
        return {
            "success": True,
            "query": query,
            "analysis": result,
            "context": analysis_context,
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_type": "market_data_crew"
        }
        
    except Exception as e:
        logger.error(f"Market data analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@router.post("/agent/api/asia-tech")
async def get_asia_tech_analysis(payload: Dict[str, Any]):
    """
    Specialized Asia tech market analysis using dedicated crew task
    """
    try:
        query = payload.get("query", "Provide Asia tech market overview with risk exposure analysis")
        
        # Use specialized Asia tech analysis
        result = market_data_crew.get_asia_tech_overview(query)
        
        return {
            "success": True,
            "query": query,
            "asia_tech_analysis": result,
            "timestamp": datetime.datetime.now().isoformat(),
            "specialization": "asia_pacific_tech"
        }
        
    except Exception as e:
        logger.error(f"Asia tech analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@router.post("/agent/api/earnings")
async def analyze_earnings(payload: Dict[str, Any]):
    """
    Specialized earnings analysis using CrewAI agent
    """
    try:
        query = payload.get("query", "Analyze recent earnings results and surprises")
        companies = payload.get("companies", ["TSM", "005930.KS"])
        
        # Use specialized earnings analysis
        result = market_data_crew.analyze_earnings(query, companies)
        
        return {
            "success": True,
            "query": query,
            "companies": companies,
            "earnings_analysis": result,
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_type": "earnings_focus"
        }
        
    except Exception as e:
        logger.error(f"Earnings analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@router.post("/agent/workflow/morning-brief")
async def run_morning_brief_workflow(payload: Dict[str, Any]):
    try:
        query = payload.get("query", "What’s our risk exposure in Asia tech stocks today?")
        crew = crew_manager.create_morning_brief_crew(query)
        result = crew.kickoff()

        return {
            "success": True,
            "query": query,
            "response": str(result),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in morning brief workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Morning brief generation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


@router.post("/agent/api")
async def call_api_agent_legacy(payload: Dict[str, Any]):
    """
    Legacy endpoint for backward compatibility
    Redirects to enhanced analysis endpoint
    """
    try:
        # Transform legacy payload to new format
        enhanced_payload = {
            "query": f"Analyze market data for symbols: {payload.get('symbols', 'TSM,005930.KS')}",
            "symbols": payload.get("symbols", "TSM,005930.KS"),
            "data_type": payload.get("data_type", "quote"),
            "period": payload.get("period", "1d"),
            "context": payload.get("context", {})
        }

        # Use enhanced analysis
        result = await analyze_market_data(enhanced_payload)

        # Format for legacy compatibility
        if result.get("success"):
            structured_data = result.get("analysis", "")
            try:
                structured_data_json = json.loads(structured_data)
            except:
                structured_data_json = {"summary": structured_data}

            return {
                "result": structured_data_json,
                "success": True,
                "legacy_compatibility": True
            }

        return {"error": "Analysis failed", "success": False}

    except Exception as e:
        logger.error(f"Legacy API agent error: {e}")
        return {"error": str(e), "success": False}


async def analyze_market_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main endpoint logic to fetch and analyze market data
    """
    try:
        symbols = payload.get("symbols", "TSM,005930.KS").split(",")
        data_type = payload.get("data_type", "quote")
        period = payload.get("period", "1d")

        # Fetch data
        data = market_provider.fetch_market_data(symbols, data_type, period)

        if not data:
            return {"success": False, "analysis": "No data found for requested symbols."}

        # Prepare summary output (optional)
        summary = {
            symbol: {
                "price": d.get("price"),
                "change": d.get("change"),
                "change_percent": d.get("change_percent"),
                "volume": d.get("volume"),
                "market_cap": d.get("market_cap"),
                "sector": d.get("sector"),
                "industry": d.get("industry"),
            } for symbol, d in data.items() if isinstance(d, dict)
        }

        return {
            "success": True,
            "analysis": json.dumps(summary, indent=2)
        }

    except Exception as e:
        logger.error(f"Error in analyze_market_data: {e}")
        return {"success": False, "analysis": f"Exception occurred: {str(e)}"}

@router.post("/agent/scrape")
async def call_scraping_agent(payload: Dict[str, Any]):
    try:
        tool = crew_manager.scraping_agent.tools[0]  # SECFilingsTool
        ticker = payload.get("ticker", "TSM")
        filing_types = payload.get("filing_types", ["10-K", "10-Q"])
        limit = payload.get("limit", 5)
        return tool._run(ticker, filing_types, limit)
    except Exception as e:
        return {"error": str(e)}

@router.post("/agent/retrieve")
async def call_retriever_agent(payload: Dict[str, Any]):
    try:
        query = payload.get("query", "Asia tech exposure")
        context = payload.get("context", "")
        category = payload.get("category", "")
        context["category"] = category 

        
        # Create task and crew properly
        task = crew_manager.retriever_agent.create_retrieval_task(query, context)
        crew = crew_manager.create_crew([crew_manager.retriever_agent.agent], [task])
        
        # Execute the crew and get results
        result = crew.kickoff()
        
        # Handle CrewAI result format
        if hasattr(result, 'raw'):
            return {"result": result.raw}
        elif hasattr(result, 'output'):
            return {"result": result.output}
        else:
            return {"result": str(result)}
            
    except Exception as e:
        logger.error(f"Error in retriever agent: {e}")
        return {"error": str(e)}

@router.post("/agent/retrieve/direct")
async def direct_search(payload: Dict[str, Any]):
    try:
        query = payload.get("query", "")
        k = payload.get("k", 5)
        filters = payload.get("filters", {})
        
        # Use direct search method from retriever agent
        results = crew_manager.retriever_agent.search_documents(query, k, filters)
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error in direct search: {e}")
        return {"error": str(e)}

@router.post("/agent/analyze")
async def call_analysis_agent(payload: Dict[str, Any]):
    try:
        portfolio = payload.get("portfolio_data", {})
        market = payload.get("market_data", {})
        query = payload.get("query", "")
        agent = crew_manager.analysis_agent
        return agent.analyze_portfolio(portfolio, market, query)
    except Exception as e:
        return {"error": str(e)}

@router.post("/agent/narrative")
async def call_language_agent(payload: Dict[str, Any]):
    try:
        query = payload.get("query", "What happened in Asia tech today?")
        market = payload.get("market_data", {})
        analysis = payload.get("analysis_results", {})
        docs = payload.get("context_documents", [])
        agent = crew_manager.language_agent
        return {"brief": agent.synthesize_market_brief(query, market, analysis, docs)}
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/agent/orchestrated")
async def orchestrated_workflow(payload: Dict[str, Any]):
    """
    Orchestrated workflow using multiple agents in sequence
    This mimics your process_text_query functionality
    """
    try:
        query = payload.get("query", "What's our risk exposure in Asia tech stocks today?")
        settings = payload.get("settings", {
            "symbols": "TSM,005930.KS,ASML",
            "data_type": "quote",
            "period": "1d"
        })
        
        # Use the crew manager's morning brief functionality
        result = crew_manager.process_voice_query(query)
        
        return {
            "orchestrated_result": result,
            "status": "success",
            "workflow": "morning_brief"
        }
        
    except Exception as e:
        logger.error(f"Orchestrated workflow error: {e}")
        return {"error": str(e), "status": "error"}

@router.post("/agent/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        stt_tool = crew_manager.voice_agent.tools[0]  # SpeechToTextTool
        return {"transcript": stt_tool._run(audio_input=audio_bytes)}
    except Exception as e:
        return {"error": str(e)}

@router.post("/agent/tts")
async def text_to_speech(payload: Dict[str, Any]):
    try:
        text = payload.get("text", "Hello, this is a sample TTS output.")
        tts_tool = crew_manager.voice_agent.tools[1]  # TextToSpeechTool
        return {"result": tts_tool._run(text)}
    except Exception as e:
        return {"error": str(e)}
