import streamlit as st
import httpx
import io
import json
from typing import Dict, Any, Optional
import asyncio
import time
from datetime import datetime
from streamlit_app.components import AudioRecorder, MarketDashboard
import re


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Finance Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FASTAPI_BASE_URL = "http://localhost:8000"  # Adjust based on your FastAPI server

class FinanceAssistantUI:
    def __init__(self):
        self.client = httpx.Client(base_url=FASTAPI_BASE_URL, timeout=30.0)
        self.recorder = AudioRecorder()
        self.dashboard = MarketDashboard()
        
        # Initialize portfolio context
        self.portfolio_context = self._initialize_portfolio_context()
    
    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()

    def _initialize_portfolio_context(self) -> Dict[str, Any]:
        """Initialize realistic portfolio context data"""
        return {
            "total_aum": 50_000_000,  # $50M AUM
            "last_updated": datetime.now().isoformat(),
            "regional_allocations": {
                "asia_tech": 0.18,  # 18% baseline allocation
                "us_tech": 0.25,
                "europe_financials": 0.15,
                "emerging_markets": 0.12,
                "bonds": 0.20,
                "cash": 0.10
            },
            "key_holdings": {
                "TSM": {"allocation": 0.08, "shares": 15000, "cost_basis": 95.50},
                "005930.KS": {"allocation": 0.06, "shares": 2500, "cost_basis": 68000},  # Samsung
                "AAPL": {"allocation": 0.12, "shares": 8000, "cost_basis": 150.25},
                "NVDA": {"allocation": 0.08, "shares": 1200, "cost_basis": 220.75}
            },
            "risk_metrics": {
                "var_1d": 0.025,  # 2.5% 1-day VaR
                "beta": 1.15,
                "sharpe_ratio": 1.45,
                "max_drawdown": 0.08
            },
            "sector_exposure": {
                "technology": 0.42,
                "semiconductors": 0.18,
                "consumer_electronics": 0.12,
                "financials": 0.15,
                "other": 0.13
            }
        }

    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract context from user query to make retrieval more targeted"""
        query_lower = query.lower()
        
        # Initialize context
        context = {
            "query_type": "general",
            "sectors": [],
            "regions": [],
            "time_horizon": "current",
            "analysis_focus": [],
            "entities": []
        }
        
        # Detect query type
        if any(word in query_lower for word in ["risk", "exposure", "var", "volatility"]):
            context["query_type"] = "risk_analysis"
        elif any(word in query_lower for word in ["earnings", "results", "beat", "miss"]):
            context["query_type"] = "earnings_analysis"
        elif any(word in query_lower for word in ["performance", "return", "gain", "loss"]):
            context["query_type"] = "performance_analysis"
        elif any(word in query_lower for word in ["allocation", "position", "holding"]):
            context["query_type"] = "portfolio_composition"
        elif any(word in query_lower for word in ["brief", "summary", "overview"]):
            context["query_type"] = "market_brief"
        
        # Extract sectors
        sector_keywords = {
            "tech": ["tech", "technology", "semiconductor", "software"],
            "financials": ["bank", "financial", "insurance"],
            "healthcare": ["pharma", "biotech", "medical"],
            "energy": ["oil", "gas", "renewable"]
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                context["sectors"].append(sector)
        
        # Extract regions
        region_keywords = {
            "asia": ["asia", "asian", "japan", "korea", "taiwan", "china"],
            "us": ["us", "usa", "america", "american"],
            "europe": ["europe", "european", "eu"]
        }
        
        for region, keywords in region_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                context["regions"].append(region)
        
        # Extract specific entities (tickers, companies)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, query)
        context["entities"].extend(potential_tickers)
        
        # Extract time horizon
        if any(word in query_lower for word in ["today", "now", "current"]):
            context["time_horizon"] = "current"
        elif any(word in query_lower for word in ["week", "weekly"]):
            context["time_horizon"] = "weekly"
        elif any(word in query_lower for word in ["month", "monthly"]):
            context["time_horizon"] = "monthly"
        
        # Extract analysis focus
        analysis_keywords = {
            "surprise": ["surprise", "beat", "miss", "unexpected"],
            "sentiment": ["sentiment", "mood", "outlook"],
            "technical": ["support", "resistance", "trend", "momentum"],
            "fundamental": ["valuation", "pe", "revenue", "growth"]
        }
        
        for focus, keywords in analysis_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                context["analysis_focus"].append(focus)
        
        return context

    def _build_dynamic_portfolio_data(self, query_context: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build dynamic portfolio data based on query context and current market data"""
        base_portfolio = self.portfolio_context.copy()
        
        # Update allocations based on market movements (simulated dynamic behavior)
        if market_data.get("result"):
            market_results = market_data["result"]
            
            # Calculate new asia tech allocation based on market performance
            asia_tech_performance = 0
            asia_tech_count = 0
            
            for symbol, data in market_results.items():
                if symbol in ["TSM", "005930.KS"] or any(region in ["asia", "tech"] for region in query_context.get("regions", [])):
                    if isinstance(data, dict) and "change_percent" in data:
                        change_percent = data.get("change_percent", 0)
                        # Safe conversion to float, handling None and non-numeric values
                        try:
                            if change_percent is not None:
                                asia_tech_performance += float(change_percent)
                                asia_tech_count += 1
                        except (ValueError, TypeError):
                            # Skip invalid values and continue
                            continue
            
            if asia_tech_count > 0:
                avg_performance = asia_tech_performance / asia_tech_count
                # Simulate allocation drift based on performance
                base_allocation = base_portfolio["regional_allocations"]["asia_tech"]
                # If performance is positive, allocation naturally increases due to price appreciation
                new_allocation = base_allocation * (1 + avg_performance / 100)
                base_portfolio["regional_allocations"]["asia_tech"] = min(new_allocation, 0.30)  # Cap at 30%
        
        # Add query-specific context
        dynamic_portfolio = {
            **base_portfolio,
            "query_context": query_context,
            "focused_holdings": self._get_focused_holdings(query_context),
            "relevant_metrics": self._get_relevant_metrics(query_context),
            "risk_factors": self._identify_risk_factors(query_context, market_data)
        }
        
        return dynamic_portfolio

    def _get_focused_holdings(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get holdings relevant to the query context"""
        focused = {}
        
        # Filter holdings based on query regions and sectors
        for symbol, holding_data in self.portfolio_context["key_holdings"].items():
            include_holding = False
            
            # Include if specific entity mentioned
            if symbol in query_context.get("entities", []):
                include_holding = True
            
            # Include based on regions
            if "asia" in query_context.get("regions", []):
                if symbol in ["TSM", "005930.KS"]:  # Asian stocks
                    include_holding = True
            
            # Include based on sectors
            if "tech" in query_context.get("sectors", []):
                if symbol in ["TSM", "AAPL", "NVDA", "005930.KS"]:  # Tech stocks
                    include_holding = True
            
            # If no specific filters, include major holdings
            if not query_context.get("regions") and not query_context.get("sectors") and not query_context.get("entities"):
                include_holding = True
            
            if include_holding:
                focused[symbol] = holding_data
        
        return focused

    def _get_relevant_metrics(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk metrics relevant to the query"""
        base_metrics = self.portfolio_context["risk_metrics"].copy()
        
        # Add context-specific metrics
        if query_context["query_type"] == "risk_analysis":
            base_metrics.update({
                "concentration_risk": 0.25,  # Top 5 holdings concentration
                "sector_concentration": max(self.portfolio_context["sector_exposure"].values()),
                "currency_exposure": {"USD": 0.65, "KRW": 0.15, "TWD": 0.20}
            })
        
        return base_metrics

    def _identify_risk_factors(self, query_context: Dict[str, Any], market_data: Dict[str, Any]) -> list:
        """Identify current risk factors based on context and market data"""
        risk_factors = []
        
        # Market-based risk factors
        if market_data.get("result"):
            for symbol, data in market_data["result"].items():
                if isinstance(data, dict):
                    change = float(data.get("change_percent", 0))
                    if abs(change) > 5:  # Significant movement
                        risk_factors.append(f"High volatility in {symbol}: {change:+.1f}%")
        
        # Context-based risk factors
        if "asia" in query_context.get("regions", []):
            risk_factors.extend([
                "Geopolitical tensions in Asia-Pacific region",
                "Currency fluctuation risk (KRW, TWD vs USD)",
                "Regulatory changes in semiconductor industry"
            ])
        
        if "tech" in query_context.get("sectors", []):
            risk_factors.extend([
                "Interest rate sensitivity in growth stocks",
                "Supply chain disruptions in semiconductor sector"
            ])
        
        return risk_factors

    async def call_api_endpoint(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async call to FastAPI endpoint"""
        try:
            async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL, timeout=30.0) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    async def upload_voice_query(self, audio_file) -> Dict[str, Any]:
        """Upload voice file and process query"""
        try:
            files = {"file": (audio_file.name, audio_file.read(), "audio/wav")}
            async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL, timeout=60.0) as client:
                response = await client.post("/voice/query", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": f"Voice processing failed: {str(e)}"}

    def render_sidebar(self):
        """Render sidebar with agent controls"""
        st.sidebar.title("🤖 Agent Controls")
        
        # Portfolio Overview
        st.sidebar.subheader("📊 Portfolio Overview")
        st.sidebar.metric("Total AUM", f"${self.portfolio_context['total_aum']:,.0f}")
        st.sidebar.metric("Asia Tech Allocation", f"{self.portfolio_context['regional_allocations']['asia_tech']:.1%}")
        
        # Market Data Settings
        st.sidebar.subheader("Market Data")
        symbols = st.sidebar.text_input("Stock Symbols", value="TSM,005930.KS,AAPL,NVDA", help="Comma-separated symbols")
        data_type = st.sidebar.selectbox("Data Type", ["quote", "overview", "earnings"])
        period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
        
        # Analysis Settings
        st.sidebar.subheader("Analysis Settings")
        confidence_threshold = st.sidebar.slider("Retrieval Confidence Threshold", 0.0, 1.0, 0.7)
        
        # Portfolio Settings
        st.sidebar.subheader("Portfolio Context")
        use_dynamic_context = st.sidebar.checkbox("Use Dynamic Context", value=True, help="Enable context-aware analysis")
        
        return {
            "symbols": symbols,
            "data_type": data_type,
            "period": period,
            "confidence_threshold": confidence_threshold,
            "use_dynamic_context": use_dynamic_context
        }

    def render_main_interface(self, settings: Dict[str, Any]):
        """Render main interface"""
        st.title("📈 Multi-Agent Finance Assistant")
        st.markdown("*Powered by CrewAI, FastAPI, and Streamlit*")
        
        # Create tabs for different interaction modes
        tab1, tab2, tab3 = st.tabs(["💬 Text Query", "🎤 Voice Query", "🔧 Agent Testing"])
        
        with tab1:
            self.render_text_query_tab(settings)
        
        with tab2:
            self.render_voice_query_tab()
        
        with tab3:
            self.render_agent_testing_tab(settings)

    def render_text_query_tab(self, settings: Dict[str, Any]):
        """Render text query interface"""
        st.header("Text-Based Financial Query")
        
        # Quick query buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Morning Brief"):
                st.session_state.query_input = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
                
        # Additional example queries
        st.markdown("**Example queries:**")
        st.markdown("- *How did TSMC and Samsung perform compared to expectations?*")
        st.markdown("- *What's driving volatility in our semiconductor positions?*")
        st.markdown("- *Should we rebalance our Asia tech allocation based on recent performance?*")
        
        # Text input
        query = st.text_area(
            "Enter your financial query:",
            value=st.session_state.get("query_input", ""),
            height=100,
            placeholder="e.g., What's our risk exposure in Asia tech stocks today?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            process_query = st.button("🚀 Process Query", type="primary")
        
        if process_query and query:
            self.process_text_query(query, settings)

    def process_text_query(self, query: str, settings: Dict[str, Any]):
        """Process text query through the agent pipeline with dynamic context"""
        with st.spinner("Processing your query through the agent pipeline..."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract query context for targeted retrieval and analysis
            status_text.text("Analyzing query context...")
            query_context = self._extract_query_context(query)
            progress_bar.progress(10)
            
            # Display detected context
            with st.expander("🔍 Detected Query Context", expanded=False):
                st.json(query_context)
            
            # Step 1: Fetch market data
            status_text.text("Step 1/4: Fetching real-time market data...")
            progress_bar.progress(25)
            
            market_payload = {
                "symbols": settings["symbols"],
                "data_type": settings["data_type"],
                "period": settings["period"],
                "context": query_context  # Add context to market data request
            }
            
            market_response = asyncio.run(self.call_api_endpoint("/agent/api", market_payload))
            
            # Step 2: Build dynamic portfolio data
            status_text.text("Step 2/4: Analyzing portfolio context...")
            progress_bar.progress(40)
            
            if settings.get("use_dynamic_context"):
                portfolio_data = self._build_dynamic_portfolio_data(query_context, market_response)
            else:
                portfolio_data = self.portfolio_context
            
            # Step 3: Retrieve relevant documents with enhanced context
            status_text.text("Step 3/4: Retrieving relevant documents...")
            progress_bar.progress(60)
            
            # Build comprehensive retrieval context
            retrieval_context = {
                "query_type": query_context["query_type"],
                "sectors": query_context["sectors"],
                "regions": query_context["regions"],
                "entities": query_context["entities"],
                "time_horizon": query_context["time_horizon"],
                "portfolio_focus": list(portfolio_data.get("focused_holdings", {}).keys()),
                "risk_factors": portfolio_data.get("risk_factors", []),
                "current_allocations": portfolio_data.get("regional_allocations", {}),
                "analysis_focus": query_context.get("analysis_focus", [])
            }
            
            retrieval_payload = {
                "query": query,
                "context": retrieval_context,
                "confidence_threshold": settings.get("confidence_threshold", 0.7),
                "max_documents": 10
            }
            
            retrieval_response = asyncio.run(self.call_api_endpoint("/agent/retrieve", retrieval_payload))
            
            # Step 4: Perform analysis with comprehensive portfolio data
            status_text.text("Step 4/4: Performing portfolio analysis...")
            progress_bar.progress(80)
            
            analysis_payload = {
                "portfolio_data": portfolio_data,
                "market_data": market_response.get("result", {}),
                "query": query,
                "query_context": query_context,
                "retrieved_documents": retrieval_response.get("result", []),
                "analysis_parameters": {
                    "include_risk_metrics": query_context["query_type"] == "risk_analysis",
                    "include_performance_attribution": query_context["query_type"] == "performance_analysis",
                    "include_earnings_analysis": query_context["query_type"] == "earnings_analysis",
                    "focus_sectors": query_context["sectors"],
                    "focus_regions": query_context["regions"]
                }
            }
            
            analysis_response = asyncio.run(self.call_api_endpoint("/agent/analyze", analysis_payload))
            
            # Step 5: Generate narrative with full context
            status_text.text("Step 5/5: Generating market brief...")
            progress_bar.progress(90)
            
            narrative_payload = {
                "query": query,
                "query_context": query_context,
                "market_data": market_response.get("result", {}),
                "portfolio_data": portfolio_data,
                "analysis_results": analysis_response,
                "context_documents": retrieval_response.get("result", []),
                "briefing_style": self._determine_briefing_style(query_context),
                "target_audience": "portfolio_manager",
                "include_voice_synthesis": False  # Set to True if voice output is needed
            }
            
            narrative_response = asyncio.run(self.call_api_endpoint("/agent/narrative", narrative_payload))
            
            # Complete progress
            progress_bar.progress(100)
            time.sleep(0.5)  # Brief pause for UX
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            self.display_query_results(query, {
                "query_context": query_context,
                "portfolio_data": portfolio_data,
                "market_data": market_response,
                "retrieval": retrieval_response,
                "analysis": analysis_response,
                "narrative": narrative_response
            })

    def _determine_briefing_style(self, query_context: Dict[str, Any]) -> str:
        """Determine appropriate briefing style based on query context"""
        if query_context["query_type"] == "market_brief":
            return "executive_summary"
        elif query_context["query_type"] == "risk_analysis":
            return "risk_focused"
        elif query_context["query_type"] == "earnings_analysis":
            return "earnings_focused"
        elif query_context["query_type"] == "performance_analysis":
            return "performance_focused"
        else:
            return "comprehensive"

    def render_voice_query_tab(self):
        """Render voice query interface"""
        st.header("Voice-Based Financial Query")
        
        st.info("💡 Upload an audio file or record your financial question")
        
        # Audio upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an audio file with your financial query"
        )
        
        # Audio recording (placeholder - requires additional setup)
        st.markdown("### 🎙️ Record Audio")
        audio_data, process_clicked = self.recorder.render_with_controls("voice_query")

        if process_clicked and audio_data:
            st.success("Processing voice input...")
            audio_bytes = AudioRecorder.process_audio_data(audio_data)
            
            if audio_bytes:
                # Convert to BytesIO for upload
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "query.webm"
                
                response = asyncio.run(self.upload_voice_query(audio_file))
                
                if "error" in response:
                    st.error(f"Error processing voice query: {response['error']}")
                else:
                    st.success("Voice query processed successfully!")
                    if "transcript" in response:
                        st.subheader("📝 Transcription")
                        st.write(response["transcript"])
                    if "response" in response:
                        st.subheader("🤖 Assistant Response")
                        st.write(response["response"])

                
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🎤 Process Voice Query", type="primary"):
                with st.spinner("Processing voice query..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    response = asyncio.run(self.upload_voice_query(uploaded_file))
                    
                    if "error" in response:
                        st.error(f"Error processing voice query: {response['error']}")
                    else:
                        st.success("Voice query processed successfully!")
                        
                        # Display transcription if available
                        if "transcript" in response:
                            st.subheader("📝 Transcription")
                            st.write(response["transcript"])
                        
                        # Display response
                        if "response" in response:
                            st.subheader("🤖 Assistant Response")
                            st.write(response["response"])

    def render_agent_testing_tab(self, settings: Dict[str, Any]):
        """Render individual agent testing interface"""
        st.header("Individual Agent Testing")
        
        agent_type = st.selectbox(
            "Select Agent to Test",
            ["API Agent", "Scraping Agent", "Retriever Agent", "Analysis Agent", "Language Agent", "Voice Agent (STT/TTS)"]
        )
        
        if agent_type == "API Agent":
            self.test_api_agent(settings)
        elif agent_type == "Scraping Agent":
            self.test_scraping_agent()
        elif agent_type == "Retriever Agent":
            self.test_retriever_agent()
        elif agent_type == "Analysis Agent":
            self.test_analysis_agent()
        elif agent_type == "Language Agent":
            self.test_language_agent()
        elif agent_type == "Voice Agent (STT/TTS)":
            self.test_voice_agent()

    def test_api_agent(self, settings: Dict[str, Any]):
        """Test API agent functionality"""
        st.subheader("📊 API Agent Testing")
        
        if st.button("Fetch Market Data"):
            payload = {
                "symbols": settings["symbols"],
                "data_type": settings["data_type"],
                "period": settings["period"]
            }
            
            with st.spinner("Fetching market data..."):
                response = asyncio.run(self.call_api_endpoint("/agent/api", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Market data fetched successfully!")
                st.json(response)

    def test_scraping_agent(self):
        """Test scraping agent functionality"""
        st.subheader("🔍 Scraping Agent Testing")
        
        ticker = st.text_input("Ticker Symbol", value="TSM")
        filing_types = st.multiselect("Filing Types", ["10-K", "10-Q", "8-K"], default=["10-K", "10-Q"])
        limit = st.number_input("Limit", min_value=1, max_value=20, value=5)
        
        if st.button("Scrape SEC Filings"):
            payload = {
                "ticker": ticker,
                "filing_types": filing_types,
                "limit": limit
            }
            
            with st.spinner("Scraping SEC filings..."):
                response = asyncio.run(self.call_api_endpoint("/agent/scrape", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("SEC filings scraped successfully!")
                st.json(response)

    def test_retriever_agent(self):
        """Test retriever agent functionality with enhanced context"""
        st.subheader("🔎 Retriever Agent Testing")
        
        query = st.text_input("Search Query", value="Asia tech exposure risk analysis")
        
        # Enhanced context input
        st.markdown("**Context Parameters:**")
        col1, col2 = st.columns(2)
        
        with col1:
            sectors = st.multiselect("Focus Sectors", ["tech", "financials", "healthcare"], default=["tech"])
            regions = st.multiselect("Focus Regions", ["asia", "us", "europe"], default=["asia"])
        
        with col2:
            query_type = st.selectbox("Query Type", ["risk_analysis", "earnings_analysis", "performance_analysis", "market_brief"])
            entities = st.text_input("Entities (comma-separated)", value="TSM,005930.KS")
        
        if st.button("Retrieve Documents"):
            context = {
                "query_type": query_type,
                "sectors": sectors,
                "regions": regions,
                "entities": entities.split(",") if entities else [],
                "time_horizon": "current"
            }
            
            payload = {
                "query": query,
                "context": context,
                "confidence_threshold": 0.7,
                "max_documents": 10
            }
            
            with st.spinner("Retrieving documents..."):
                response = asyncio.run(self.call_api_endpoint("/agent/retrieve", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Documents retrieved successfully!")
                st.json(response)

    def test_analysis_agent(self):
        """Test analysis agent functionality with realistic portfolio data"""
        st.subheader("📈 Analysis Agent Testing")
        
        # Use realistic portfolio data
        sample_query_context = {
            "query_type": "risk_analysis",
            "sectors": ["tech"],
            "regions": ["asia"],
            "entities": ["TSM", "005930.KS"]
        }
        
        portfolio_data = self._build_dynamic_portfolio_data(sample_query_context, {"result": {}})
        
        if st.button("Run Portfolio Analysis"):
            payload = {
                "portfolio_data": portfolio_data,
                "market_data": {"TSM": {"price": 100.50, "change_percent": 2.1}, "005930.KS": {"price": 70500, "change_percent": -1.3}},
                "query": "What's our risk exposure in Asia tech stocks?",
                "query_context": sample_query_context,
                "analysis_parameters": {
                    "include_risk_metrics": True,
                    "include_performance_attribution": True,
                    "focus_sectors": ["tech"],
                    "focus_regions": ["asia"]
                }
            }
            
            with st.spinner("Running portfolio analysis..."):
                response = asyncio.run(self.call_api_endpoint("/agent/analyze", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Portfolio analysis completed!")
                st.json(response)

    def test_language_agent(self):
        """Test language agent functionality"""
        st.subheader("📝 Language Agent Testing")
        
        if st.button("Generate Market Brief"):
            sample_context = self._extract_query_context("What happened in Asia tech today?")
            portfolio_data = self._build_dynamic_portfolio_data(sample_context, {"result": {}})
            
            payload = {
                "query": "What happened in Asia tech today?",
                "query_context": sample_context,
                "market_data": {"TSM": {"price": 100, "change": 0.05}},
                "analysis_results": {"risk_level": "medium"},
                "context_documents": ["TSMC earnings beat expectations"]
            }
            
            with st.spinner("Generating market brief..."):
                response = asyncio.run(self.call_api_endpoint("/agent/narrative", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Market brief generated!")
                st.markdown("### Generated Brief:")
                st.write(response.get("brief", "No brief generated"))

    async def test_voice_agent(self):
        """Test voice agent functionality"""
        st.subheader("🎤 Voice Agent Testing")
        
        # TTS Test
        st.markdown("#### Text-to-Speech Test")
        tts_text = st.text_area("Text to synthesize", value="Hello, this is a test of the text-to-speech system.")
        
        if st.button("Generate Speech"):
            payload = {"text": tts_text}
            
            with st.spinner("Generating speech..."):
                response = asyncio.run(self.call_api_endpoint("/agent/tts", payload))
                
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Speech generated successfully!")
                st.json(response)
        
        # STT Test
        st.markdown("#### Speech-to-Text Test")
        stt_file = st.file_uploader("Upload audio for transcription", type=['wav', 'mp3'])
        
        if stt_file and st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                files = {"file": (stt_file.name, stt_file.read(), "audio/wav")}
                try:
                    async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL, timeout=30.0) as client:
                        response = await client.post("/agent/stt", files=files)
                        result = response.json()
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Audio transcribed successfully!")
                        st.write(f"**Transcript:** {result.get('transcript', 'No transcript')}")
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")

    def display_query_results(self, query: str, results: Dict[str, Any]):
        """Display comprehensive query results"""
        st.subheader("📋 Query Results")
        
        # Display original query
        st.markdown(f"**Query:** {query}")
        st.markdown("---")
        
        # Display final agent response
        final_response = results.get("narrative", {}).get("response") or results.get("narrative", {}).get("brief")

        if final_response:
            st.markdown("### 🎯 Final Answer")
            st.markdown(final_response)
        else:
            st.info("No final answer received from agents.")
        
        st.markdown("---")

        # Expandable sections for debug or transparency
        with st.expander("📊 Market Data Details"):
            if "error" not in results.get("market_data", {}):
                st.json(results.get("market_data", {}))
            else:
                st.error(results["market_data"]["error"])
        
        with st.expander("🔍 Retrieved Documents"):
            if "error" not in results.get("retrieval", {}):
                st.json(results.get("retrieval", {}))
            else:
                st.error(results["retrieval"]["error"])
        
        with st.expander("📈 Analysis Results"):
            if "error" not in results.get("analysis", {}):
                st.json(results.get("analysis", {}))
            else:
                st.error(results["analysis"]["error"])
        
        meta = results.get("narrative", {})
        if meta.get("timestamp"):
            st.caption(f"🕒 Generated on: {meta['timestamp']}")
        if meta.get("query_type"):
            st.caption(f"🔎 Query type: `{meta['query_type']}`")




def main():
    """Main application entry point"""
    # Initialize session state
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    
    # Create UI instance
    ui = FinanceAssistantUI()
    
    # Render sidebar
    settings = ui.render_sidebar()
    
    # Render main interface
    ui.render_main_interface(settings)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>Multi-Agent Finance Assistant</strong> | Built with CrewAI, FastAPI & Streamlit</p>
            <p><em>Real-time market data • SEC filings • Voice interaction • AI-powered analysis</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()