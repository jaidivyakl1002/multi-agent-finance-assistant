Finance Assistant
Finance Assistant is an AI-powered financial analysis platform designed to empower portfolio managers with real-time market insights, portfolio risk assessment, and voice-based interaction. Leveraging large language models (LLMs), retrieval-augmented generation (RAG), and a modular agent architecture, it delivers professional market briefs, risk summaries, and actionable recommendations. Built with CrewAI, FastAPI, and Streamlit, the platform integrates data from SEC EDGAR, Yahoo Finance, and other sources, with scalable deployment via Docker.
Table of Contents

Overview
Project Structure
Features
Agents
API Agent
Scraping Agent
Retriever Agent
Analysis Agent
Language Agent
Voice Agent


Orchestration
Task Router
Crew Manager
Workflow


Services
API Service
Vector Store Service
Voice Service


Data Ingestion
Frontend
Setup Instructions
Usage
Contributing
License

Overview
The Finance Assistant is a sophisticated tool for portfolio management, offering:

Real-time Market Data: Fetches quotes and news, with a focus on Asia-Pacific tech stocks.
Portfolio Analysis: Computes risk metrics (VaR, beta, volatility), sector allocation, and performance attribution.
Document Retrieval: Uses RAG to provide context from SEC filings, financial news, and market data.
Voice Interaction: Supports speech-to-text (STT) and text-to-speech (TTS) for hands-free operation.
Web Interface: A Streamlit-based UI for portfolio visualization, voice queries, and analysis.

The project uses a modular architecture with specialized agents orchestrated via CrewAI, served through FastAPI endpoints, and deployed with Docker. No caching service is implemented to simplify dependency management.
Project Structure
finance-assistant/
├── agents/
│   ├── __init__.py
│   ├── api_agent.py          # Market data fetching
│   ├── scraping_agent.py     # SEC filings & news
│   ├── retriever_agent.py    # Vector search & RAG
│   ├── analysis_agent.py     # Portfolio analysis
│   ├── language_agent.py     # LLM synthesis
│   └── voice_agent.py        # STT/TTS pipeline
├── data_ingestion/
│   ├── __init__.py
│   ├── market_data.py        # API connectors
│   ├── document_loaders.py   # SEC, news scrapers
│   └── embeddings.py         # Vector indexing
├── orchestrator/
│   ├── __init__.py
│   ├── crew_manager.py       # CrewAI orchestration
│   ├── task_router.py        # Request routing
│   └── workflow.py           # Agent coordination
├── services/
│   ├── __init__.py
│   ├── api_service.py        # FastAPI endpoints
│   ├── vector_store.py       # FAISS/Pinecone
│   └── voice_service.py      # STT/TTS handlers
├── streamlit_app/
│   ├── components/           # UI components
│   │   ├── audio_recorder.py
│   │   └── market_dashboard.py
│   └── static/               # Assets
├── main.py                   # Streamlit UI
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── docs/
│   └── ai_tool_usage.md
├── requirements.txt
├── .env.example
└── README.md

Features

Market Data Integration: Real-time quotes and news for stocks like TSMC, Samsung, and Nvidia.
Portfolio Risk Analysis: Calculates volatility, VaR, beta, and sector concentration risks.
Financial Document Scraping: Extracts SEC filings (10-K, 10-Q, 8-K) and earnings reports.
Voice-Based Interaction: Processes spoken queries and delivers spoken responses.
RAG-Enhanced Responses: Combines LLM-generated narratives with retrieved financial context.
Scalable Deployment: Containerized with Docker and served via FastAPI.
Interactive UI: Streamlit interface with portfolio dashboards, voice input, and visualizations.

Agents
The platform's core functionality is powered by six specialized agents, each handling a specific aspect of financial analysis.
API Agent
File: agents/api_agent.py
Purpose: Fetches and analyzes real-time and historical market data, specializing in Asia-Pacific tech stocks.
Key Components:

MarketDataTool: Retrieves quotes, historical data, and news from Yahoo Finance and Alpha Vantage.
AsiaFocusedMarketTool: Analyzes Asia-Pacific tech stocks (e.g., TSMC, Samsung, SK Hynix) with sentiment analysis.
MarketDataCrew: Orchestrates tasks like morning briefs, Asia tech overviews, and earnings analysis.

Capabilities:

Fetches real-time quotes with price changes, volume, and market cap.
Analyzes historical data for volatility and trends.
Provides regional sentiment based on price movements.
Supports queries like "What's our risk exposure in Asia tech stocks today?"

Example:
from agents.api_agent import get_market_data_crew
crew = get_market_data_crew()
result = crew.process_morning_brief_query("What's our risk exposure in Asia tech stocks today?")
print(result["analysis"])

Scraping Agent
File: agents/scraping_agent.py
Purpose: Extracts financial documents and news, including SEC filings and earnings reports.
Key Components:

SECFilingsTool: Scrapes 10-K, 10-Q, and 8-K filings from SEC EDGAR.
FinancialNewsTool: Scrapes earnings reports, analyst ratings, and news from Yahoo Finance, MarketWatch, and Reuters.

Capabilities:

Retrieves recent SEC filings for specified tickers.
Extracts earnings data and analyst recommendations.
Supports asynchronous scraping for multiple tickers.
Combines insights from filings and news for comprehensive analysis.

Example:
from agents.scraping_agent import ScrapingAgent
scraper = ScrapingAgent()
result = scraper.scrape_filings(ticker="AAPL", filing_types=["10-Q"], limit=2)
print(result["filings"])

Retriever Agent
File: agents/retriever_agent.py
Purpose: Manages document retrieval for RAG operations using a vector store.
Key Components:

DocumentSearchTool: Performs semantic similarity searches.
ContextRetrievalTool: Retrieves context-specific documents for financial, earnings, or risk queries.
HybridSearchTool: Combines semantic and keyword searches.

Capabilities:

Supports multi-strategy searches (semantic, hybrid, contextual).
Updates vector store with new documents.
Provides index statistics for monitoring.

Example:
from agents.retriever_agent import RetrieverAgent
from services.vector_store import VectorStoreService
retriever = RetrieverAgent(vector_service=VectorStoreService())
context = retriever.get_rag_context("Asia tech earnings", context_type="earnings")
print(context["context_documents"])

Analysis Agent
File: agents/analysis_agent.py
Purpose: Conducts portfolio risk assessment and performance analysis.
Key Components:

RiskAnalysisTool: Calculates risk metrics (volatility, VaR, beta, max drawdown).
SectorAnalysisTool: Analyzes sector allocation and concentration risk.
PerformanceAnalysisTool: Evaluates performance against benchmarks (e.g., SPY).

Capabilities:

Computes comprehensive portfolio metrics.
Identifies concentration risks in sectors like technology.
Generates actionable recommendations for rebalancing.
Handles missing data with reasonable defaults.

Example:
from agents.analysis_agent import AnalysisAgent
agent = AnalysisAgent()
result = agent.analyze_portfolio(
    portfolio_data={"holdings": [{"symbol": "TSM", "sector": "Technology", "weight": 0.08}]},
    market_data={"returns": {"TSM": 0.03}}
)
print(result["portfolio_metrics"])

Language Agent
File: agents/language_agent.py
Purpose: Synthesizes natural language financial narratives using LLM and RAG.
Key Components:

FinancialNarrativeTool: Generates professional market briefs.
MarketSentimentTool: Analyzes sentiment from news and market indicators.

Capabilities:

Creates concise, professional market briefs.
Analyzes sentiment based on keywords and indicators.
Formats responses for voice delivery.
Integrates RAG context for enhanced accuracy.

Example:
from agents.language_agent import LanguageAgent
agent = LanguageAgent()
result = agent.synthesize_market_brief(
    query="What's our risk exposure in Asia tech stocks?",
    market_data={"portfolio_allocation": {"asia_tech": 22}}
)
print(result)

Voice Agent
File: agents/voice_agent.py
Purpose: Handles voice-based interactions using STT and TTS.
Key Components:

SpeechToTextTool: Converts audio to text using OpenAI Whisper.
TextToSpeechTool: Generates speech using pyttsx3 or Google TTS.

Capabilities:

Transcribes spoken queries with silence detection.
Generates clear, natural speech output.
Supports saving TTS output to files.
Integrates with other agents for voice-driven analysis.

Example:
from agents.voice_agent import VoiceAgent
voice_agent = VoiceAgent()
transcribed = voice_agent.process_voice_input()  # Records from microphone
voice_agent.generate_voice_output("Market brief ready.")

Orchestration
The orchestration layer, implemented in the orchestrator directory, coordinates agent interactions, routes tasks, and manages workflows using CrewAI. It ensures seamless integration of data fetching, analysis, and response generation, with robust async/sync handling and error recovery.
Task Router
File: orchestrator/task_router.py
Purpose: Routes incoming requests (text or voice) to appropriate agent workflows based on query type.
Key Components:

TaskRouter Class: Main entry point for routing requests, initialized with a configuration dictionary.
Input Handling:
Voice Input: Processes audio using the voice service for STT, then routes the transcript as a text query. Handles async operations with event loop management and a 30-second timeout.
Text Input: Categorizes queries into morning briefs, market data, general financial, or general queries based on keywords (e.g., "risk exposure", "price", "earnings").


Query Routing:
Morning brief queries (e.g., "What's our risk exposure in Asia tech stocks?") are routed to the crew manager's voice query processing.
Market data queries (e.g., "What's the price of TSMC?") fetch real-time data.
General financial queries (e.g., "Latest earnings for Samsung") use RAG context.
General queries provide guidance on supported query types.


Health Check: Monitors the status of the crew manager and voice service, reporting "healthy", "degraded", or "limited" states.
Supported Queries: Lists supported query types with examples for user guidance.

Capabilities:

Robust async/sync handling for voice and text inputs.
Keyword-based query classification for accurate routing.
Error handling with fallback responses (e.g., processing transcript as text if voice fails).
Logging for debugging and monitoring.

Example:
from orchestrator.task_router import create_task_router
router = create_task_router()
result = router.route({
    "input_type": "text",
    "query": "What's our risk exposure in Asia tech stocks today?"
})
print(result["response"])

Crew Manager
File: orchestrator/crew_manager.py
Purpose: Orchestrates agent workflows using CrewAI, managing tasks and dependencies for complex queries like morning briefs.
Key Components:

EnhancedAgentCrewManager Class: Initializes agents and core services (e.g., vector store), with fallback mechanisms for error recovery.
Agent Initialization:
Initializes all agents (api_agent, scraping_agent, retriever_agent, analysis_agent, language_agent, voice_agent).
Falls back to api_agent if vector store initialization fails.


Sample Data: Populates the vector store with sample financial data (e.g., TSMC and Samsung earnings) if empty, ensuring functionality for testing.
Morning Brief Workflow:
Creates a Crew with tasks for fetching market data, retrieving context, analyzing risk, and generating narratives.
Tasks are chained with dependencies (e.g., analysis depends on market data and retrieval).
Focuses on Asia tech stocks, including prices, earnings, risk metrics, and recommendations.


Context Search: Performs RAG-based searches using the vector store for general queries.
Health Check: Reports agent initialization status and vector store statistics.

Capabilities:

Orchestrates multi-agent workflows with task dependencies.
Ensures data availability with sample financial documents.
Supports direct document searches without CrewAI orchestration.
Generates professional morning briefs with earnings surprises and risk metrics.

Example:
from orchestrator.crew_manager import create_enhanced_crew_manager
manager = create_enhanced_crew_manager()
result = manager.process_voice_query("What's our risk exposure in Asia tech stocks today?")
print(result["response"])

Workflow
File: orchestrator/workflow.py
Purpose: Defines high-level workflows for specific use cases, such as the morning brief.
Key Components:

handle_morning_brief Function: Processes morning brief queries by delegating to the crew manager's voice query processing.
Crew Manager Integration: Uses a shared EnhancedAgentCrewManager instance for consistency.

Capabilities:

Simplifies workflow execution for common use cases.
Provides a clear interface for integrating with the UI or API.
Handles errors with descriptive responses.

Example:
from orchestrator.workflow import handle_morning_brief
result = handle_morning_brief("What's our risk exposure in Asia tech stocks today?")
print(result["response"])

Services
The services layer, implemented in the services directory, provides core functionality for API interactions, vector storage, and voice processing. The platform does not use a caching service to reduce dependencies.
API Service
File: services/api_service.py
Purpose: Exposes FastAPI endpoints for agent interactions, voice processing, and orchestrated workflows.
Key Components:

APIRouter: Defines endpoints for voice transcription, synthesis, agent calls, and workflows.
Endpoints:
Voice Processing:
/voice/transcribe: Transcribes uploaded audio files using the voice service.
/voice/synthesize: Generates speech from text using local (pyttsx3) or Google TTS.
/voice/query: Processes voice queries by transcribing and routing to the task router.
/agent/stt and /agent/tts: Specialized STT/TTS endpoints for agent integration.


Agent-Specific:
/agent/api/asia-tech: Fetches Asia-Pacific tech market analysis using the API agent.
/agent/retrieve: Performs multi-strategy document retrieval (semantic, hybrid, contextual).
/agent/scrape: Scrapes SEC filings for specified tickers.
/agent/analyze: Analyzes portfolio risk and performance.
/agent/narrative: Generates financial narratives.
/agent/api (legacy): Supports older market data queries.


Workflow:
/agent/workflow/morning-brief: Runs the morning brief workflow, combining market data, risk analysis, and narrative generation, with optional TTS output.
/agent/orchestrated: Executes general orchestrated workflows via the crew manager.


Health:
/health: Checks the health of the task router and underlying services.




Integration: Uses the task router, crew manager, market data provider, and voice service for seamless operation.

Capabilities:

Handles asynchronous voice and text queries.
Supports multi-strategy document retrieval for RAG.
Provides specialized endpoints for Asia tech analysis and portfolio management.
Includes error handling and logging for robustness.
Returns JSON responses with timestamps and success indicators.

Example:
curl -X POST "http://localhost:8000/agent/workflow/morning-brief" \
     -H "Content-Type: application/json" \
     -d '{"query": "What'\''s our risk exposure in Asia tech stocks today?"}'

Vector Store Service
File: services/vector_store.py
Purpose: Manages document storage and retrieval for RAG using FAISS or Pinecone backends.
Key Components:

VectorStoreService Class: Main interface for vector store operations, supporting FAISS (default) or Pinecone.
FAISSVectorStore:
Uses FAISS for local vector storage with inner product similarity.
Stores documents, metadata, and embeddings in a file-based index (./vector_store).
Supports adding documents and similarity searches.


PineconeVectorStore:
Uses Pinecone for cloud-based vector storage with cosine similarity.
Requires a PINECONE_API_KEY environment variable.
Supports adding documents, similarity searches, and filtering.


EmbeddingService: Generates embeddings for documents and queries (implemented in data_ingestion/embeddings.py).
Search Methods:
Similarity Search: Finds documents based on semantic similarity.
Keyword Search: Fallback search using query keywords.
Hybrid Search: Combines semantic and keyword searches for improved results.


Utilities:
Generates unique document IDs using MD5 hashes.
Provides statistics (e.g., total documents, embedding dimension).
Supports batch document addition and index persistence.



Capabilities:

Flexible backend support (FAISS for local, Pinecone for cloud).
Efficient similarity and hybrid searches for RAG.
Persistent storage with automatic index loading/saving.
Error handling and logging for reliability.

Example:
from services.vector_store import VectorStoreService
vector_store = VectorStoreService(backend="faiss")
documents = [{"content": "TSMC Q3 earnings beat estimates", "metadata": {"ticker": "TSM"}}]
result = vector_store.add_documents(documents)
print(result["document_ids"])

Voice Service
File: services/voice_service.py
Purpose: Handles speech-to-text (STT) and text-to-speech (TTS) for voice-based interactions.
Key Components:

VoiceService Class: Unified interface for STT and TTS operations.
STTService:
Uses OpenAI Whisper (base model by default) for audio transcription.
Supports async transcription of uploaded files or raw bytes.
Validates audio quality (duration, RMS level) before processing.
Caches transcriptions in Redis (or in-memory if Redis is unavailable) with a 1-hour TTL.


TTSService:
Supports local (pyttsx3) and Google TTS engines.
Configures voice properties (rate, volume, female voice preference).
Caches generated audio in Redis (or in-memory) with a 1-hour TTL.
Converts outputs to WAV format for consistency.


AudioProcessor: Converts audio formats (MP3, M4A, OGG, FLAC) to WAV and validates quality.
Configuration:
Sample rate: 16kHz, single channel.
Supported formats: WAV, MP3, M4A, OGG, FLAC.
Max recording duration: 30 seconds.


Health Check: Monitors STT/TTS service status and cache availability.

Capabilities:

Accurate transcription with language auto-detection.
Natural speech synthesis with multiple engine options.
Async support for FastAPI integration.
Robust audio validation and format conversion.
Temporary file management for processing.

Example:
from services.voice_service import get_voice_service
voice_service = get_voice_service()
with open("query.wav", "rb") as f:
    result = await voice_service.transcribe_upload(f)
print(result["text"])

Data Ingestion
The data ingestion layer, implemented in the data_ingestion directory, handles the collection, processing, and indexing of financial data for use in RAG and analysis.
Market Data
File: data_ingestion/market_data.py
Purpose: Connects to external APIs to fetch real-time and historical market data.
Capabilities:

Integrates with Yahoo Finance and Alpha Vantage for quotes, historical prices, and news.
Supports batch data retrieval for multiple tickers.
Handles rate limiting and API errors with retries.

Document Loaders
File: data_ingestion/document_loaders.py
Purpose: Scrapes and processes financial documents and news articles.
Capabilities:

Scrapes SEC EDGAR filings (10-K, 10-Q, 8-K) and earnings reports.
Extracts text from financial news sources (e.g., Yahoo Finance, Reuters).
Normalizes document formats for vector store indexing.

Embeddings
File: data_ingestion/embeddings.py
Purpose: Generates embeddings for documents and queries to enable semantic search in the vector store.
Key Components:

EmbeddingService Class: Main interface for embedding generation with multiple backends.
Backends:
Sentence Transformers: Uses models like all-MiniLM-L6-v2 for local, efficient embeddings (384 dimensions by default).
OpenAI: Supports models like text-embedding-ada-002 for cloud-based embeddings (1536 dimensions).
Hugging Face: Uses transformer models with mean pooling for embeddings (384 dimensions for all-MiniLM-L6-v2).


EmbeddingCache: File-based cache for embeddings with configurable TTL (24 hours by default), stored in data/embedding_cache.
Utilities:
Computes cosine similarity and Euclidean distance between embeddings.
Finds top-k similar embeddings for a query.


Features:
Supports batch embedding for efficient processing.
Async embedding methods for integration with FastAPI.
Cache statistics and management (e.g., clear cache, precompute embeddings).
Fallback to zero embeddings for empty texts.



Capabilities:

Flexible backend selection for embedding generation.
Efficient caching to reduce redundant computations.
Scalable batch processing with configurable batch sizes.
Robust error handling and logging.

Example:
from data_ingestion.embeddings import get_embedding_service
embedder = get_embedding_service(backend="sentence_transformers")
embedding = embedder.embed_text("TSMC Q3 earnings beat estimates")
print(len(embedding))  # 384 for MiniLM
embeddings = embedder.embed_batch(["Text 1", "Text 2"])
print(len(embeddings))  # 2

Frontend
The frontend, implemented in the streamlit_app directory and main.py, provides an interactive Streamlit-based UI for portfolio management, voice-based queries, and data visualization.
File: main.py
Purpose: Serves as the main entry point for the Streamlit application, orchestrating the UI and backend interactions.
Key Components:

FinanceAssistantUI Class: Manages the UI, HTTP client, audio recorder, and market dashboard.
Portfolio Context: Initializes a realistic portfolio with $50M AUM, regional allocations (e.g., 18% Asia tech), and key holdings (e.g., TSM, Samsung, AAPL, NVDA).
Query Context Extraction: Analyzes queries to detect type (e.g., risk, earnings), sectors, regions, and entities using keyword matching and regex.
Dynamic Portfolio Data: Adjusts allocations based on market data and query context.
API Integration: Makes async calls to FastAPI endpoints for market data, retrieval, analysis, and narrative generation.
Interface:
Sidebar: Displays portfolio metrics (AUM, allocations), market data settings (symbols, data type, period), and analysis parameters (confidence threshold, dynamic context).
Tabs: Text query, voice query, and agent testing interfaces.
Text Query: Supports queries like "What's our risk exposure in Asia tech stocks?" with a multi-step pipeline (context extraction, market data, retrieval, analysis, narrative).
Voice Query: Allows audio uploads or browser-based recording with transcription and response display.
Agent Testing: Tests individual agents (API, scraping, retriever, analysis, language, voice) with customizable inputs.


Result Display: Shows final answers, market data, retrieved documents, and analysis results in expandable sections.

Capabilities:

Interactive portfolio management with real-time updates.
Voice-based interaction with transcription and synthesis.
Detailed query processing with context-aware analysis.
Comprehensive agent testing for debugging and development.
Robust error handling and logging.

Example:
# Run the Streamlit app
streamlit run main.py
# Access at http://localhost:8501

File: streamlit_app/components/audio_recorder.py
Purpose: Provides a browser-based audio recording component for voice queries.
Key Components:

AudioRecorder Class: Renders an HTML/JavaScript component using Web Audio API for microphone capture.
Features:
Start/stop recording, play, download, and clear controls.
Supports WAV, MP3, OGG, and WebM formats.
Displays recording duration and metadata (size, format, source).


Processing: Converts audio to base64 for processing and creates WAV files for uploads.
Controls: Includes Streamlit buttons for processing and resetting recordings, with file upload as an alternative input method.

Capabilities:

Seamless browser-based audio capture.
Flexible audio format support.
Integration with voice service for transcription.
User-friendly interface with real-time feedback.

Example:
from streamlit_app.components.audio_recorder import AudioRecorder
recorder = AudioRecorder()
audio_data, process_clicked = recorder.render_with_controls("voice_query")
if process_clicked and audio_data:
    audio_bytes = AudioRecorder.process_audio_data(audio_data)

File: streamlit_app/components/market_dashboard.py
Purpose: Renders interactive visualizations for portfolio and market data using Plotly.
Key Components:

MarketDashboard Class: Generates charts for portfolio overview, market performance, risk metrics, earnings calendar, and market sentiment.
Visualizations:
Portfolio Overview: Pie chart for allocations, metrics for AUM, Asia tech allocation, daily P&L, and risk score.
Market Performance: Price charts (TSM, Samsung), volume, RSI, volatility analysis, and correlation matrix heatmap.
Risk Metrics: VaR gauge, drawdown chart, and risk factor breakdown bar chart.
Earnings Calendar: Bar chart for earnings surprises and table for upcoming earnings.
Market Sentiment: Fear & Greed Index gauge and VIX trend line chart.


Features:
Customizable colors and chart configurations.
Sample data generation for demonstration.
Responsive layouts with tabs and columns.



Capabilities:

Comprehensive financial data visualization.
Interactive charts with hover details and zooming.
Modular design for easy extension.
Fallback to sample data if real data is unavailable.

Example:
from streamlit_app.components.market_dashboard import MarketDashboard
dashboard = MarketDashboard()
dashboard.render_complete_dashboard({
    "portfolio": {"total_aum": 1500000, "allocations": {"Asia Tech": 22}},
    "market": {},
    "sentiment": {"fear_greed_index": 35},
    "earnings": []
})

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/finance-assistant.git
cd finance-assistant


Set Up Environment:

Copy .env.example to .env and configure API keys (e.g., Alpha Vantage, OpenAI, PINECONE_API_KEY for Pinecone, OPENAI_API_KEY for OpenAI embeddings) and vector store settings (e.g., VECTOR_STORE_PATH=./vector_store, VECTOR_STORE_BACKEND=faiss, EMBEDDING_BACKEND=sentence_transformers).

cp .env.example .env
nano .env


Install Dependencies:

Ensure Python 3.8+ is installed.

pip install -r requirements.txt


For Pinecone, install the client:

pip install pinecone-client


For Sentence Transformers or Hugging Face embeddings, install additional dependencies:

pip install sentence-transformers transformers torch


For OpenAI embeddings, install the client:

pip install openai


Run the Application:

Use Docker for production deployment.

docker compose up -d


Or run locally for development.

streamlit run main.py


Access the Application:

Streamlit UI: http://localhost:8501
FastAPI endpoints: http://localhost:8000
FastAPI documentation: http://localhost:8000/docs


Initialize Vector Store:

The vector store (FAISS by default) is initialized at ./vector_store with sample financial data if empty.
For Pinecone, ensure PINECONE_API_KEY and PINECONE_ENVIRONMENT are set in .env.



Usage
Below are example use cases for key functionalities.
Run a Morning Brief Workflow
curl -X POST "http://localhost:8000/agent/workflow/morning-brief" \
     -H "Content-Type: application/json" \
     -d '{"query": "What'\''s our risk exposure in Asia tech stocks today?"}'

Transcribe Audio
from services.voice_service import get_voice_service
voice_service = get_voice_service()
with open("query.wav", "rb") as f:
    result = await voice_service.transcribe_upload(f)
print(result["text"])

Analyze Portfolio Risk
from agents.analysis_agent import AnalysisAgent
agent = AnalysisAgent()
portfolio_data = {
    "holdings": [
        {"symbol": "TSM", "sector": "Technology", "weight": 0.08, "market_value": 80000}
    ]
}
market_data = {"returns": {"TSM": 0.03}}
result = agent.analyze_portfolio(portfolio_data, market_data)
print(result["recommendations"])

Check System Health
from orchestrator.task_router import create_task_router
router = create_task_router()
health = router.health_check()
print(health)

Use the Streamlit UI

Start the app:streamlit run main.py


Open http://localhost:8501 in a browser.
Enter a text query (e.g., "What's our risk exposure in Asia tech stocks?") or record a voice query.
View results, including market data, retrieved documents, and analysis.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add feature X').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure tests pass and follow the coding style in existing files.
License
MIT License (to be confirmed based on project requirements).
