# Finance Assistant

An AI-powered financial analysis platform that empowers portfolio managers with real-time market insights, portfolio risk assessment, and voice-based interaction capabilities.

## 🚀 Overview

Finance Assistant leverages large language models (LLMs), retrieval-augmented generation (RAG), and a modular agent architecture to deliver professional market briefs, risk summaries, and actionable recommendations. Built with **CrewAI**, **FastAPI**, and **Streamlit**, the platform integrates data from SEC EDGAR, Yahoo Finance, and other financial sources with scalable deployment via Docker.

### Key Features

- 📊 **Real-time Market Data**: Fetches quotes and news with focus on Asia-Pacific tech stocks
- 🔍 **Portfolio Analysis**: Computes risk metrics (VaR, beta, volatility), sector allocation, and performance attribution
- 📑 **Document Retrieval**: Uses RAG to provide context from SEC filings, financial news, and market data
- 🎤 **Voice Interaction**: Supports speech-to-text (STT) and text-to-speech (TTS) for hands-free operation
- 🖥️ **Web Interface**: Streamlit-based UI for portfolio visualization, voice queries, and analysis
- 🐳 **Scalable Deployment**: Containerized with Docker and served through FastAPI endpoints

## 🏗️ Architecture

The platform uses a modular architecture with specialized agents orchestrated via CrewAI:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   FastAPI       │    │   Orchestrator  │
│   (Streamlit)   │◄──►│   Services      │◄──►│   (CrewAI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   Vector Store  │    │   Agent Network │
                    │   (FAISS/Pine)  │    │   (6 Agents)    │
                    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
finance-assistant/
├── agents/                   # Specialized AI agents
│   ├── api_agent.py         # Market data fetching
│   ├── scraping_agent.py    # SEC filings & news
│   ├── retriever_agent.py   # Vector search & RAG
│   ├── analysis_agent.py    # Portfolio analysis
│   ├── language_agent.py    # LLM synthesis
│   └── voice_agent.py       # STT/TTS pipeline
├── data_ingestion/          # Data collection & processing
│   ├── market_data.py       # API connectors
│   ├── document_loaders.py  # SEC, news scrapers
│   └── embeddings.py        # Vector indexing
├── orchestrator/            # Agent coordination
│   ├── crew_manager.py      # CrewAI orchestration
│   ├── task_router.py       # Request routing
│   └── workflow.py          # Agent coordination
├── services/                # Core services
│   ├── api_service.py       # FastAPI endpoints
│   ├── vector_store.py      # FAISS/Pinecone
│   └── voice_service.py     # STT/TTS handlers
├── streamlit_app/           # Frontend components
│   ├── components/
│   │   ├── audio_recorder.py
│   │   └── market_dashboard.py
│   └── static/
├── docker/                  # Containerization
│   ├── Dockerfile
│   └── docker-compose.yaml
├── main.py                  # Streamlit UI entry point
├── requirements.txt
└── .env.example
```

## 🤖 Agent Architecture

### Core Agents

| Agent | Purpose | Key Capabilities |
|-------|---------|-----------------|
| **API Agent** | Market data fetching | Real-time quotes, historical data, Asia-Pacific focus |
| **Scraping Agent** | Document extraction | SEC filings (10-K, 10-Q, 8-K), earnings reports |
| **Retriever Agent** | RAG operations | Semantic search, hybrid retrieval, context management |
| **Analysis Agent** | Portfolio analysis | Risk metrics, sector analysis, performance attribution |
| **Language Agent** | LLM synthesis | Financial narratives, market sentiment analysis |
| **Voice Agent** | Voice interaction | STT/TTS processing, audio handling |

### Orchestration Layer

- **Task Router**: Routes requests to appropriate agent workflows
- **Crew Manager**: Orchestrates multi-agent workflows using CrewAI
- **Workflow Engine**: Manages complex financial analysis pipelines

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional)
- API keys for external services

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/finance-assistant.git
   cd finance-assistant
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   
   **Option A: Docker (Recommended)**
   ```bash
   docker compose up -d
   ```
   
   **Option B: Local Development**
   ```bash
   streamlit run main.py
   ```

5. **Access the application**
   - Streamlit UI: http://localhost:8501
   - FastAPI endpoints: http://localhost:8000
   - API documentation: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

# Vector Store Configuration
VECTOR_STORE_BACKEND=faiss  # or pinecone
VECTOR_STORE_PATH=./vector_store
EMBEDDING_BACKEND=sentence_transformers  # or openai, huggingface

# Voice Service
TTS_ENGINE=pyttsx3  # or google
STT_MODEL=base  # whisper model size
```

### Supported Backends

- **Vector Store**: FAISS (local) or Pinecone (cloud)
- **Embeddings**: Sentence Transformers, OpenAI, Hugging Face
- **Voice**: OpenAI Whisper (STT), pyttsx3/Google TTS

## 📖 Usage Examples

### Morning Brief Query

```python
from orchestrator.task_router import create_task_router

router = create_task_router()
result = router.route({
    "input_type": "text",
    "query": "What's our risk exposure in Asia tech stocks today?"
})
print(result["response"])
```

### Voice Query Processing

```python
from services.voice_service import get_voice_service

voice_service = get_voice_service()
with open("query.wav", "rb") as f:
    result = await voice_service.transcribe_upload(f)
print(result["text"])
```

### Portfolio Risk Analysis

```python
from agents.analysis_agent import AnalysisAgent

agent = AnalysisAgent()
portfolio_data = {
    "holdings": [
        {"symbol": "TSM", "sector": "Technology", "weight": 0.08}
    ]
}
result = agent.analyze_portfolio(portfolio_data, market_data)
print(result["portfolio_metrics"])
```

### API Endpoints

#### Morning Brief Workflow
```bash
curl -X POST "http://localhost:8000/agent/workflow/morning-brief" \
     -H "Content-Type: application/json" \
     -d '{"query": "What'\''s our risk exposure in Asia tech stocks today?"}'
```

#### Voice Processing
```bash
curl -X POST "http://localhost:8000/voice/transcribe" \
     -F "audio=@query.wav"
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

## 🎯 Supported Query Types

- **Risk Analysis**: "What's our risk exposure in Asia tech stocks?"
- **Market Data**: "What's the price of TSMC?"
- **Earnings**: "Latest earnings for Samsung"
- **Portfolio**: "Show me our technology sector allocation"
- **Voice Queries**: Upload audio or use browser recording

## 🔍 Key Components

### Data Sources
- **Yahoo Finance**: Real-time quotes, historical data, news
- **Alpha Vantage**: Market data and financial indicators
- **SEC EDGAR**: Official company filings (10-K, 10-Q, 8-K)
- **Financial News**: MarketWatch, Reuters, earnings reports

### Analysis Capabilities
- **Risk Metrics**: VaR, beta, volatility, maximum drawdown
- **Sector Analysis**: Allocation, concentration risk
- **Performance**: Benchmark comparison, attribution analysis
- **Sentiment**: Market sentiment from news and indicators

### Voice Features
- **Speech Recognition**: OpenAI Whisper with multiple model sizes
- **Text-to-Speech**: Local (pyttsx3) or cloud (Google TTS)
- **Audio Processing**: Format conversion, quality validation
- **Browser Recording**: Web Audio API integration

## 🧪 Testing

The platform includes comprehensive agent testing through the Streamlit interface:

1. Navigate to the "Agent Testing" tab
2. Select an agent to test
3. Provide test inputs
4. View detailed results and performance metrics

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add feature X'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

### Development Guidelines

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions, issues, or support:

- 📧 Create an issue on GitHub
- 📖 Check the [documentation](docs/)
- 💬 Join our community discussions

## 🙏 Acknowledgments

- **CrewAI** for agent orchestration framework
- **Streamlit** for the interactive web interface
- **FastAPI** for high-performance API endpoints
- **OpenAI** for language models and voice processing
- **Yahoo Finance** and **Alpha Vantage** for market data

---

**Finance Assistant** - Empowering portfolio managers with AI-driven insights and voice-enabled financial analysis.s