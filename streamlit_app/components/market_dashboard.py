# streamlit_app/components/market_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

class MarketDashboard:
    """
    Advanced market dashboard component for financial data visualization
    """
    
    def __init__(self):
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }

    def render_portfolio_overview(self, portfolio_data: Dict[str, Any], key: str = "portfolio_overview"):
        """Render comprehensive portfolio overview"""
        st.subheader("💼 Portfolio Overview")
        
        if not portfolio_data:
            st.warning("No portfolio data available")
            return
            
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        total_aum = portfolio_data.get('total_aum', 1000000)
        asia_tech_allocation = portfolio_data.get('asia_tech_allocation', 0.22)
        daily_pnl = portfolio_data.get('daily_pnl', 15000)
        risk_score = portfolio_data.get('risk_score', 7.2)
        
        with col1:
            st.metric(
                "Total AUM",
                f"${total_aum:,.0f}",
                delta=f"{portfolio_data.get('aum_change', 0.025):.1%}",
                delta_color="normal"
            )
            
        with col2:
            st.metric(
                "Asia Tech Allocation", 
                f"{asia_tech_allocation:.1%}",
                delta=f"{portfolio_data.get('allocation_change', 0.04):.1%}",
                delta_color="normal"
            )
            
        with col3:
            st.metric(
                "Daily P&L",
                f"${daily_pnl:,.0f}",
                delta=f"{portfolio_data.get('pnl_change', 0.08):.1%}",
                delta_color="normal"
            )
            
        with col4:
            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/10",
                delta=f"{portfolio_data.get('risk_change', -0.3):.1f}",
                delta_color="inverse"
            )
        
        # Portfolio allocation pie chart
        self._render_portfolio_allocation(portfolio_data, key)

    def _render_portfolio_allocation(self, portfolio_data: Dict[str, Any], key: str):
        """Render portfolio allocation pie chart"""
        allocations = portfolio_data.get('allocations', {
            'Asia Tech': 22,
            'US Equities': 35,
            'European Stocks': 18,
            'Bonds': 15,
            'Cash': 10
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=list(allocations.keys()),
            values=list(allocations.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#FFFFFF', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<br>Value: $%{value}M<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': "Portfolio Allocation",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.colors['dark']}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=400,
            margin=dict(t=60, b=80, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_allocation")

    def render_market_performance(self, market_data: Dict[str, Any], key: str = "market_performance"):
        """Render market performance charts"""
        st.subheader("📈 Market Performance")
        
        if not market_data:
            st.warning("No market data available")
            return
            
        # Create sample data if not provided
        if 'time_series' not in market_data:
            market_data = self._generate_sample_market_data()
        
        # Market performance tabs
        tab1, tab2, tab3 = st.tabs(["📊 Price Charts", "📉 Volatility", "🔄 Correlation"])
        
        with tab1:
            self._render_price_charts(market_data, key)
        
        with tab2:
            self._render_volatility_analysis(market_data, key)
        
        with tab3:
            self._render_correlation_matrix(market_data, key)

    def _render_price_charts(self, market_data: Dict[str, Any], key: str):
        """Render price movement charts"""
        time_series = market_data.get('time_series', {})
        
        if not time_series:
            st.warning("No time series data available")
            return
        
        # Multi-asset price chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('TSM Price Movement', 'Samsung (005930.KS) Price', 
                          'Volume Analysis', 'RSI Indicator'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12
        )
        
        # TSM price data
        tsm_data = time_series.get('TSM', {})
        if tsm_data:
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            prices = tsm_data.get('prices', np.random.randn(30).cumsum() + 100)
            
            fig.add_trace(
                go.Scatter(x=dates, y=prices, name='TSM Price', 
                          line=dict(color=self.colors['primary'], width=3)),
                row=1, col=1
            )
            
            # Volume bars
            volumes = tsm_data.get('volumes', np.random.randint(1000000, 5000000, 30))
            fig.add_trace(
                go.Bar(x=dates, y=volumes, name='TSM Volume', 
                      marker_color=self.colors['info'], opacity=0.6),
                row=2, col=1
            )
        
        # Samsung data
        samsung_data = time_series.get('005930.KS', {})
        if samsung_data:
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            prices = samsung_data.get('prices', np.random.randn(30).cumsum() + 70000)
            
            fig.add_trace(
                go.Scatter(x=dates, y=prices, name='Samsung Price',
                          line=dict(color=self.colors['secondary'], width=3)),
                row=1, col=2
            )
            
            # RSI indicator
            rsi = samsung_data.get('rsi', np.random.uniform(30, 70, 30))
            fig.add_trace(
                go.Scatter(x=dates, y=rsi, name='RSI',
                          line=dict(color=self.colors['warning'], width=2)),
                row=2, col=2
            )
            
            # RSI threshold lines
            fig.add_hline(y=70, line=dict(color=self.colors['danger'], dash='dash'), 
                         row=2, col=2)
            fig.add_hline(y=30, line=dict(color=self.colors['success'], dash='dash'), 
                         row=2, col=2)
        
        fig.update_layout(
            height=600,
            title_text="Market Performance Dashboard",
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price (KRW)", row=1, col=2)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_prices")

    def _render_volatility_analysis(self, market_data: Dict[str, Any], key: str):
        """Render volatility analysis charts"""
        # Generate sample volatility data
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        
        # Historical volatility
        tsm_vol = np.random.uniform(0.15, 0.45, 252)
        samsung_vol = np.random.uniform(0.20, 0.50, 252)
        
        # Rolling volatility (30-day)
        tsm_rolling = pd.Series(tsm_vol).rolling(30).mean()
        samsung_rolling = pd.Series(samsung_vol).rolling(30).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Historical Volatility', 'Volatility Distribution'),
            vertical_spacing=0.15
        )
        
        # Time series volatility
        fig.add_trace(
            go.Scatter(x=dates, y=tsm_vol, name='TSM Daily Vol',
                      line=dict(color=self.colors['primary'], width=1), opacity=0.3),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=tsm_rolling, name='TSM 30-day Vol',
                      line=dict(color=self.colors['primary'], width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=samsung_vol, name='Samsung Daily Vol',
                      line=dict(color=self.colors['secondary'], width=1), opacity=0.3),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=samsung_rolling, name='Samsung 30-day Vol',
                      line=dict(color=self.colors['secondary'], width=3)),
            row=1, col=1
        )
        
        # Volatility distribution
        fig.add_trace(
            go.Histogram(x=tsm_vol, name='TSM Vol Distribution', 
                        marker_color=self.colors['primary'], opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=samsung_vol, name='Samsung Vol Distribution',
                        marker_color=self.colors['secondary'], opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="Volatility Analysis",
            title_x=0.5,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Volatility", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_volatility")

    def _render_correlation_matrix(self, market_data: Dict[str, Any], key: str):
        """Render correlation matrix heatmap"""
        # Generate sample correlation data
        symbols = ['TSM', 'Samsung', 'ASML', 'NVDA', 'AMD', 'INTC']
        np.random.seed(42)  # For reproducible results
        
        # Generate correlation matrix
        corr_matrix = np.random.uniform(-0.8, 0.9, (len(symbols), len(symbols)))
        # Make it symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Asset Correlation Matrix",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_correlation")

    def render_risk_metrics(self, portfolio_data: Dict[str, Any], key: str = "risk_metrics"):
        """Render comprehensive risk metrics dashboard"""
        st.subheader("⚠️ Risk Analytics")
        
        # Risk metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_var_gauge(portfolio_data, key)
        
        with col2:
            self._render_drawdown_chart(portfolio_data, key)
        
        with col3:
            self._render_risk_breakdown(portfolio_data, key)

    def _render_var_gauge(self, portfolio_data: Dict[str, Any], key: str):
        """Render Value at Risk gauge chart"""
        var_95 = portfolio_data.get('var_95', 2.5)  # 2.5% VaR
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=var_95,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "VaR (95%)"},
            delta={'reference': 2.0, 'increasing': {'color': self.colors['danger']}},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 1.5], 'color': self.colors['success']},
                    {'range': [1.5, 3], 'color': self.colors['warning']},
                    {'range': [3, 5], 'color': self.colors['danger']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3.5
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_var")

    def _render_drawdown_chart(self, portfolio_data: Dict[str, Any], key: str):
        """Render drawdown analysis chart"""
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        
        # Generate sample drawdown data
        returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,
            fill='tonexty',
            name='Drawdown',
            line=dict(color=self.colors['danger']),
            fillcolor=f'rgba(220, 53, 69, 0.3)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_drawdown")

    def _render_risk_breakdown(self, portfolio_data: Dict[str, Any], key: str):
        """Render risk factor breakdown"""
        risk_factors = portfolio_data.get('risk_factors', {
            'Market Risk': 45,
            'Credit Risk': 20,
            'Liquidity Risk': 15,
            'Operational Risk': 12,
            'Model Risk': 8
        })
        
        fig = go.Figure(data=[go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            marker=dict(
                color=px.colors.sequential.Reds_r,
                line=dict(color='rgba(0,0,0,0.8)', width=1)
            ),
            text=[f'{v}%' for v in risk_factors.values()],
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Risk Factor Breakdown",
            xaxis_title="Risk Contribution (%)",
            height=300,
            margin=dict(t=40, b=40, l=100, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_breakdown")

    def render_earnings_calendar(self, earnings_data: List[Dict[str, Any]], key: str = "earnings_calendar"):
        """Render earnings calendar and surprises"""
        st.subheader("📅 Earnings Calendar & Surprises")
        
        if not earnings_data:
            # Generate sample earnings data
            earnings_data = self._generate_sample_earnings_data()
        
        # Earnings surprises chart
        companies = [item['company'] for item in earnings_data]
        surprises = [item['surprise_pct'] for item in earnings_data]
        
        colors = [self.colors['success'] if x > 0 else self.colors['danger'] for x in surprises]
        
        fig = go.Figure(data=[go.Bar(
            x=companies,
            y=surprises,
            marker_color=colors,
            text=[f'{s:+.1f}%' for s in surprises],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Surprise: %{y:+.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title="Recent Earnings Surprises",
            xaxis_title="Company",
            yaxis_title="Earnings Surprise (%)",
            height=400,
            showlegend=False
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_surprises")
        
        # Earnings table
        df = pd.DataFrame(earnings_data)
        st.subheader("Upcoming Earnings")
        st.dataframe(
            df[['company', 'date', 'estimate', 'actual', 'surprise_pct']].round(2),
            use_container_width=True
        )

    def render_market_sentiment(self, sentiment_data: Dict[str, Any], key: str = "market_sentiment"):
        """Render market sentiment indicators"""
        st.subheader("🌡️ Market Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fear & Greed Index
            fear_greed = sentiment_data.get('fear_greed_index', 45)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fear_greed,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fear & Greed Index"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_sentiment_color(fear_greed)},
                    'steps': [
                        {'range': [0, 25], 'color': "#8B0000", 'name': 'Extreme Fear'},
                        {'range': [25, 45], 'color': "#FF4500", 'name': 'Fear'},
                        {'range': [45, 55], 'color': "#FFD700", 'name': 'Neutral'},
                        {'range': [55, 75], 'color': "#32CD32", 'name': 'Greed'},
                        {'range': [75, 100], 'color': "#006400", 'name': 'Extreme Greed'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_fear_greed")
        
        with col2:
            # VIX trend
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            vix_values = sentiment_data.get('vix_history', np.random.uniform(15, 35, 30))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=vix_values,
                mode='lines',
                name='VIX',
                line=dict(color=self.colors['warning'], width=3)
            ))
            
            # VIX threshold lines
            fig.add_hline(y=20, line_dash="dash", line_color="green", 
                         annotation_text="Low Volatility")
            fig.add_hline(y=30, line_dash="dash", line_color="red",
                         annotation_text="High Volatility")
            
            fig.update_layout(
                title="VIX Trend (30 days)",
                xaxis_title="Date",
                yaxis_title="VIX Level",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True, config=self.chart_config, key=f"{key}_vix")

    def _get_sentiment_color(self, value: float) -> str:
        """Get color based on sentiment value"""
        if value < 25:
            return "#8B0000"  # Extreme Fear - Dark Red
        elif value < 45:
            return "#FF4500"  # Fear - Orange Red
        elif value < 55:
            return "#FFD700"  # Neutral - Gold
        elif value < 75:
            return "#32CD32"  # Greed - Lime Green
        else:
            return "#006400"  # Extreme Greed - Dark Green

    def _generate_sample_market_data(self) -> Dict[str, Any]:
        """Generate sample market data for demonstration"""
        return {
            'time_series': {
                'TSM': {
                    'prices': np.random.randn(30).cumsum() + 100,
                    'volumes': np.random.randint(1000000, 5000000, 30)
                },
                '005930.KS': {
                    'prices': np.random.randn(30).cumsum() + 70000,
                    'rsi': np.random.uniform(30, 70, 30)
                }
            }
        }

    def _generate_sample_earnings_data(self) -> List[Dict[str, Any]]:
        """Generate sample earnings data"""
        companies = ['TSM', 'Samsung', 'ASML', 'NVDA', 'AMD']
        return [
            {
                'company': company,
                'date': '2024-01-15',
                'estimate': np.random.uniform(2.0, 5.0),
                'actual': np.random.uniform(1.5, 5.5),
                'surprise_pct': np.random.uniform(-10, 15)
            }
            for company in companies
        ]

    def render_complete_dashboard(self, data: Dict[str, Any], key: str = "complete_dashboard"):
        """Render the complete market dashboard"""
        st.title("📊 Market Intelligence Dashboard")
        
        # Portfolio overview
        self.render_portfolio_overview(
            data.get('portfolio', {}), 
            key=f"{key}_portfolio"
        )
        
        st.markdown("---")
        
        # Market performance
        self.render_market_performance(
            data.get('market', {}), 
            key=f"{key}_market"
        )
        
        st.markdown("---")
        
        # Risk metrics
        self.render_risk_metrics(
            data.get('portfolio', {}), 
            key=f"{key}_risk"
        )
        
        st.markdown("---")
        
        # Market sentiment and earnings
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_market_sentiment(
                data.get('sentiment', {}), 
                key=f"{key}_sentiment"
            )
        
        with col2:
            self.render_earnings_calendar(
                data.get('earnings', []), 
                key=f"{key}_earnings"
            )

def create_dashboard_demo():
    """Demo function for the market dashboard"""
    st.set_page_config(page_title="Market Dashboard Demo", layout="wide")
    
    dashboard = MarketDashboard()
    
    # Sample data
    sample_data = {
        'portfolio': {
            'total_aum': 1500000,
            'asia_tech_allocation': 0.22,
            'daily_pnl': 25000,
            'risk_score': 6.8,
            'aum_change': 0.035,
            'allocation_change': 0.04,
            'pnl_change': 0.12,
            'risk_change': -0.4,
            'var_95': 2.8,
            'allocations': {
                'Asia Tech': 22,
                'US Equities': 35,
                'European Stocks': 18,
                'Bonds': 15,
                'Cash': 10
            }
        },
        'market': {},
        'sentiment': {
            'fear_greed_index': 35,
            'vix_history': np.random.uniform(18, 28, 30)
        },
        'earnings': []
    }
    
    dashboard.render_complete_dashboard(sample_data, "demo")

if __name__ == "__main__":
    create_dashboard_demo()