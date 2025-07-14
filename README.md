# Kalshi Trading Bot with Octagon Deep Research

A sophisticated automated trading bot for Kalshi prediction markets that leverages Octagon Deep Research for comprehensive market analysis and trading insights.

## 🚀 Features

### Trading Strategies
- **Sentiment Momentum**: Trades based on market sentiment analysis
- **Event Arbitrage**: Exploits price-probability mismatches
- **Research-Based Trades**: Uses comprehensive research insights
- **Political Polling**: Specialized for political markets using polling data
- **Mean Reversion**: Technical analysis-based strategy
- **Correlation Pairs**: Multi-market correlation analysis

### Research Integration
- **Octagon Deep Research Agent**: Comprehensive market research and analysis
- **Sentiment Analysis**: Real-time sentiment tracking from multiple sources
- **Market Forecasting**: AI-powered probability predictions
- **Event Impact Analysis**: Research on how events affect markets
- **Portfolio Optimization**: Multi-market allocation recommendations

### Risk Management
- **Position Sizing**: Kelly criterion, volatility-adjusted, and confidence-based sizing
- **Stop Losses**: Automatic stop-loss orders with multiple trigger types
- **Portfolio Risk**: Real-time risk monitoring and alerts
- **Drawdown Protection**: Maximum drawdown limits
- **Concentration Risk**: Position diversification monitoring

### Advanced Features
- **Real-time WebSocket**: Live market data streaming
- **HMAC Authentication**: Secure API access
- **Rate Limiting**: Intelligent request throttling
- **Error Handling**: Comprehensive error recovery
- **Logging**: Detailed trading logs and monitoring
- **Database**: SQLite persistence for trades and research
- **Dry Run Mode**: Test strategies without real money

## 📋 Requirements

- Python 3.8+
- Kalshi API account and keys
- Octagon Deep Research API access
- At least 4GB RAM (for research processing)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/kalshi-trading-bot.git
   cd kalshi-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file based on the template:
   ```bash
   cp .env.template .env
   ```

4. **Configure API keys**
   Edit `.env` and add your API credentials:
   ```bash
   # Kalshi API Configuration
   KALSHI_API_KEY=your_kalshi_api_key_here
   KALSHI_API_SECRET=your_kalshi_api_secret_here
   
   # Octagon Deep Research API Configuration
   OCTAGON_API_KEY=your_octagon_api_key_here
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KALSHI_API_KEY` | Kalshi API key | Required |
| `KALSHI_API_SECRET` | Kalshi API secret | Required |
| `OCTAGON_API_KEY` | Octagon Deep Research API key | Required |
| `DRY_RUN` | Run in simulation mode | `true` |
| `MAX_POSITION_SIZE` | Maximum position size (% of portfolio) | `0.1` |
| `MAX_DAILY_LOSS` | Maximum daily loss limit | `0.05` |
| `STOP_LOSS_PERCENT` | Stop loss percentage | `0.15` |
| `MIN_PROBABILITY_THRESHOLD` | Minimum confidence for trades | `0.55` |
| `MAX_CONCURRENT_POSITIONS` | Maximum concurrent positions | `5` |

### Trading Strategies

Configure enabled strategies in `.env`:
```bash
ENABLED_STRATEGIES=sentiment_momentum,event_arbitrage,research_based_trades,political_polling
```

### Market Categories

Select market categories to trade:
```bash
MARKET_CATEGORIES=politics,economics,technology,sports,entertainment
```

## 🚀 Usage

### Basic Usage

1. **Start the bot**
   ```bash
   python trading_bot.py
   ```

2. **Monitor performance**
   The bot displays real-time status including:
   - Active trades
   - P&L tracking
   - Risk metrics
   - Trading opportunities

### Advanced Usage

```python
from trading_bot import TradingBot
from config import load_config

# Load configuration
config = load_config()

# Create and run bot
bot = TradingBot(config)
await bot.initialize()
await bot.run()
```

### Dry Run Mode

Always start with dry run mode enabled:
```bash
DRY_RUN=true
```

This allows you to:
- Test strategies without real money
- Validate API connections
- Monitor bot behavior
- Analyze performance

## 📊 Monitoring

### Real-time Dashboard

The bot provides a rich console interface showing:
- Portfolio value and daily P&L
- Active positions and their performance
- Risk alerts and metrics
- Trading opportunities analysis
- Strategy performance breakdown

### Database Tracking

All trades and research are stored in SQLite:
- `trades` table: Complete trade history
- `research_cache` table: Market research data
- `risk_alerts` table: Risk management alerts

### Logging

Comprehensive logging includes:
- Trade executions and closures
- Research analysis results
- Risk management actions
- API interactions
- Error tracking

## 🔧 API Integration

### Kalshi API

The bot uses the official Kalshi API (https://docs.kalshi.com/api-reference/api_keys/get-api_keys) with:
- API key authentication
- Rate limiting (5 requests/second default)
- WebSocket support for real-time data
- Order management (place, cancel, modify)
- Portfolio tracking

### Octagon Deep Research API

Integration with Octagon Deep Research Agent (https://docs.octagonagents.com/guide/agents/deep-research-agent.html):
- Comprehensive market analysis using `octagon-deep-research-agent` model
- Sentiment tracking from multiple sources
- Event impact research
- Portfolio optimization
- Multi-source data aggregation

## 🛡️ Risk Management

### Position Sizing

Multiple position sizing models:
- **Kelly Criterion**: Optimal bet sizing based on probability and odds
- **Volatility Adjusted**: Reduces size for high-volatility markets
- **Confidence Based**: Scales with research confidence
- **Risk Level**: Adjusts for strategy risk level

### Stop Loss Management

Automatic stop loss protection:
- Percentage-based stops
- Market volatility adjustments
- Strategy-specific parameters
- Real-time monitoring

### Portfolio Risk

Comprehensive risk monitoring:
- Daily loss limits
- Maximum drawdown protection
- Position concentration limits
- Real-time alerts

## 📈 Strategies

### Sentiment Momentum
Trades based on market sentiment analysis from news, social media, and analyst reports.

**Parameters:**
- Sentiment threshold: 0.3 (positive) / -0.3 (negative)
- Confidence threshold: 0.7
- Signal strength: 0.6+

### Event Arbitrage
Exploits price-probability mismatches based on research insights.

**Parameters:**
- Probability difference threshold: 0.15
- Confidence threshold: 0.7
- Conservative return estimate: 80% of probability difference

### Research-Based Trades
Uses comprehensive research analysis for high-confidence trades.

**Parameters:**
- Overall research score: 0.6+
- Signal strength: 0.7+
- Signal probability: 0.65+

### Political Polling
Specialized strategy for political markets using polling data.

**Parameters:**
- Polling confidence: 0.8+
- Price divergence: 0.1+
- Political market identification

### Mean Reversion
Technical analysis strategy based on price deviations from moving averages.

**Parameters:**
- Short MA: 5 periods
- Long MA: 20 periods
- Reversion threshold: 15% deviation

## 🔍 Research Analysis

### Market Research Pipeline

1. **Sentiment Analysis**: Multi-source sentiment tracking
2. **Forecast Generation**: AI-powered probability predictions
3. **Trading Signal**: Comprehensive buy/sell/hold recommendations
4. **Risk Assessment**: Market-specific risk evaluation
5. **Portfolio Impact**: Cross-market correlation analysis

### Research Components

- **Sentiment Score**: -1 to 1 range with confidence levels
- **Probability Forecast**: 0 to 1 with supporting factors
- **Trading Signal**: BUY/SELL/HOLD with strength and reasoning
- **Risk Level**: Low/Medium/High with specific factors
- **Expected Return**: Quantitative return estimates

## 🔒 Security

### API Security
- HMAC signature authentication
- Secure environment variable storage
- Rate limiting and error handling
- Request validation and sanitization

### Trading Security
- Dry run mode for testing
- Position size limits
- Stop loss protection
- Risk monitoring and alerts

## 📝 Logging and Monitoring

### Log Levels
- **DEBUG**: API requests and responses
- **INFO**: Trading decisions and executions
- **WARNING**: Risk alerts and unusual conditions
- **ERROR**: System errors and failures

### Log Rotation
- Maximum file size: 10MB
- Retention period: 30 days
- Automatic compression

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Backtesting
```bash
python backtest.py --start-date 2024-01-01 --end-date 2024-12-31
```

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check API keys and secrets
   - Verify network connectivity
   - Check rate limits

2. **Research Failures**
   - Verify Octagon API key
   - Check daily rate limits
   - Monitor response parsing

3. **Trading Errors**
   - Check account balance
   - Verify market status
   - Review position limits

### Debug Mode
```bash
LOG_LEVEL=DEBUG python trading_bot.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves significant risk and you should:

- Never risk more than you can afford to lose
- Thoroughly test strategies in dry run mode
- Monitor bot performance closely
- Understand all risks involved
- Consider consulting with financial professionals

The authors are not responsible for any trading losses or damages resulting from the use of this software.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 📋 Roadmap

- [ ] Advanced backtesting framework
- [ ] Strategy optimization tools
- [ ] Multi-exchange support
- [ ] Advanced risk metrics
- [ ] Web-based monitoring dashboard
- [ ] Machine learning integration
- [ ] Social trading features

---

**Happy Trading! 🚀** 