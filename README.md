# Simple Kalshi Trading Bot

A straightforward trading bot for Kalshi prediction markets that uses Octagon Deep Research for market analysis and OpenAI for structured betting decisions.

## How It Works

The bot follows a simple 4-step workflow:

1. **Fetch Markets**: Retrieves all active markets from Kalshi, sorted by volume (descending)
2. **Research Markets**: Uses Octagon Deep Research to analyze each market for trading insights
3. **Make Decisions**: Feeds research results into OpenAI for structured betting decisions
4. **Place Bets**: Executes the recommended bets via Kalshi API

## Features

- **Simple & Direct**: No complex strategies or risk management systems
- **AI-Powered**: Uses Octagon Deep Research for market analysis and OpenAI for decision making
- **Flexible Environment**: Supports both demo and production environments
- **Dry Run Mode**: Test the bot without placing real bets
- **Rich Console**: Beautiful progress tracking and result display

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy the environment template and fill in your API credentials:

```bash
cp env_template.txt .env
# Edit .env with your API keys
```

Required API keys:
- **Kalshi API**: Get from [kalshi.com](https://kalshi.com/profile/api) or [demo.kalshi.co](https://demo.kalshi.co)
- **Octagon API**: Contact Octagon team for access
- **OpenAI API**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

### 3. Run the Bot

```bash
python trading_bot.py
```

## Configuration

Key settings in `.env`:

```env
# Environment
KALSHI_USE_DEMO=true          # Use demo environment for testing
DRY_RUN=true                  # Simulate trades without real money

# Limits
MAX_MARKETS=50                # Maximum markets to process
MAX_BET_AMOUNT=25.0           # Maximum bet per market

# API Keys
KALSHI_API_KEY=your_key
KALSHI_PRIVATE_KEY=your_private_key
OCTAGON_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Recommended Testing Flow

1. **Demo + Dry Run**: Start with `KALSHI_USE_DEMO=true` and `DRY_RUN=true`
2. **Demo + Live**: Test with `KALSHI_USE_DEMO=true` and `DRY_RUN=false`
3. **Production**: Only use `KALSHI_USE_DEMO=false` after thorough testing

## Project Structure

```
├── trading_bot.py          # Main bot execution
├── config.py               # Configuration management
├── kalshi_client.py        # Kalshi API client
├── research_client.py      # Octagon Deep Research client
├── betting_models.py       # Pydantic models for betting decisions
├── requirements.txt        # Python dependencies
├── env_template.txt        # Environment configuration template
└── README.md              # This file
```

## API Integrations

### Kalshi API
- **Authentication**: RSA signature-based authentication
- **Markets**: Fetches active markets sorted by volume
- **Orders**: Places buy/sell orders for YES/NO positions

### Octagon Deep Research
- **Research**: Analyzes market sentiment, news, and trading factors
- **Insights**: Provides actionable trading recommendations
- **Risk Assessment**: Identifies key risk factors for each market

### OpenAI API
- **Structured Output**: Uses GPT-4 for betting decision analysis
- **Decision Making**: Processes research data into actionable bets
- **Risk Management**: Built-in confidence thresholds and position sizing

## Example Output

```
Step 1: Fetching active markets...
✓ Found 50 active markets

Step 2: Researching 50 markets...
✓ Researched PRES-2024-12-31
✓ Researched STOCKS-2024-12-31
✓ Completed research on 48 markets

Step 3: Generating betting decisions...
✓ Generated 15 betting decisions

Step 4: Placing bets...
✓ Placed buy_yes bet on PRES-2024-12-31 for $25.00
✓ Placed buy_no bet on STOCKS-2024-12-31 for $15.00
✓ Successfully placed 12 bets
✓ Total amount bet: $245.00
```

## Safety Features

- **Demo Environment**: Test with mock funds before live trading
- **Dry Run Mode**: Simulate all operations without real money
- **Position Limits**: Configurable maximum bet amounts
- **Confidence Thresholds**: Only bet on high-confidence opportunities
- **Error Handling**: Comprehensive error handling and logging

## Development

### Architecture

The bot uses a simple, linear workflow:
1. `SimpleTradingBot.get_active_markets()` - Fetch markets from Kalshi
2. `SimpleTradingBot.research_markets()` - Research each market with Octagon
3. `SimpleTradingBot.get_betting_decisions()` - Process research with OpenAI
4. `SimpleTradingBot.place_bets()` - Execute bets via Kalshi

### Key Classes

- **SimpleTradingBot**: Main orchestration class
- **KalshiClient**: Kalshi API interface
- **OctagonClient**: Octagon Deep Research interface
- **BettingDecision**: Individual betting decision model
- **MarketAnalysis**: Complete analysis with multiple decisions

### Error Handling

The bot handles various error scenarios:
- API rate limits and timeouts
- Market data inconsistencies
- Authentication failures
- Network connectivity issues

## Limitations

- **Market Coverage**: Processes markets sequentially to avoid rate limits
- **Research Quality**: Depends on Octagon Deep Research data quality
- **Decision Making**: Relies on OpenAI's analysis capabilities
- **Risk Management**: Basic position sizing and confidence thresholds only

## License

This project is for educational and research purposes. Use at your own risk.

## Support

For issues or questions:
1. Check the error logs for detailed error messages
2. Verify API credentials and rate limits
3. Test with smaller market limits first
4. Use dry run mode for debugging 