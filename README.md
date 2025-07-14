# Simple Kalshi Trading Bot

A straightforward trading bot for Kalshi prediction markets that uses Octagon Deep Research for market analysis and OpenAI for structured betting decisions.

## How It Works

The bot follows a simple 4-step workflow:

1. **Fetch Events**: Gets top 50 events from Kalshi sorted by volume
2. **Fetch Markets**: Gets all markets for each event
3. **Research Events**: Uses Octagon Deep Research to analyze event + markets (without odds)
4. **Make Decisions**: Feeds research results into OpenAI for structured betting decisions
5. **Place Bets**: Executes the recommended bets via Kalshi API

## Features

- **Simple & Direct**: No complex strategies or risk management systems
- **AI-Powered**: Uses Octagon Deep Research for market analysis and OpenAI for decision making
- **Event-Based**: Analyzes entire events with all markets for better context
- **Flexible Environment**: Supports both demo and production environments
- **Dry Run Mode**: Test the bot without placing real bets
- **Rich Console**: Beautiful progress tracking and result display with probability predictions

## Quick Start

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set Up Environment

Copy the environment template and fill in your API credentials:

```bash
cp env_template.txt .env
# Edit .env with your API keys
```

Required API keys:
- **Kalshi API**: Get from [kalshi.com](https://kalshi.com/profile/api) or [demo.kalshi.co](https://demo.kalshi.co)
- **Octagon API**: Contact Octagon team for access
- **OpenAI API**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

### 4. Run the Bot

```bash
uv run trading_bot.py
```

Or use the installed command:

```bash
uv run trading-bot
```

## Configuration

Key settings in `.env`:

```env
# Environment
KALSHI_USE_DEMO=true          # Use demo environment for testing
DRY_RUN=true                  # Simulate trades without real money

# Limits
MAX_MARKETS=50                # Maximum events to process
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
├── pyproject.toml          # uv dependencies and project config
├── env_template.txt        # Environment configuration template
└── README.md              # This file
```

## API Integrations

### Kalshi API
- **Authentication**: RSA signature-based authentication
- **Events**: Fetches top events sorted by volume
- **Markets**: Gets all markets for each event
- **Orders**: Places buy/sell orders for YES/NO positions

### Octagon Deep Research
- **Research**: Analyzes event + markets for sentiment, news, and trading factors
- **Probability Predictions**: Provides independent probability assessments
- **Insights**: Gives actionable trading recommendations
- **Risk Assessment**: Identifies key risk factors for each market

### OpenAI API
- **Structured Output**: Uses GPT-4 for betting decision analysis
- **Decision Making**: Processes research data into actionable bets
- **Risk Management**: Built-in confidence thresholds and position sizing

## Example Output

```
Step 1: Fetching top events...
✓ Found 50 events

Step 2: Fetching markets for 50 events...
✓ Found 247 total markets across 45 events

Step 3: Researching 45 events...
✓ Researched NYC-MAYOR-2025
Predicted probabilities for NYC-MAYOR-2025:
  NYC-MAYOR-ZOHRAN: 71.0%
  NYC-MAYOR-ADAMS: 13.0%
✓ Completed research on 42 events

Step 4: Generating betting decisions...
✓ Generated 34 betting decisions

Step 5: Placing bets...
Bets to be placed:
NYC-MAYOR-ZOHRAN: buy_yes $25.00
  Reasoning: Research shows 71% probability, current market odds undervalue this candidate
  Confidence: 0.85

✓ Placed buy_yes bet on NYC-MAYOR-ZOHRAN for $25.00
  Reasoning: Research shows 71% probability, current market odds undervalue this candidate
✓ Successfully placed 28 bets
✓ Total amount bet: $1,247.50
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
1. `SimpleTradingBot.get_top_events()` - Fetch top events from Kalshi
2. `SimpleTradingBot.get_markets_for_events()` - Get markets for each event
3. `SimpleTradingBot.research_events()` - Research each event with Octagon
4. `SimpleTradingBot.get_betting_decisions()` - Process research with OpenAI
5. `SimpleTradingBot.place_bets()` - Execute bets via Kalshi

### Key Classes

- **SimpleTradingBot**: Main orchestration class
- **KalshiClient**: Kalshi API interface
- **OctagonClient**: Octagon Deep Research interface
- **BettingDecision**: Individual betting decision model
- **MarketAnalysis**: Complete analysis with multiple decisions

### Development Commands

```bash
# Install development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8 .
```

### Error Handling

The bot handles various error scenarios:
- API rate limits and timeouts
- Market data inconsistencies
- Authentication failures
- Network connectivity issues

## Limitations

- **Event Coverage**: Processes events sequentially to avoid rate limits
- **Research Quality**: Depends on Octagon Deep Research data quality
- **Decision Making**: Relies on OpenAI's analysis capabilities
- **Risk Management**: Basic position sizing and confidence thresholds only

## License

This project is for educational and research purposes. Use at your own risk.

## Support

For issues or questions:
1. Check the error logs for detailed error messages
2. Verify API credentials and rate limits
3. Test with smaller event limits first
4. Use dry run mode for debugging 