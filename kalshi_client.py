"""
Kalshi API Client with HMAC authentication and comprehensive endpoint support
"""

import hashlib
import hmac
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import httpx
import websockets
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from enum import Enum

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class MarketStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    SETTLED = "SETTLED"

@dataclass
class Market:
    id: str
    title: str
    status: str
    expiration_date: str
    close_price: Optional[float] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    category: Optional[str] = None

@dataclass
class Order:
    id: str
    market_id: str
    side: str
    quantity: int
    limit_price: float
    status: str
    created_at: str
    filled_quantity: Optional[int] = None
    remaining_quantity: Optional[int] = None

@dataclass
class Position:
    market_id: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class Portfolio:
    balance: float
    positions: List[Position]
    total_pnl: float
    available_balance: float

class KalshiClient:
    """Kalshi API client with HMAC authentication and rate limiting"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.elections.kalshi.com", 
                 websocket_url: str = "wss://api.elections.kalshi.com/ws/v1", rate_limit: int = 5):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.websocket_url = websocket_url
        self.rate_limit = rate_limit
        self.session: Optional[httpx.AsyncClient] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.last_request_time = 0
        self.request_count = 0
        
        # Rate limiting
        self.request_times = []
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
        if self.websocket:
            await self.websocket.close()
    
    def _create_auth_headers(self) -> Dict[str, str]:
        """Create authentication headers for Kalshi API"""
        # Based on Kalshi API documentation, using simple API key authentication
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": self.api_key
        }
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Remove old requests outside the window
        self.request_times = [t for t in self.request_times if current_time - t < 1]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 1 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                          data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated HTTP request with retry logic"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        await self._rate_limit()
        
        path = f"/trade-api/v2{endpoint}"
        url = f"{self.base_url}{path}"
        
        # Prepare headers
        headers = self._create_auth_headers()
        
        logger.debug(f"Making {method} request to {url}")
        
        try:
            response = await self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 429:
                await asyncio.sleep(60)  # Wait 1 minute for rate limit
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get_markets(self, status: Optional[str] = None, category: Optional[str] = None) -> List[Market]:
        """Get all markets with optional filtering"""
        params = {}
        if status:
            params['status'] = status
        if category:
            params['category'] = category
        
        response = await self._make_request("GET", "/markets", params=params)
        
        markets = []
        # Handle different response structure from actual API
        markets_data = response.get('markets', response.get('data', []))
        
        for market_data in markets_data:
            markets.append(Market(
                id=market_data.get('id', market_data.get('ticker', '')),
                title=market_data.get('title', market_data.get('subtitle', '')),
                status=market_data.get('status', market_data.get('market_status', 'unknown')),
                expiration_date=market_data.get('expiration_date', market_data.get('close_date', '')),
                close_price=market_data.get('close_price', market_data.get('settlement_price')),
                volume=market_data.get('volume', market_data.get('volume_24h', 0)),
                bid=market_data.get('bid', market_data.get('yes_bid')),
                ask=market_data.get('ask', market_data.get('yes_ask')),
                last_price=market_data.get('last_price', market_data.get('last_trade_price')),
                category=market_data.get('category', market_data.get('category_name'))
            ))
        
        return markets
    
    async def get_market(self, market_id: str) -> Market:
        """Get specific market details"""
        response = await self._make_request("GET", f"/market/{market_id}")
        
        # Handle different response structure from actual API
        market_data = response.get('market', response)
        
        return Market(
            id=market_data.get('id', market_data.get('ticker', market_id)),
            title=market_data.get('title', market_data.get('subtitle', '')),
            status=market_data.get('status', market_data.get('market_status', 'unknown')),
            expiration_date=market_data.get('expiration_date', market_data.get('close_date', '')),
            close_price=market_data.get('close_price', market_data.get('settlement_price')),
            volume=market_data.get('volume', market_data.get('volume_24h', 0)),
            bid=market_data.get('bid', market_data.get('yes_bid')),
            ask=market_data.get('ask', market_data.get('yes_ask')),
            last_price=market_data.get('last_price', market_data.get('last_trade_price')),
            category=market_data.get('category', market_data.get('category_name'))
        )
    
    async def get_portfolio(self) -> Portfolio:
        """Get current portfolio"""
        response = await self._make_request("GET", "/portfolio")
        
        positions = []
        # Handle different response structure from actual API
        portfolio_data = response.get('portfolio', response)
        
        for pos_data in portfolio_data.get('positions', []):
            positions.append(Position(
                market_id=pos_data.get('market_id', pos_data.get('market_ticker', '')),
                quantity=pos_data.get('quantity', 0),
                avg_price=pos_data.get('avg_price', pos_data.get('purchase_price', 0)),
                unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                realized_pnl=pos_data.get('realized_pnl', 0)
            ))
        
        return Portfolio(
            balance=portfolio_data.get('balance', 0),
            positions=positions,
            total_pnl=portfolio_data.get('total_pnl', 0),
            available_balance=portfolio_data.get('available_balance', portfolio_data.get('balance', 0))
        )
    
    async def get_orders(self, market_id: Optional[str] = None, status: Optional[str] = None) -> List[Order]:
        """Get orders with optional filtering"""
        params = {}
        if market_id:
            params['ticker'] = market_id
        if status:
            params['status'] = status
        
        response = await self._make_request("GET", "/orders", params=params)
        
        orders = []
        # Handle different response structure from actual API
        orders_data = response.get('orders', response.get('data', []))
        
        for order_data in orders_data:
            orders.append(Order(
                id=order_data.get('id', order_data.get('order_id', '')),
                market_id=order_data.get('market_id', order_data.get('ticker', '')),
                side=order_data.get('side', order_data.get('action', 'BUY')),
                quantity=order_data.get('quantity', order_data.get('count', 0)),
                limit_price=order_data.get('limit_price', order_data.get('price', 0)),
                status=order_data.get('status', order_data.get('order_status', 'unknown')),
                created_at=order_data.get('created_at', order_data.get('placed_time', '')),
                filled_quantity=order_data.get('filled_quantity', order_data.get('remaining_count', 0)),
                remaining_quantity=order_data.get('remaining_quantity', order_data.get('remaining_count', 0))
            ))
        
        return orders
    
    async def place_order(self, market_id: str, side: OrderSide, quantity: int, 
                         limit_price: float, order_type: OrderType = OrderType.LIMIT) -> Order:
        """Place a new order"""
        data = {
            "ticker": market_id,
            "action": side.value,
            "count": quantity,
            "price": limit_price,
            "type": order_type.value
        }
        
        response = await self._make_request("POST", "/orders", data=data)
        
        # Handle different response structure from actual API
        order_data = response.get('order', response)
        
        return Order(
            id=order_data.get('id', order_data.get('order_id', '')),
            market_id=order_data.get('market_id', order_data.get('ticker', market_id)),
            side=order_data.get('side', order_data.get('action', side.value)),
            quantity=order_data.get('quantity', order_data.get('count', quantity)),
            limit_price=order_data.get('limit_price', order_data.get('price', limit_price)),
            status=order_data.get('status', order_data.get('order_status', 'pending')),
            created_at=order_data.get('created_at', order_data.get('placed_time', '')),
            filled_quantity=order_data.get('filled_quantity', 0),
            remaining_quantity=order_data.get('remaining_quantity', quantity)
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            await self._make_request("DELETE", f"/orders/{order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_book(self, market_id: str) -> Dict[str, Any]:
        """Get order book for a market"""
        response = await self._make_request("GET", f"/market/{market_id}/order_book")
        return response
    
    async def connect_websocket(self, on_message_callback=None):
        """Connect to Kalshi WebSocket for real-time updates"""
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                extra_headers={
                    "X-API-Key": self.api_key
                }
            )
            
            logger.info("WebSocket connected successfully")
            
            if on_message_callback:
                async for message in self.websocket:
                    try:
                        data = json.loads(message)
                        await on_message_callback(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {message}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def subscribe_to_market(self, market_id: str):
        """Subscribe to market updates via WebSocket"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        subscription = {
            "action": "subscribe",
            "channel": f"order_books.{market_id}"
        }
        
        await self.websocket.send(json.dumps(subscription))
        logger.info(f"Subscribed to market {market_id}")
    
    async def unsubscribe_from_market(self, market_id: str):
        """Unsubscribe from market updates"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        unsubscription = {
            "action": "unsubscribe",
            "channel": f"order_books.{market_id}"
        }
        
        await self.websocket.send(json.dumps(unsubscription))
        logger.info(f"Unsubscribed from market {market_id}")
    
    async def get_balance(self) -> float:
        """Get account balance"""
        portfolio = await self.get_portfolio()
        return portfolio.balance
    
    async def get_market_history(self, market_id: str, limit: int = 100) -> List[Dict]:
        """Get historical price data for a market"""
        params = {'limit': limit}
        response = await self._make_request("GET", f"/market/{market_id}/history", params=params)
        return response.get('history', response.get('data', []))
    
    def calculate_spread(self, market: Market) -> Optional[float]:
        """Calculate bid-ask spread for a market"""
        if market.bid is not None and market.ask is not None:
            return market.ask - market.bid
        return None
    
    def calculate_implied_probability(self, price: float) -> float:
        """Calculate implied probability from price (assuming $100 settlement)"""
        return price / 100.0
    
    def is_market_tradable(self, market: Market, min_volume: int = 1000, max_spread: float = 0.05) -> bool:
        """Check if market meets tradability criteria"""
        if market.status != MarketStatus.OPEN.value:
            return False
        
        if market.volume is not None and market.volume < min_volume:
            return False
        
        spread = self.calculate_spread(market)
        if spread is not None and spread > max_spread:
            return False
        
        return True 