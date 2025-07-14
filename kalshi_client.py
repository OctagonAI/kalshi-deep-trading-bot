"""
Simple Kalshi API Client with RSA authentication
"""

import hashlib
import json
import time
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import httpx
from loguru import logger
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from config import KalshiConfig


class KalshiClient:
    """Simple Kalshi API client for basic trading operations."""
    
    def __init__(self, config: KalshiConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.private_key = config.private_key
        self.client = None
        self.session_token = None
        
    async def login(self):
        """Login to Kalshi API."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        
        # For now, we'll assume the client handles authentication
        # In the real implementation, you'd do login here
        logger.info(f"Connected to Kalshi API at {self.base_url}")
        
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get all active markets."""
        try:
            headers = await self._get_headers("GET", "/trade-api/v2/markets")
            response = await self.client.get(
                "/trade-api/v2/markets",
                headers=headers,
                params={"status": "open"}
            )
            response.raise_for_status()
            
            data = response.json()
            markets = data.get("markets", [])
            
            # Transform to simpler format
            simple_markets = []
            for market in markets:
                simple_markets.append({
                    "ticker": market.get("ticker", ""),
                    "title": market.get("title", ""),
                    "status": market.get("status", ""),
                    "volume": market.get("volume", 0),
                    "yes_bid": market.get("yes_bid", 0),
                    "no_bid": market.get("no_bid", 0),
                    "yes_ask": market.get("yes_ask", 0),
                    "no_ask": market.get("no_ask", 0),
                })
            
            logger.info(f"Retrieved {len(simple_markets)} markets")
            return simple_markets
            
        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            return []
    
    async def place_order(self, ticker: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a simple order."""
        try:
            order_data = {
                "ticker": ticker,
                "side": side,  # "yes" or "no"
                "action": "buy",
                "amount": amount,
                "type": "market"
            }
            
            headers = await self._get_headers("POST", "/trade-api/v2/orders")
            response = await self.client.post(
                "/trade-api/v2/orders",
                headers=headers,
                json=order_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Order placed: {ticker} {side} ${amount}")
            return {"success": True, "order_id": result.get("order_id", "")}
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate headers with RSA signature."""
        timestamp = str(int(time.time() * 1000))
        
        # Create message to sign
        message = f"{timestamp}{method}{path}"
        
        # Sign the message
        signature = self._sign_message(message)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
    
    def _sign_message(self, message: str) -> str:
        """Sign a message using RSA private key."""
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            # Sign the message
            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Return base64 encoded signature
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose() 