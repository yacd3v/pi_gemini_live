"""
Gemini Client Module for VocalGem robot
Handles Gemini API client and session management
"""

import os

# Conditional import for Google Genai
try:
    from google import genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None

from .config import GOOGLE_API_KEY, MODEL, CONFIG

class GeminiClient:
    """Handles Gemini API client and session management"""
    
    def __init__(self):
        self.client = None
        self.session = None
        
    def initialize_client(self):
        """Initialize the Gemini client"""
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("Google Genai library not available")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        print("[GeminiClient] Connecting to Gemini Live...")
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=GOOGLE_API_KEY
        )
        
    def create_session(self):
        """Create a new Gemini Live session context manager"""
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("Google Genai library not available")
        if not self.client:
            self.initialize_client()
            
        # The connect method returns an async context manager, not a session directly
        # We return the context manager directly for use with async with
        return self.client.aio.live.connect(model=MODEL, config=CONFIG)
        
    def get_session(self):
        """Get the current session"""
        return self.session 