# Vocal Gemini

A Python application that enables voice conversations with Google's Gemini AI using the Live API.

## Prerequisites

- Python 3.11 or higher
- A Google API key with access to the Gemini API
- A USB microphone and speaker connected to your Raspberry Pi

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Google API key as an environment variable:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

1. Run the script:
```bash
python vocal_gemini.py
```

2. Start speaking to the AI. The conversation will be:
   - Your voice input will be sent to Gemini
   - Gemini's responses will be played through your speakers
   - Text transcriptions will be shown in the terminal

3. To end the conversation, press Ctrl+C

## Notes

- Make sure your microphone and speakers are properly connected and configured
- The script uses the default audio input and output devices
- For best results, use headphones to prevent echo/feedback
- The conversation can be ended at any time by pressing Ctrl+C 