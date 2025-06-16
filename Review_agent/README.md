# Bus Service Analyzer

This script analyzes bus service feedback data using OpenAI's GPT model to provide insights and recommendations based on user feedback.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
   - Create a `.env` file in the project directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your-api-key-here`

## Usage

1. Place your JSON data file (named `ex.json`) in the same directory as the script
2. Run the script:
```bash
python bus_analyzer.py
```

## Features

- Processes bus service feedback data from JSON
- Analyzes sentiment for different service aspects (tags)
- Calculates sentiment scores for each aspect
- Uses GPT-3.5-turbo to provide detailed analysis and recommendations
- Handles multiple bus services in a single JSON file

## Output

The script will provide:
- Service details (route, bus type, overall rating)
- Analysis of each feedback tag
- LLM-generated recommendations for improvements
- Sentiment scores for different aspects of the service