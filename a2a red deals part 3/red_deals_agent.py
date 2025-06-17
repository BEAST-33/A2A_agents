import os
import json
import logging
import requests
import tiktoken
from datetime import datetime, UTC
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Default input/output modes
DEFAULT_INPUT_MODES = ["application/json", "text/plain"]
DEFAULT_OUTPUT_MODES = ["application/json"]

def _load_system_prompt():
    try:
        with open('system_prompt.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return ""

def fetch_redbus_deals() -> Optional[Dict[str, Any]]:
    """
    Fetches deals data from RedBus API
    """
    try:
        api_url = os.getenv("REDBUS_API_URL", "http://api-rbplus-prod.redbus.in/api/Campaign/GetAllCampaignDetails")
        operator_id = os.getenv("OPERATOR_ID", "15926")
        is_international = os.getenv("IS_INTERNATIONAL", "false")
        
        params = {
            "operatorID": operator_id,
            "isInternational": is_international
        }
        
        logger.info(f"Fetching deals from RedBus API with params: {params}")
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        deals_data = response.json()
        logger.info(f"Successfully fetched {len(deals_data) if isinstance(deals_data, list) else 'unknown number of'} deals")
        return deals_data
    except Exception as e:
        logger.error(f"Error fetching deals from RedBus API: {e}")
        return None

def count_tokens(text, model="cl100k_base"):
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

def generate_chat_response(user_message: str, deals_data: Optional[Dict[str, Any]] = None) -> tuple[str, dict]:
    api_url = os.getenv("CHAT_API", "http://10.120.17.147:11434/api/chat")
    model_name = os.getenv("CHAT_MODEL", "qwen2.5-coder:14b")
    system_prompt = _load_system_prompt()
    
    token_usage = {"system_tokens": 0, "user_tokens": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_message)
        total_input_tokens = system_tokens + user_tokens
        token_usage["system_tokens"] = system_tokens
        token_usage["user_tokens"] = user_tokens
        token_usage["input_tokens"] = total_input_tokens
        logger.info(f"Input tokens - System: {system_tokens}, User: {user_tokens}, Total: {total_input_tokens}")
        
        # If deals data is provided, include it in the user message
        enhanced_message = user_message
        if deals_data:
            deals_json = json.dumps(deals_data, indent=2)
            enhanced_message = f"{user_message}\n\nHere is the deals data to analyze:\n{deals_json}"
            enhanced_tokens = count_tokens(enhanced_message)
            token_usage["user_tokens"] = enhanced_tokens
            token_usage["input_tokens"] = token_usage["system_tokens"] + enhanced_tokens
        
        payload = {
            "stream": False, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_message}
            ], 
            "model": model_name
        }
        headers = {'Content-Type': 'application/json'}
        
        logger.info("Making API call to generate response...")
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        response_content = response_json["message"]["content"]
        
        output_tokens = count_tokens(response_content)
        token_usage["output_tokens"] = output_tokens
        token_usage["total_tokens"] = token_usage["input_tokens"] + output_tokens
        
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"Total tokens: {token_usage['total_tokens']}")
        
        return response_content, token_usage
    except Exception as e:
        logger.error(f"Error in generate_chat_response: {e}")
        return f"Error generating response: {str(e)}", token_usage

def make_a2a_response(request_id, task_id, state, message_text, results=None):
    now = datetime.now(UTC).isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    
    if state == "completed" and results is not None:
        result["artifacts"] = [
            {"parts": [
                {"type": "json", "json": results}
            ], "index": 0}
        ]
    else:
        result["artifacts"] = [
            {"parts": [
                {"type": "text", "text": message_text}
            ], "index": 0}
        ]
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logger.debug(f"Received payload: {payload}")
    
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    message = params.get("message", {})
    parts = message.get("parts", [])
    
    if isinstance(parts, dict):
        user_text = parts.get("text", "")
    elif isinstance(parts, list) and parts:
        user_text = parts[0].get("text", "") if isinstance(parts[0], dict) else parts[0]
    else:
        user_text = ""
        
    if not user_text:
        logger.error("No user text provided in the request.")
        error_message = "No text provided."
        return make_a2a_response(request_id, task_id, "failed", error_message)

    # Fetch deals data from RedBus API
    deals_data = fetch_redbus_deals()
    if deals_data is None:
        logger.error("Failed to fetch deals data from RedBus API.")
        error_message = "Error fetching deals data."
        return make_a2a_response(request_id, task_id, "failed", error_message)

    response_content, token_usage = generate_chat_response(user_text, deals_data)
    if response_content:
        logger.info(f"Generated response: {response_content[::]}...")
        return make_a2a_response(request_id, task_id, "completed", response_content)
    else:
        logger.error("Failed to generate response.")
        error_message = "Error generating response."
        return make_a2a_response(request_id, task_id, "failed", error_message)     
        
@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)
        
@app.get("/.well-known/agent.json")
async def agent_manifest():
    return {
        "name": "RedDealsAnalysisAgent",
        "description": "Agent that analyzes and categorizes RedDeals data from RedBus API.",
        "url": os.getenv("PUBLIC_URL", "http://localhost:8035/"),
        "version": "1.0.0",
        "skills": [
            {
                "id": "red_deals_analysis",
                "name": "RedDeals Analysis",
                "description": "Analyzes and categorizes RedDeals data from RedBus API, identifying patterns and special conditions.",
                "tags": ["deals"],
                "examples": ["Analyze current deals", "Show active deals", "Find deals ending soon"],
                "inputModes": DEFAULT_INPUT_MODES,
                "outputModes": DEFAULT_OUTPUT_MODES
            }
        ]
    }

# Healthcheck endpoint - useful for monitoring
@app.get("/health")
async def health():
    # Test RedBus API connection
    deals_data = fetch_redbus_deals()
    if deals_data is None:
        return {"status": "error", "message": "RedBus API connection failed"}
    
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8010"))
    
    uvicorn.run(app, host=host, port=port)