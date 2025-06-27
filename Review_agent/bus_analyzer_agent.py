import os
import json
import logging
import requests
import tiktoken
from datetime import datetime, UTC
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict

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

@dataclass
class TagAnalysis:
    tag: str
    positive_count: int
    negative_count: int
    total_count: int
    sentiment_score: float

def _load_system_prompt():
    try:
        with open('bus_analyzer_prompt.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return ""

def fetch_bus_ratings(operator_id: str, source_id: str, dest_id: str, limit: int = 10, page: int = 1) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches bus service ratings data from RedBus API
    """
    try:
        base_url = os.getenv("REDBUS_RATINGS_API_URL", "http://rbplus-api-prod.redbus.in:8085/rpwapp/v1/rnr/servicewiseratings")
        
        params = {
            'limit': limit,
            'page': page,
            'sourceId': source_id,
            'destId': dest_id,
            'sortOrder': 'desc',
            'sortColumn': 'sales'
        }
        
        logger.info(f"Fetching ratings from RedBus API with params: {params}")
        response = requests.get(f"{base_url}/{operator_id}", params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status_key') != 'suc':
            raise Exception(f"API returned error: {data.get('status_key')}")
            
        services_data = data.get('data', [])
        logger.info(f"Successfully fetched {len(services_data)} services")
        return services_data
    except Exception as e:
        logger.error(f"Error fetching ratings from RedBus API: {e}")
        return None

def process_tags(tags_json: List[Dict]) -> Dict[str, TagAnalysis]:
    """
    Process and analyze tags from bus service ratings
    """
    logger.info(f"Processing {len(tags_json)} tags")
    
    tag_analysis = defaultdict(lambda: TagAnalysis(
        tag="",
        positive_count=0,
        negative_count=0,
        total_count=0,
        sentiment_score=0.0
    ))
    
    for tag_data in tags_json:
        tag = tag_data['tags']
        sentiment = int(tag_data['sentimentId'])
        user_count = tag_data['userCount']
        
        analysis = tag_analysis[tag]
        analysis.tag = tag
        if sentiment == 0:  # Negative sentiment
            analysis.negative_count += user_count
        else:  # Positive sentiment
            analysis.positive_count += user_count
        analysis.total_count += user_count
    
    # Calculate sentiment scores
    for tag, analysis in tag_analysis.items():
        if analysis.total_count > 0:
            analysis.sentiment_score = ((analysis.positive_count - analysis.negative_count) / analysis.total_count * 100)
    
    return dict(tag_analysis)

def count_tokens(text, model="cl100k_base"):
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

def generate_chat_response(user_message: str, services_data: Optional[List[Dict[str, Any]]] = None) -> tuple[str, dict]:
    api_url = os.getenv("CHAT_API", "http://10.166.8.126:11434/api/chat")
    model_name = os.getenv("CHAT_MODEL", "mistral:7b-instruct")
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
        
        # If services data is provided, process and analyze it
        enhanced_message = user_message
        if services_data:
            analysis_data = []
            for service in services_data:
                service_info = {
                    "source": service['ServiceDetails']['SrcNm'],
                    "destination": service['ServiceDetails']['DestNm'],
                    "bus_type": service['BusTypelist'][0],
                    "overall_rating": service['overallRating'],
                    "tag_analysis": process_tags(service['TagsJson'])
                }
                analysis_data.append(service_info)
            
            current_date = datetime.now(UTC).strftime("%Y-%m-%d")
            analysis_summary = {
                "current_date": current_date,
                "total_services": len(services_data),
                "services_analysis": analysis_data
            }
            analysis_json = json.dumps(analysis_summary, indent=2)
            enhanced_message = f"{user_message}\n\nHere is the bus service analysis data:\n{analysis_json}"
            enhanced_tokens = count_tokens(enhanced_message)
            token_usage["user_tokens"] = enhanced_tokens
            token_usage["input_tokens"] = token_usage["system_tokens"] + enhanced_tokens
        
        payload = {
            "stream": False, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_message}
            ], 
            "model": model_name,
            "options": {
                "temperature": 0.0
            }
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

    # Extract parameters from user text or use defaults
    operator_id = os.getenv("OPERATOR_ID", "22")
    source_id = os.getenv("SOURCE_ID", "122")
    dest_id = os.getenv("DEST_ID", "124")

    # Fetch bus service ratings data
    services_data = fetch_bus_ratings(operator_id, source_id, dest_id)
    if services_data is None:
        logger.error("Failed to fetch bus service ratings data.")
        error_message = "Error fetching bus service ratings data."
        return make_a2a_response(request_id, task_id, "failed", error_message)

    response_content, token_usage = generate_chat_response(user_text, services_data)
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
        "name": "BusServiceAnalyzerAgent",
        "description": "Agent that analyzes bus service ratings and feedback from RedBus API.",
        "url": os.getenv("PUBLIC_URL", "http://localhost:8035/"),
        "version": "1.0.0",
        "skills": [
            {
                "id": "bus_service_analysis",
                "name": "Bus Service Analysis",
                "description": "Analyzes bus service ratings, feedback, and provides insights about service quality.",
                "tags": ["ratings", "feedback", "analysis"],
                "examples": ["Analyze bus service ratings", "Show service feedback", "Compare bus services"],
                "inputModes": DEFAULT_INPUT_MODES,
                "outputModes": DEFAULT_OUTPUT_MODES
            }
        ]
    }

# Healthcheck endpoint
@app.get("/health")
async def health():
    # Test RedBus API connection
    operator_id = os.getenv("OPERATOR_ID", "22")
    source_id = os.getenv("SOURCE_ID", "122")
    dest_id = os.getenv("DEST_ID", "124")
    
    services_data = fetch_bus_ratings(operator_id, source_id, dest_id)
    if services_data is None:
        return {"status": "error", "message": "RedBus API connection failed"}
    
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8010"))
    
    uvicorn.run(app, host=host, port=port) 