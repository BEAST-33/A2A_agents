from fastapi import FastAPI, Request
from datetime import datetime
import logging
import json
import requests
from typing import Dict, Optional, List
import re

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Configuration for the three agents
ROUTE_AGENT_URL = "http://localhost:8053/tasks/send"
REVIEW_AGENT_URL = "http://localhost:8021/tasks/send"
SEAT_AGENT_URL = "http://localhost:8028/tasks/send"

def make_a2a_response(
    request_id: str,
    task_id: str,
    state: str,
    message_text: str,
    combined_data: Optional[Dict] = None
) -> Dict:
    """Create a standardized A2A response."""
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    
    if state == "completed" and combined_data is not None:
        result["artifacts"] = [
            {
                "parts": [
                    {"type": "text", "text": json.dumps(combined_data, indent=2)}
                ],
                "index": 0
            }
        ]
    else:
        result["artifacts"] = [
            {
                "parts": [
                    {"type": "text", "text": message_text}
                ],
                "index": 0
            }
        ]
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

def call_agent(url: str, payload: Dict) -> Optional[Dict]:
    """Make a call to an agent and return its response."""
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error calling agent at {url}: {e}")
        return None

def extract_route_id_from_route_response(route_response: Dict) -> Optional[str]:
    """Extract route ID from the route agent's response."""
    try:
        if route_response and "result" in route_response:
            artifacts = route_response["result"].get("artifacts", [])
            if artifacts and len(artifacts) > 0:
                text = artifacts[0].get("parts", [{}])[0].get("text", "")
                # Look for route ID in the text
                match = re.search(r'Route ID: (\d+)', text)
                if match:
                    return match.group(1)
    except Exception as e:
        logging.error(f"Error extracting route ID: {e}")
    return None

@app.post("/")
async def root(request: Request):
    return await process_workflow(request)

@app.post("/tasks/send")
async def process_workflow(request: Request):
    payload = await request.json()
    logging.info(f"Received payload: {payload}")
    
    # Extract task and request IDs
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    
    # Step 1: Call Route Agent
    route_response = call_agent(ROUTE_AGENT_URL, payload)
    if not route_response:
        return make_a2a_response(
            request_id,
            task_id,
            "failed",
            "Failed to get response from route agent."
        )
    
    # Extract route ID from route agent's response
    route_id = extract_route_id_from_route_response(route_response)
    if not route_id:
        return make_a2a_response(
            request_id,
            task_id,
            "failed",
            "Could not extract route ID from route agent's response."
        )
    
    # Step 2: Call Review Agent with route ID
    review_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "params": {
            "id": task_id,
            "message": {
                "parts": [{"type": "text", "text": f"Get review for route {route_id}"}]
            }
        }
    }
    review_response = call_agent(REVIEW_AGENT_URL, review_payload)
    if not review_response:
        return make_a2a_response(
            request_id,
            task_id,
            "failed",
            "Failed to get response from review agent."
        )
    
    # Step 3: Call Seat Availability Agent with route ID
    seat_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "params": {
            "id": task_id,
            "message": {
                "parts": [{"type": "text", "text": f"Get seat availability for route {route_id}"}]
            }
        }
    }
    seat_response = call_agent(SEAT_AGENT_URL, seat_payload)
    if not seat_response:
        return make_a2a_response(
            request_id,
            task_id,
            "failed",
            "Failed to get response from seat availability agent."
        )
    
    # Combine all responses
    combined_data = {
        "route_info": route_response.get("result", {}),
        "review_info": review_response.get("result", {}),
        "seat_info": seat_response.get("result", {})
    }
    
    return make_a2a_response(
        request_id,
        task_id,
        "completed",
        "Successfully retrieved route, review, and seat availability information.",
        combined_data=combined_data
    )

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")
    
    return {
        "name": "Bus Information Workflow Agent",
        "description": "Orchestrates a workflow between route, review, and seat availability agents to provide comprehensive bus information.",
        "url": "http://localhost:8030/",
        "version": "1.0.0",
        "provider": {
            "organization": "redBus",
            "url": "https://redbus.in"
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text", "text/plain", "application/json"],
        "defaultOutputModes": ["text", "text/plain", "application/json"],
        "skills": [
            {
                "id": "bus_information_workflow",
                "name": "Bus Information Workflow",
                "description": "Sequentially retrieves route information, reviews, and seat availability for a bus route.",
                "tags": ["bus", "workflow", "orchestration", "route", "review", "seat"],
                "examples": [
                    "Get complete information for a bus from Bangalore to Chennai",
                    "Show me route details, reviews and seat availability for Delhi to Mumbai"
                ],
                "inputModes": ["text", "text/plain", "application/json"],
                "outputModes": ["text", "text/plain", "application/json"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "object",
                            "properties": {
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": ["text"]},
                                            "text": {"type": "string", "description": "The route description to search for."}
                                        },
                                        "required": ["type", "text"]
                                    }
                                }
                            },
                            "required": ["parts"]
                        }
                    },
                    "required": ["message"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "route_info": {
                            "type": "object",
                            "description": "Route information from the route agent."
                        },
                        "review_info": {
                            "type": "object",
                            "description": "Review information from the review agent."
                        },
                        "seat_info": {
                            "type": "object",
                            "description": "Seat availability information from the seat agent."
                        }
                    },
                    "required": ["route_info", "review_info", "seat_info"]
                }
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("workflow-orchestrator-agent:app", host="localhost", port=8030, reload=True) 