from fastapi import FastAPI, Request
from datetime import datetime
import logging
import json
import re
import clickhouse_connect

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

def make_a2a_response(request_id, task_id, state, message_text, review=None, route_id=None):
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    if state == "completed" and review is not None:
        result["artifacts"] = [
            {
                "parts": [
                    {"type": "text", "text": json.dumps({"review": review, "route_id": route_id})}
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

def get_bus_review(route_id, database="UGC"):
    try:
        client = clickhouse_connect.get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        query = f"""
            SELECT rv.Review
            FROM UGC.UserReviews rv
            WHERE rv.Id IN (
                SELECT r.Id
                FROM UGC.UserRatings r
                WHERE r.RouteID = {route_id}
            )"""

        print("Executing query:", query)
        result = client.query(query, parameters={"bus_id": route_id}).result_rows
        print("Query response:", result)
        if result:
            return result  # Only the review
        return None
    except Exception as e:
        logging.error(f"Error fetching review from ClickHouse: {e}")
        return None



@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logging.debug(f"Received payload: {payload}")
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    message = params.get("message", {})
    parts = message.get("parts", [])
    if isinstance(parts, list) and parts:
        user_text = parts[0].get("text", "") if isinstance(parts[0], dict) else str(parts[0])
    elif isinstance(parts, dict):
        user_text = parts.get("text", "")
    else:
        user_text = ""

    # Extract route ID from text
    match = re.search(r'\d+', user_text)
    if match:
        route_id = match.group()
    else:
        error_message = "No valid route ID found in the input."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

    review = get_bus_review(route_id)
    if review:
        output = make_a2a_response(request_id, task_id, "completed", "Success", review=review, route_id=route_id)
        print("Output ==> ",output)
        return output
    else:
        error_message = f"No review found for RouteID {route_id}."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")
    return {
        "name": "Bus Review Agent",
        "description": "Agent that fetches a user review for a given bus route.",
        "url": "http://localhost:8021/",
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
                "id": "get_bus_review",
                "name": "Get Bus Review",
                "description": "Fetches a sample user review for a given bus route ID from ClickHouse.",
                "tags": ["bus", "review", "clickhouse", "feedback"],
                "examples": [
                    "Get a review for route ID '123'.",
                    "Show me a user review for bus route 456."
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
                                            "text": {"type": "string", "description": "The bus route ID to fetch the review for."}
                                        },
                                        "required": ["type", "text"]
                                    },
                                    "description": "The list of message parts, with at least one containing the route ID."
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
                        "review": {
                            "type": "string",
                            "description": "A sample user review for the requested route."
                        },
                        "route_id": {
                            "type": "string",
                            "description": "The bus route ID that was queried."
                        }
                    },
                    "required": ["review", "route_id"]
                }
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reviews-a2a-agent:app", host="localhos", port=8021, reload=True)
