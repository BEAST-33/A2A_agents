from fastapi import FastAPI, Request
from datetime import datetime
import logging
import json
import re
from collections import defaultdict
import requests
from typing import Optional, List, Dict

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

# --- Seat availability functions (from your original code) ---

def get_seat_availability_with_date(route_id: str, journey_date: str) -> dict:
    try:
        api_url = f"http://channels.omega.redbus.in:8001/IASPublic/getRealTimeUpdate/{route_id}/{journey_date}"

        all_seat_data = []
        available_seats_count = 0
        fare_availability_groups = defaultdict(lambda: {'count': 0, 'currency': 'INR', 'seat_numbers': []})
        total_seats_from_api = None
        error_message = None
        print("api url => ",api_url)
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        total_seats_from_api = data.get('totalSeats')
        if total_seats_from_api is not None:
            try:
                total_seats_from_api = int(total_seats_from_api)
            except (ValueError, TypeError):
                error_message = f"Warning: 'totalSeats' field invalid: {total_seats_from_api}"
                total_seats_from_api = None
        else:
            error_message = "'totalSeats' field missing in API response."

        seat_status_list = data.get('seatStatus', [])
        if not isinstance(seat_status_list, list):
            error_message = "'seatStatus' field is not a list."
            seat_status_list = []
        elif not seat_status_list:
            error_message = "'seatStatus' list is empty."

        for seat_data in seat_status_list:
            if isinstance(seat_data, dict):
                seat_static = seat_data.get('seatStatic', {})
                st_volatile = seat_data.get('stVolatile', {})
                if isinstance(seat_static, dict) and isinstance(st_volatile, dict):
                    seat_number = seat_static.get('no') or st_volatile.get('no')
                    seat_availability = st_volatile.get('stAv')
                    fare_info = st_volatile.get('fare', {})
                    seat_type = fare_info.get('seatType')
                    seat_amount = fare_info.get('amount')
                    currency_type = fare_info.get('currencyType', 'INR')

                    seat_info = {
                        'seatNumber': seat_number,
                        'availabilityStatus': seat_availability,
                        'seatType': seat_type,
                        'amount': seat_amount,
                        'currency': currency_type
                    }
                    all_seat_data.append(seat_info)

                    if seat_availability == 'AVAILABLE':
                        available_seats_count += 1
                        if seat_amount is not None:
                            try:
                                fare_key = float(seat_amount)
                                fare_availability_groups[fare_key]['count'] += 1
                                fare_availability_groups[fare_key]['currency'] = currency_type
                                fare_availability_groups[fare_key]['seat_numbers'].append(seat_number)
                            except (ValueError, TypeError):
                                pass
            else:
                continue

    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception):
        error_message = "We couldn't get the seat information for this service. Please visit redbus.in for details."

    return {
        'available_seats': all_seat_data,
        'fare_groups': fare_availability_groups,
        'available_seats_count': available_seats_count,
        'total_seats': total_seats_from_api,
        'error': error_message
    }


def get_seat_availability(route_id: str) -> dict:
    journey_date = datetime.today().strftime('%Y-%m-%d')
    return get_seat_availability_with_date(route_id, journey_date)

# --- A2A response builder ---

def make_a2a_response(
        request_id: str,
        task_id: str,
        state: str,
        message_text: str,
        seat_data: Optional[dict] = None,
        route_id: Optional[str] = None
) -> Dict:
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    if state == "completed" and seat_data is not None:
        # Serialize seat data (you can customize what to include)
        artifact_text = json.dumps({
            "route_id": route_id,
            "available_seats_count": seat_data.get('available_seats_count'),
            "total_seats": seat_data.get('total_seats'),
            "available_seats": seat_data.get('available_seats')
        }, indent=2)
        result["artifacts"] = [
            {
                "parts": [
                    {"type": "text", "text": artifact_text}
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


# --- FastAPI endpoints ---

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

    # Extract route ID (digits) from user text
    match = re.search(r'\d+', user_text)
    if match:
        route_id = match.group()
    else:
        error_message = "No valid route ID found in the input."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

    seat_data = get_seat_availability(route_id)
    if seat_data['error']:
        return make_a2a_response(request_id, task_id, "failed", seat_data['error'])

    return make_a2a_response(
        request_id,
        task_id,
        "completed",
        "Seat availability fetched successfully.",
        seat_data=seat_data,
        route_id=route_id
    )


@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)


@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")
    return {
        "name": "Seat Availability Agent",
        "description": "Agent that fetches seat availability for a given bus route ID.",
        "url": "http://localhost:8028/",
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
                "id": "get_seat_availability",
                "name": "Get Seat Availability",
                "description": "Fetches seat availability for a given bus route ID from RedBus API.",
                "tags": ["bus", "seat", "availability", "redbus"],
                "examples": [
                    "Show me available seats for route 12345.",
                    "Get seat availability for bus route 67890."
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
                                            "text": {"type": "string", "description": "The bus route ID to fetch seat availability for."}
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
                        "route_id": {
                            "type": "string",
                            "description": "The bus route ID that was queried."
                        },
                        "available_seats_count": {
                            "type": "integer",
                            "description": "Number of available seats."
                        },
                        "total_seats": {
                            "type": ["integer", "null"],
                            "description": "Total seats on the bus."
                        },
                        "available_seats": {
                            "type": "array",
                            "description": "List of available seats with details."
                        }
                    },
                    "required": ["route_id", "available_seats_count", "total_seats", "available_seats"]
                }
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("seat-availability-a2a-agent:app", host="localhost", port=8028, reload=True)
