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

def preprocess_deals_data(deals_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pre-process deals data to extract relevant information and categorize by status
    """
    current_date = datetime.now(UTC)
    processed_deals = {
        "active": [],
        "expired": [],
        "upcoming": [],
        "on_hold": []
    }
    
    for deal in deals_data:
        # Extract only relevant fields
        processed_deal = {
            "campaignId": deal.get("campaignId"),
            "campaignDesc": deal.get("campaignDesc"),
            "campaignType": deal.get("campaignType"),
            "discountValue": deal.get("discountValue"),
            "discountPercent": deal.get("discountPercent"),
            "minTicketVal": deal.get("minTicketVal"),
            "maxDiscountVal": deal.get("maxDiscountVal"),
            "daysCSV": deal.get("daysCSV"),
            "startDate": deal.get("startDate"),
            "endDate": deal.get("endDate"),
            "status": deal.get("status"),
            "routeIDCSV": deal.get("routeIDCSV"),
            "RouteList": deal.get("RouteList", [])
        }
        
        # Categorize deal
        if deal.get("status") == "ACTIVE":
            processed_deals["active"].append(processed_deal)
        elif deal.get("status") == "EXPIRED":
            processed_deals["expired"].append(processed_deal)
        elif deal.get("status") == "ONHOLD":
            processed_deals["on_hold"].append(processed_deal)
        else:
            # Check if it's upcoming
            try:
                start_date_str = deal.get("startDate", "")
                if start_date_str:
                    # Try parsing with milliseconds first
                    try:
                        start_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # If that fails, try without milliseconds
                        start_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%S")
                    
                    # Make it timezone-aware
                    start_date = start_date.replace(tzinfo=UTC)
                    if start_date > current_date:
                        processed_deals["upcoming"].append(processed_deal)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing date for deal {deal.get('campaignId')}: {e}")
                continue
    
    # Sort active deals by discount value/percentage
    processed_deals["active"].sort(
        key=lambda x: x.get("discountValue", 0) + (x.get("discountPercent", 0) * 100),
        reverse=True
    )
    
    # Limit the number of deals in each category
    for category in processed_deals:
        processed_deals[category] = processed_deals[category][:300]  # Keep only top 10 deals per category
    
    return processed_deals

def parse_date_flexible(date_str):
    """Parse date string that may or may not have milliseconds"""
    if not date_str:
        return None
    
    # Try with milliseconds first
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        # If that fails, try without milliseconds
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    
    return dt.replace(tzinfo=UTC)

def search_deals(deals_data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Search and filter deals based on user query
    """
    current_date = datetime.now(UTC)
    filtered_deals = []
    
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    
    for deal in deals_data:
        # Check if deal is active and within valid date range
        try:
            # Parse start date with milliseconds
            start_date = parse_date_flexible(deal.get("startDate", ""))
            end_date = parse_date_flexible(deal.get("endDate", ""))
            
            # Skip if deal is not active or not in valid date range
            if deal.get("status") != "ACTIVE" or not (start_date <= current_date <= end_date):
                continue
                
            # Extract relevant fields for searching
            campaign_desc = deal.get("campaignDesc", "").lower()
            route_list = deal.get("RouteList", [])
            days_csv = deal.get("daysCSV", "").lower()
            campaign_type = deal.get("campaignType", "").lower()
            
            # Check if any search criteria matches
            matches = False
            
            # Search in campaign description
            if query in campaign_desc:
                matches = True
                
            # Search in route information
            for route in route_list:
                source = route.get("Source", "").lower()
                destination = route.get("Destination", "").lower()
                if query in source or query in destination:
                    matches = True
                    break
                    
            # Search in days
            if query in days_csv:
                matches = True
                
            # Search in campaign type
            if query in campaign_type:
                matches = True
                
            # If no specific search criteria, include all active deals
            if not any(word in query for word in ["route", "day", "type", "deal", "offer", "discount"]):
                matches = True
                
            if matches:
                filtered_deals.append({
                    "campaignId": deal.get("campaignId"),
                    "campaignDesc": deal.get("campaignDesc"),
                    "campaignType": deal.get("campaignType"),
                    "discountValue": deal.get("discountValue"),
                    "discountPercent": deal.get("discountPercent"),
                    "minTicketVal": deal.get("minTicketVal"),
                    "maxDiscountVal": deal.get("maxDiscountVal"),
                    "daysCSV": deal.get("daysCSV"),
                    "startDate": deal.get("startDate"),
                    "endDate": deal.get("endDate"),
                    "status": deal.get("status"),
                    "routeIDCSV": deal.get("routeIDCSV"),
                    "RouteList": deal.get("RouteList", []),
                    "days_remaining": (end_date - current_date).days
                })
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing deal {deal.get('campaignId')}: {e}")
            continue
    
    # Sort by discount value/percentage
    filtered_deals.sort(
        key=lambda x: x.get("discountValue", 0) + (x.get("discountPercent", 0) * 100),
        reverse=True
    )
    
    return filtered_deals
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

def filter_valid_deals(campaigns: List[Dict[str, Any]], today_date: str = None) -> List[Dict[str, Any]]:
    """
    Filters out invalid deals based on today's date.
    
    Args:
        campaigns: List of campaign dictionaries
        today_date: Optional date string in format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'
                   If None, uses current date
    
    Returns:
        List of valid campaigns that are currently active and within date range
    """
    
    # Get today's date
    if today_date is None:
        today = datetime.now()
    else:
        # Handle different date formats
        try:
            if 'T' in today_date:
                today = datetime.fromisoformat(today_date.replace('Z', '+00:00'))
            else:
                today = datetime.strptime(today_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {today_date}. Use 'YYYY-MM-DD' or ISO format")
    
    valid_campaigns = []
    
    for campaign in campaigns:
        try:
            # Parse start and end dates
            start_date_str = campaign.get('startDate', '')
            end_date_str = campaign.get('endDate', '')
            
            if not start_date_str or not end_date_str:
                print(f"Warning: Campaign {campaign.get('campaignId')} missing date information")
                continue
            
            # Parse dates (handle different formats)
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            
            # Check if campaign is within valid date range
            is_date_valid = start_date <= today <= end_date
            
            # Check if status is not EXPIRED (additional validation)
            status = campaign.get('status', '').upper()
            is_status_valid = status in ['ACTIVE', 'ONHOLD']  # Consider ONHOLD as potentially valid
            
            # Additional checks
            campaign_id = campaign.get('campaignId', 'Unknown')
            campaign_desc = campaign.get('campaignDesc', 'No description')
            
            if is_date_valid and is_status_valid:
                valid_campaigns.append(campaign)
                print(f"✓ Valid: {campaign_desc} (ID: {campaign_id}) - Status: {status}")
            else:
                reason = []
                if not is_date_valid:
                    if today < start_date:
                        reason.append("not started yet")
                    elif today > end_date:
                        reason.append("expired by date")
                if not is_status_valid:
                    reason.append(f"status is {status}")
                
                print(f"✗ Invalid: {campaign_desc} (ID: {campaign_id}) - {', '.join(reason)}")
                
        except Exception as e:
            print(f"Error processing campaign {campaign.get('campaignId', 'Unknown')}: {str(e)}")
            continue
    
    return valid_campaigns

def get_deal_summary(campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a summary of deals by status and type
    """
    summary = {
        'total_campaigns': len(campaigns),
        'by_status': {},
        'by_type': {},
        'valid_deals': 0,
        'expired_deals': 0,
        'upcoming_deals': 0
    }
    
    today = datetime.now()
    
    for campaign in campaigns:
        # Count by status
        status = campaign.get('status', 'Unknown')
        summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
        
        # Count by type
        campaign_type = campaign.get('campaignType', 'Unknown')
        summary['by_type'][campaign_type] = summary['by_type'].get(campaign_type, 0) + 1
        
        # Count by date validity
        try:
            start_date = datetime.fromisoformat(campaign.get('startDate', '').replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(campaign.get('endDate', '').replace('Z', '+00:00'))
            
            if today < start_date:
                summary['upcoming_deals'] += 1
            elif today > end_date:
                summary['expired_deals'] += 1
            else:
                summary['valid_deals'] += 1
        except:
            pass
    
    return summary

# Example usage and testing
if __name__ == "__main__":
    # Sample data (your campaigns)
    sample_campaigns = [
        {
            "campaignId": 607507,
            "campaignType": "SL:PERCENT",
            "campaignDesc": "15% Extra OFF",
            "status": "ONHOLD",
            "startDate": "2025-06-16T17:40:27.061",
            "endDate": "2025-07-16T23:59:59"
        },
        {
            "campaignId": 606705,
            "campaignType": "FLAT",
            "campaignDesc": "INR-100 Extra OFF",
            "status": "EXPIRED",
            "startDate": "2025-06-13T15:20:10.561",
            "endDate": "2025-06-14T23:59:59"
        },
        {
            "campaignId": 606255,
            "campaignType": "RETURN_TRIP",
            "campaignDesc": "10% Extra OFF",
            "status": "ACTIVE",
            "startDate": "2025-06-12T10:50:09.803",
            "endDate": "2025-07-31T23:59:59"
        }
    ]
    
    print("=" * 50)
    print("FILTERING VALID DEALS")
    print("=" * 50)
    
    # Filter valid deals
    valid_deals = filter_valid_deals(sample_campaigns)
    
    print(f"\nFound {len(valid_deals)} valid deals out of {len(sample_campaigns)} total campaigns")
    
    print("\n" + "=" * 50)
    print("CAMPAIGN SUMMARY")
    print("=" * 50)
    
    # Get summary
    summary = get_deal_summary(sample_campaigns)
    print(f"Total campaigns: {summary['total_campaigns']}")
    print(f"Valid deals (by date): {summary['valid_deals']}")
    print(f"Expired deals (by date): {summary['expired_deals']}")
    print(f"Upcoming deals: {summary['upcoming_deals']}")
    print(f"Status breakdown: {summary['by_status']}")
    print(f"Type breakdown: {summary['by_type']}")

# Additional utility functions
def filter_by_discount_type(campaigns: List[Dict[str, Any]], discount_type: str = "PERCENT") -> List[Dict[str, Any]]:
    """Filter campaigns by discount type (PERCENT, FLAT, etc.)"""
    return [c for c in campaigns if discount_type.upper() in c.get('campaignType', '').upper()]

def filter_by_minimum_discount(campaigns: List[Dict[str, Any]], min_percent: float = 0, min_flat: float = 0) -> List[Dict[str, Any]]:
    """Filter campaigns by minimum discount amount"""
    filtered = []
    for campaign in campaigns:
        discount_percent = campaign.get('discountPercent', 0)
        discount_value = campaign.get('discountValue', 0)
        
        if discount_percent >= min_percent or discount_value >= min_flat:
            filtered.append(campaign)
    
    return filtered

def get_campaigns_ending_soon(campaigns: List[Dict[str, Any]], days: int = 7) -> List[Dict[str, Any]]:
    """Get campaigns ending within specified number of days"""
    from datetime import timedelta
    
    today = datetime.now()
    cutoff_date = today + timedelta(days=days)
    
    ending_soon = []
    for campaign in campaigns:
        try:
            end_date = datetime.fromisoformat(campaign.get('endDate', '').replace('Z', '+00:00'))
            if today <= end_date <= cutoff_date:
                ending_soon.append(campaign)
        except:
            continue
    
    return ending_soon

def generate_chat_response(user_message: str, deals_data: Optional[Dict[str, Any]] = None) -> tuple[str, dict]:
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
        
        # If deals data is provided, search and filter based on user query
        enhanced_message = user_message
        if deals_data:
            filtered_deals = search_deals(deals_data, user_message)
            current_date = datetime.now(UTC).strftime("%Y-%m-%d")
            deals_summary = {
                "current_date": current_date,
                "total_deals": len(deals_data),
                "matching_deals": len(filtered_deals),
                "deals": filtered_deals
            }
            deals_json = json.dumps(deals_summary, indent=2)
            enhanced_message = f"{user_message}\n\nHere are the relevant deals based on your query:\n{deals_json}"
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