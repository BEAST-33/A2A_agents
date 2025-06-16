# app.py (FINAL - No Mock Data, Operator ID from UI)

from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime
import logging
import sys

from red_deals_agent import RedDealsAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---
OLLAMA_API_URL = "http://10.166.8.126:11434/api/chat"
OLLAMA_MODEL_NAME = "mistral:7b-instruct"

# --- Initialize Agents ---
# Initializing RedDealsAgent without mock data parameters
red_deals_agent = RedDealsAgent()
logger.warning("RedDealsAgent initialized")

# --- Define Available Tools/Functions for Ollama ---
AVAILABLE_TOOLS = {
    "get_red_deals_data": red_deals_agent.get_red_deals_data,
}

# --- Ollama Interaction Function ---

def call_ollama(messages: list) -> dict:
    """
    Sends messages to the Ollama instance and returns the full JSON response.
    """
    logger.warning(f"Calling Ollama with {len(messages)} messages")
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1000
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        logger.warning("Successfully received response from Ollama")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Ollama response timed out")
        return {"error": "Ollama response timed out. It might be too busy or the response is too slow."}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get response from Ollama: {e}")
        return {"error": f"Could not get response from Ollama. Details: {e}"}

# --- Orchestration Logic (called by Flask API) ---

def orchestrate_query(user_query: str, session_operator_id: int) -> str:
    """
    Orchestrates the interaction between the user, Ollama, and the RedDealsAgent.
    The session_operator_id is provided by the UI.
    """
    logger.warning(f"Starting orchestration for query: {user_query} with operator_id: {session_operator_id}")
    
    system_prompt = f"""
You are an intelligent assistant named RedBusBot that can provide information about RedBus deals.
Your capabilities include fetching RedBus campaign details using the `get_red_deals_data` tool.

**Crucial Instructions:**
1.  **Always use the `get_red_deals_data` tool** if the user asks for information about deals, offers, campaigns, or promotions.
2.  **Do NOT invent or hallucinate any deal details.** All information about deals MUST come ONLY from the output of the `get_red_deals_data` tool.
3.  If the tool returns "No deals found" or similar, clearly state that no relevant deals could be found. Do not make up alternatives.
4.  If a user asks a question that is clearly outside the scope of RedBus deals (e.g., "What's the capital of France?"), respond naturally that you can only assist with RedBus deals.

Here is the description of the tool you can use:
Tool Name: `get_red_deals_data`
Description: Fetches and prepares RedBus campaign data, focusing on deals with discounts.
Parameters:
  - `operator_id` (integer, required): The ID of the operator. **This will be provided by the user in the UI, so do NOT try to infer or generate it yourself. Always use the `operator_id` from the context if it's available, otherwise default to {session_operator_id}.**
  - `is_international` (boolean, required): Set to `true` if international deals are requested, `false` otherwise.
  - `filter_keyword` (string, optional): A keyword to filter deals by their title or description (e.g., "cashback", "holiday").
  - `include_expired` (boolean, optional): Set to `true` if the user explicitly asks for expired deals or "all deals", otherwise `false` (default: only active deals).

To use the tool, respond ONLY with a JSON object in the following format:
{{"tool_call": {{"name": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}}}
For example: {{"tool_call": {{"name": "get_red_deals_data", "parameters": {{"operator_id": {session_operator_id}, "is_international": false, "filter_keyword": "diwali", "include_expired": false}}}}}}
If no specific operator_id is mentioned by the user in their query that would override the current session's `operator_id`, **always use {session_operator_id}**.
If no filter_keyword is mentioned but deals are requested, do not include `filter_keyword` in the parameters.
If the user asks for "all deals" or "past deals", set `include_expired` to `true`. Otherwise, it should be `false`.

After successfully getting data from the tool, process the data and provide a concise, user-friendly summary of the deals based *only* on the provided tool output.
Mention the current date to provide context for deal validity: {datetime.now().strftime('%B %d, %Y')}.
If you cannot find relevant deals or cannot use the tool, respond naturally to the user, reiterating that you can only provide information based on available RedBus deals data.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    ollama_response = call_ollama(messages)
    logger.warning("Received initial Ollama response")

    if ollama_response.get("error"):
        logger.error(f"Ollama returned an error: {ollama_response['error']}")
        return f"Sorry, I encountered an error communicating with Ollama: {ollama_response['error']}"

    ollama_content = ollama_response["message"]["content"]
    logger.warning("Processing Ollama content")
    
    try:
        response_json = json.loads(ollama_content)
        if "tool_call" in response_json:
            tool_call = response_json["tool_call"]
            tool_name = tool_call.get("name")
            tool_params = tool_call.get("parameters", {})

            if tool_name in AVAILABLE_TOOLS:
                logger.warning(f"Executing tool '{tool_name}' with params: {tool_params}")
                
                tool_function = AVAILABLE_TOOLS[tool_name]
                
                # If Ollama didn't specify operator_id, use the one from the session (user input)
                tool_params.setdefault('operator_id', session_operator_id)
                tool_params.setdefault('is_international', False)
                tool_params.setdefault('include_expired', False)

                tool_result = tool_function(**tool_params)
                logger.warning(f"Tool '{tool_name}' execution completed")

                messages.append({"role": "tool", "content": tool_result})
                messages.append({"role": "user", "content": "Based on the tool output, please answer the original query concisely and ONLY use information from the provided tool output. Do not add external details or invent deals."})
                
                final_ollama_response = call_ollama(messages)
                logger.warning("Received final Ollama response")

                if final_ollama_response.get("error"):
                    logger.error(f"Error during final interpretation: {final_ollama_response['error']}")
                    return f"Sorry, I encountered an error during final interpretation: {final_ollama_response['error']}"
                
                return final_ollama_response["message"]["content"]
            else:
                logger.error(f"Unknown tool requested: {tool_name}")
                return f"Ollama requested an unknown tool: {tool_name}. Cannot proceed."
    except json.JSONDecodeError:
        logger.error("Failed to parse Ollama response as JSON")
        return ollama_content
    except Exception as e:
        logger.error(f"Unexpected error during orchestration: {e}")
        return f"An unexpected error occurred: {e}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main chatbot HTML page."""
    logger.warning("Serving index page")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot interaction."""
    user_message = request.json.get('message')
    user_operator_id = request.json.get('operator_id')

    if not user_message:
        logger.warning("Empty message received")
        return jsonify({"response": "Please enter a message."}), 400
    
    # Ensure operator_id is an integer, default if not provided or invalid
    if user_operator_id is None:
        user_operator_id = 15926
    try:
        user_operator_id = int(user_operator_id)
        if user_operator_id <= 0:
            user_operator_id = 15926
    except ValueError:
        user_operator_id = 15926

    logger.warning(f"Processing chat request - Message: '{user_message}', Operator ID: {user_operator_id}")

    bot_response = orchestrate_query(user_message, user_operator_id)
    
    logger.warning("Sending response back to UI")
    return jsonify({"response": bot_response})

# --- Main application entry point ---
if __name__ == '__main__':
    logger.warning("Starting Flask application")
    app.run(debug=True, port=5000)