import os
import logging
import json
import urllib.parse
import http.client
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider

import spacy # Import spacy

logger = logging.getLogger("my_app_logger")
logger.setLevel(logging.INFO)

# --- Configuration ---
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "localhost").split(",")
CASSANDRA_PORT = 9042
CASSANDRA_USER = "cassandra"
CASSANDRA_PASSWORD = "cassandra"
DEFAULT_KEYSPACE = "ai_bus_info"
DEFAULT_TABLE = "rb_route_embedding"
DEFAULT_EMBEDDING_DIM = 768

NOMIC_API_URL = os.environ.get("NOMIC_API_URL", "http://10.166.8.126:11434/api/embed")
NOMIC_MODEL_NAME = os.environ.get("NOMIC_MODEL_NAME", "nomic-embed-text:latest")

app = FastAPI()
logging.basicConfig(level=logging.INFO)
# Map language codes to spaCy model names
#python -m spacy download en_core_web_sm
SPACY_MODELS = dict(en='en_core_web_sm', fr='fr_core_news_sm', de='de_core_news_sm', es='es_core_news_sm',
                    zh='zh_core_web_sm', xx='xx_ent_wiki_sm', en_lg ='en_core_web_lg')

# Cache loaded models
loaded_models = {}

def get_nlp(lang_code):
    model_name = SPACY_MODELS.get(lang_code, SPACY_MODELS['en_lg'])
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded successfully for language '{lang_code}'.")
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{model_name}': {e}. Falling back to multilingual model.")
            if model_name != SPACY_MODELS['xx']:
                return get_nlp('xx')
            loaded_models[model_name] = None
    return loaded_models[model_name]

# --- Helper: Nomic Embedding ---
def make_post_request(url, payload, headers):
    try:
        parsed_url = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80)
        conn.request("POST", parsed_url.path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        conn.close()
        return json.loads(data)
    except Exception as e:
        logging.error(f"Error making HTTP request to {url}: {e}")
        return None

def generate_nomic_embeddings(text, api_url=NOMIC_API_URL, model_name=NOMIC_MODEL_NAME):
    payload = {"model": model_name, "input": str(text)}
    headers = {'Content-Type': 'application/json'}
    result = make_post_request(api_url, payload, headers)
    if not result:
        return None
    embeddings = result.get("embeddings", [])
    if isinstance(embeddings, list) and len(embeddings) > 0:
        return embeddings[0]
    return None

# --- Helper: Cassandra ANN ---
def get_session() -> Optional[Session]:
    try:
        auth_provider = PlainTextAuthProvider(
            username=CASSANDRA_USER, password=CASSANDRA_PASSWORD
        )
        cluster = Cluster(
            contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider
        )
        session = cluster.connect()
        return session
    except Exception as e:
        logging.error(f"Failed to connect to Cassandra: {e}")
        return None

def get_table_columns(session: Session, keyspace: str, table: str) -> Optional[List[str]]:
    try:
        logger.info(f"About to query system_schema.columns for keyspace='{keyspace}', table='{table}'")
        rows = session.execute(
            "SELECT column_name FROM system_schema.columns WHERE keyspace_name=%s AND table_name=%s",
            (keyspace, table),
        )
        rows_list = list(rows)  # Materialize the iterator
        logger.info(f"Raw rows received from system_schema.columns: {rows_list}")
        return [row.column_name for row in rows_list]
    except Exception as e:
        logger.error(f"Failed to retrieve columns for {keyspace}.{table}: {e}")
        return None

def get_embedding_column(columns: List[str]) -> Optional[str]:
    for column in columns:
        if "embedding" in column.lower():
            return column
    return None


logger = logging.getLogger(__name__) # Ensure logger is initialized

def process_rows(rows, columns, target_sourcename=None, target_destinationname=None):
    """
    Processes a list of Cassandra rows, directly accessing column values
    using dot notation (e.g., row.column_name) and applying filters
    sequentially based on sourcename and then destinationname.

    Args:
        rows (list): A list of row objects returned from a Cassandra query.
                     These rows are typically accessible via attributes (e.g., row.column_name).
        columns (list): A list of column names. (This parameter is now less critical
                        as columns are accessed by name directly from the row object,
                        but kept for signature consistency).
        target_sourcename (str, optional): The sourcename to filter by.
                                            If None, no filtering on sourcename occurs.
        target_destinationname (str, optional): The destinationname to filter by.
                                                If None, no filtering on destinationname occurs.

    Returns:
        list: A list of the original Cassandra row objects that successfully
              passed all specified filtering criteria.
    """
    results = []
    for row in rows:
        # FIX: Access column values using dot notation (e.g., row.sourcename)
        # Use hasattr() for robust error handling in case a column truly doesn't exist in a row.
        # For SELECT * queries, these attributes should typically always be present.
        current_sourcename = row.sourcename if hasattr(row, 'sourcename') else None
        current_destinationname = row.destinationname if hasattr(row, 'destinationname') else None

        # Step 1: Filter by sourcename (if provided)
        if target_sourcename is not None:
            # Check if attribute exists and if it matches the target
            if current_sourcename is None or current_sourcename != target_sourcename:
                logger.debug(f"Row skipped: Sourcename '{current_sourcename}' doesn't match '{target_sourcename}'")
                continue # Move to the next row immediately

        # Step 2: Filter by destinationname (if provided)
        # This step is only reached if sourcename matched (or no sourcename filter was applied)
        if target_destinationname is not None:
            # Check if attribute exists and if it matches the target
            if current_destinationname is None or current_destinationname != target_destinationname:
                logger.debug(f"Row skipped: Destinationname '{current_destinationname}' doesn't match '{target_destinationname}'")
                continue # Move to the next row immediately

        # If we reach here, the row has passed all active filters.
        # Append the raw row object to the results list.
        results.append(row)

    logger.info(f"Total matches found after filtering: {len(results)}")
    return results

def ann_query_on_cassandra(
        session: Session,
        keyspace: str,
        table: str,
        embedding_column: str,
        embedding: List[float],
        dim: int,
        source_name: Optional[str] = None,
        destination_name: Optional[str] = None,
        top_k: int = 50, # Changed default top_k to 5 as per requirement
) -> Optional[List[Dict]]:
    try:
        # Base query for ANN
        query_parts = [
            f"""SELECT
    routeid,
    arrtime,
    bustype,
    deptime,
    destinationid,
    destinationname,
    destinationstate,
    isseater,
    issleeper,
    journeydurationinmin,
    serviceid,
    servicename,
    slid,
    sourceid,
    sourcename,
    sourcestate,
    travelsname,
    similarity_cosine({embedding_column}, ?) AS cosine_similarity""" # Use ? for prepared statement
        ]
        query_parts.append(f"FROM {keyspace}.{table}")

        where_clauses = []
        bound_params = [embedding] # Always include embedding for ANN
        #
        # if source_name:
        #     where_clauses.append("sourcename = ?")
        #     bound_params.append(source_name)
        # if destination_name:
        #     where_clauses.append("destinationname = ?")
        #     bound_params.append(destination_name)

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        # ANN ORDER BY is crucial and must be separate from WHERE clauses
        query_parts.append(f"ORDER BY {embedding_column} ANN OF ?") # Separate ? for ANN
        bound_params.append(embedding) # Duplicate embedding for ANN ordering

        query_parts.append(f"LIMIT {top_k}")
        # query_parts.append(f"ALLOW FILTERING") # Allow filtering for ANN

        query = " ".join(query_parts)

        logger.info(f"Preparing ANN query: {query}")
        logger.info(f"Bound parameters (excluding embedding values): {bound_params[1:]}") # Log params without large embedding

        prepared = session.prepare(query)
        logger.info("Executing ANN query on Cassandra...")
        rows = session.execute(prepared, bound_params) # Pass bound_params to execute

        columns = [
            'routeid',
            'arrtime',
            'arrtimezone',
            'bustype',
            'deptime',
            'deptimezone',
            'destinationid',
            'destinationname',
            'destinationstate',
            'gds',
            'isseater',
            'issleeper',
            'journeydurationinmin',
            'route_embedding', # Keep route_embedding here as it's part of the table, though not directly returned in SELECT
            'serviceid',
            'servicename',
            'slid',
            'sourceid',
            'sourcename',
            'sourcestate',
            'travelsname',
            'viacity',
            'cosine_similarity' # Add cosine_similarity here as it's returned by the query
        ]

        results = process_rows(rows, columns,source_name, destination_name)
        return results
    except Exception as e:
        logger.error(f"Error during ANN query on {keyspace}.{table}: {e}")
        return None

def extract_source_destination(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts potential source and destination names from free text using spaCy.
    This version tries to prioritize explicit "from X to Y" patterns and then falls back to NER.
    """
    nlp = get_nlp("en_core_web_lg") # Use the full model name as a string
    if nlp is None:
        logger.warning("spaCy model not loaded, cannot extract source/destination.")
        return None, None

    doc = nlp(text)
    source = None
    destination = None

    text_lower = text.lower()

    # --- Strategy 1: Look for explicit "from X to Y" or "X to Y" patterns ---
    # This is often the most reliable when present.

    # Pattern: "from X to Y"
    # Find the most specific "to" after a "from" for departure/arrival.
    from_idx = -1
    to_idx = -1

    # Find the first "from" and the first "to" after it
    first_from_match = None
    first_to_match = None



    # Simple token-based search for "from ... to ..."
    try:
        # Use spaCy's tokenization for more robust splitting
        tokens = [token.lower_ for token in doc]

        from_index = tokens.index("from")
        # Search for "to" *after* "from"
        for i in range(from_index + 1, len(tokens)):
            if tokens[i] == "to":
                to_index = i

                # Try to extract source (between "from" and "to")
                source_candidate_tokens = []
                for j in range(from_index + 1, to_index):
                    if doc[j].ent_type_ in ["GPE", "LOC"]:
                        source_candidate_tokens.append(doc[j].text)
                if source_candidate_tokens:
                    source = " ".join(source_candidate_tokens).title()

                # Try to extract destination (after "to")
                destination_candidate_tokens = []
                for k in range(to_index + 1, len(tokens)):
                    if doc[k].ent_type_ in ["GPE", "LOC"]:
                        destination_candidate_tokens.append(doc[k].text)
                if destination_candidate_tokens:
                    destination = " ".join(destination_candidate_tokens).title()

                # If we found both, we can break early
                if source and destination:
                    break
    except ValueError:
        pass # "from" or "to" not found in the expected sequence

    # If "from X to Y" didn't yield results, try "X to Y" directly for the remaining
    # We prioritize the "from X to Y" as it's more explicit about source/destination.
    if not source and not destination and " to " in text_lower:
        parts = text_lower.split(" to ")
        if len(parts) >= 2:
            # The destination is likely the last GPE/LOC after the final 'to'
            # Let's find GPE/LOCs in the text after the last 'to'
            last_to_idx = text_lower.rfind(" to ")
            if last_to_idx != -1:
                after_to_text = text[last_to_idx + len(" to "):]
                doc_after_to = nlp(after_to_text)
                for ent in doc_after_to.ents:
                    if ent.label_ in ["GPE", "LOC"]:
                        destination = ent.text.title()
                        break # Take the first one as the most likely destination

            # The source is likely the first GPE/LOC before the first 'to'
            first_to_idx = text_lower.find(" to ")
            if first_to_idx != -1:
                before_to_text = text[:first_to_idx]
                doc_before_to = nlp(before_to_text)
                # Iterate in reverse to find the last entity before the first "to"
                # which is often the source. Or just take the first entity if it's there.
                for ent in reversed(doc_before_to.ents): # Iterate in reverse for potentially closer source
                    if ent.label_ in ["GPE", "LOC"]:
                        source = ent.text.title()
                        break


    # --- Strategy 2: Fallback to general GPE/LOC entities if patterns fail ---
    potential_locations_ner = []
    if not source or not destination: # Only fallback if we haven't found both yet
        for ent in doc.ents:
            # print(f"NER: {ent.text} - {ent.label_}") # Debugging
            if ent.label_ in ["GPE", "LOC"]:
                potential_locations_ner.append(ent.text)

        if not source and potential_locations_ner:
            source = potential_locations_ner[0].title()
        if not destination and len(potential_locations_ner) > 1:
            # Try to get the second distinct location if the first was set as source
            # This is still heuristic but better than nothing
            for loc in potential_locations_ner:
                if loc.title() != source: # Make sure it's not the same as source
                    destination = loc.title()
                    break
            # If only one location is found, and it's not set as source, it could be destination
            if not destination and len(potential_locations_ner) == 1 and not source:
                destination = potential_locations_ner[0].title()


    logger.info(f"Extracted Source: '{source}', Destination: '{destination}' from text: '{text}'")
    return source, destination

def get_time_from_datetime_str(dt_input):
    # If the input is already a datetime object, format it directly
    if isinstance(dt_input, datetime):
        return dt_input.strftime("%H:%M")
    
    # If it's a string or 'N/A', proceed with parsing
    if dt_input and dt_input != 'N/A':
        try:
            # Parse the string into a datetime object
            dt_object = datetime.strptime(dt_input, "%Y-%m-%d %H:%M:%S")
            # Format the datetime object to get only the time (HH:MM)
            return dt_object.strftime("%H:%M")
        except ValueError:
            # Handle cases where the string format doesn't match
            return 'N/A'
    return 'N/A'

def make_a2a_response(
        request_id: str,
        task_id: str,
        status: str,
        description: str,
        # FIX: Updated type hint to reflect that 'matches' contains Row objects (or similar)
        # Use 'object' if the exact Row type is not imported/defined.
        matches: Optional[List[object]] = None,
        source_extracted: Optional[str] = None,
        destination_extracted: Optional[str] = None
) -> Dict:
    """
    Constructs a JSON-RPC 2.0 response in the A2A format.

    Args:
        request_id: The ID of the original request.
        task_id: The ID of the task.
        status: The status of the task ("completed", "failed").
        description: A human-readable description of the task status.
        matches: (Optional) The result of the ANN query (list of raw Cassandra Row objects).
        source_extracted: (Optional) The extracted source city.
        destination_extracted: (Optional) The extracted destination city.

    Returns:
        A dictionary representing the JSON-RPC 2.0 response.
    """
    # Compose the main text output
    text_parts = [description]
    if source_extracted:
        text_parts.append(f"Source Extracted: {source_extracted}")
    if destination_extracted:
        text_parts.append(f"Destination Extracted: {destination_extracted}")

    if matches:
        text_parts.append("--- Matches ---")
        # FIX: Access attributes from the raw Row object using getattr()
        for m in matches:
            route_info = (
                f"Route: {getattr(m, 'sourcename', 'N/A')} ({getattr(m, 'sourceid', 'N/A')}) "
                f"to {getattr(m, 'destinationname', 'N/A')} ({getattr(m, 'destinationid', 'N/A')}), "
                f"Dep: {get_time_from_datetime_str(getattr(m, 'deptime', 'N/A'))}, "
                f"Arr: {get_time_from_datetime_str(getattr(m, 'arrtime', 'N/A'))}, "
                f"Travels: {getattr(m, 'travelsname', 'N/A')}, "
                f"Bus Type: {getattr(m, 'bustype', 'N/A')}, "
                f"Seater: {getattr(m, 'isseater', 'N/A')}, Sleeper: {getattr(m, 'issleeper', 'N/A')}, "
                f"Service: {getattr(m, 'servicename', 'N/A')} (ID: {getattr(m, 'serviceid', 'N/A')}), "
                f"Route ID: {getattr(m, 'routeid', 'N/A')}, SLID: {getattr(m, 'slid', 'N/A')}, "
                # Assuming 'cosine_similarity' might also be an attribute on the Row object,
                # or will default to 0 if not present.
                f"Similarity: {getattr(m, 'cosine_similarity', 0):.4f}"
            )
            text_parts.append(route_info)
    else:
        text_parts.append("No matches found.")

    text = "\n".join(text_parts)

    response: Dict = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "id": task_id,
            "status": {
                "state": status,
                "timestamp": datetime.utcnow().isoformat()
            },
            "history": [],
            "artifacts": [
                {
                    "parts": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ],
                    "index": 0
                }
            ]
        }
    }
    logger.info(f"Final response for task {task_id} with status {status}")
    return response


@app.post("/")
async def root(request: Request):
    return await send_task(request)


# --- Main Agent Endpoint ---
@app.post("/tasks/send")
async def send_task(request: Request):
    payload = await request.json()
    logging.info(f"Received payload: {payload}")
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    message = params.get("message", {})
    parts = message.get("parts", [])
    keyspace = params.get("keyspace") or DEFAULT_KEYSPACE
    table = params.get("table") or DEFAULT_TABLE
    dim = int(params.get("dimension", DEFAULT_EMBEDDING_DIM))

    # Step 1: Extract user text
    if isinstance(parts, dict):
        user_text = parts.get("text", "")
    elif isinstance(parts, list) and parts:
        user_text = parts[0].get("text", "") if isinstance(parts[0], dict) else parts[0]
    else:
        user_text = ""

    if not user_text:
        return JSONResponse(
            make_a2a_response(request_id, task_id, "failed", "No text provided."),
            status_code=400
        )

    # Step 2: Extract source and destination from user text
    extracted_source, extracted_destination = extract_source_destination(user_text)

    # Step 3: Generate embedding
    embedding = generate_nomic_embeddings(extracted_source+" to "+extracted_destination)
    if not embedding:
        return JSONResponse(
            make_a2a_response(request_id, task_id, "failed", "Could not generate embedding."),
            status_code=500
        )
    if len(embedding) != dim:
        return JSONResponse(
            make_a2a_response(request_id, task_id, "failed", f"Embedding dimension mismatch: expected {dim}, got {len(embedding)}"),
            status_code=400
        )

    # Step 4: ANN query on Cassandra with optional filtering
    session = get_session()
    if not session:
        return JSONResponse(
            make_a2a_response(request_id, task_id, "failed", "Failed to connect to Cassandra."),
            status_code=500
        )
    try:
        columns = get_table_columns(session, keyspace, table)
        if not columns:
            return JSONResponse(
                make_a2a_response(request_id, task_id, "failed", f"Could not retrieve columns for table '{table}' in keyspace '{keyspace}'."),
                status_code=400
            )
        embedding_column = get_embedding_column(columns)
        if not embedding_column:
            return JSONResponse(
                make_a2a_response(request_id, task_id, "failed", f"No suitable embedding column found in table '{table}'."),
                status_code=400
            )

        # Pass extracted source and destination to the ANN query
        matches = ann_query_on_cassandra(
            session,
            keyspace,
            table,
            embedding_column,
            embedding,
            dim,
            source_name=extracted_source,
            destination_name=extracted_destination,
            top_k=50
        )

        if matches is not None:
            status_desc = "Top 5 ANN matches found."
            if extracted_source or extracted_destination:
                status_desc += " Filtered by detected locations."
            return JSONResponse(make_a2a_response(
                request_id=request_id,
                task_id=task_id,
                status="completed",
                description=status_desc,
                matches=matches,
                source_extracted=extracted_source,
                destination_extracted=extracted_destination))
        else:
            return JSONResponse(
                make_a2a_response(request_id, task_id, "failed", "Error during ANN query."),
                status_code=500
            )
    finally:
        if session:
            session.cluster.shutdown()

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")

    response_content = {
        "name": "Route Agent",
        "description": (
            "Agent that accepts text, extracts potential source and destination, "
            "generates a Nomic embedding, and runs Approximate Nearest Neighbor (ANN) queries on a Cassandra database. "
            "Filters by extracted locations if found, and returns the top 5 most similar matches for the provided text."
        ),
        "url": "http://localhost:8053/",
        "version": "1.0.2", # Increment version due to new features
        "provider": {
            "organization": "redBus",
            "url": "https://redbus.in"
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text, text/plain, application/json"],
        "defaultOutputModes": ["text, text/plain, application/json"],
        "skills": [
            {
                "id": "route agent",
                "name": "route agent",
                "description": (
                    "Accepts user text, extracts potential source and destination cities, "
                    "generates a Nomic embedding, and executes an Approximate Nearest Neighbor (ANN) search "
                    "against a Cassandra database, optionally filtering by the extracted locations. "
                    "Returns the top 5 most similar matches."
                ),
                "tags": ["embedding", "ann", "cassandra", "vector search", "nomic", "text", "nlp", "location"],
                "examples": [
                    "Find similar bus routes for this description: 'Bangalore to Chennai.'",
                    "I want a bus from Delhi to Mumbai.",
                    "Bus routes for Kolkata to Pune, overnight.",
                    "Show me options for Chennai." # Will try to find Chennai as source or dest
                ],
                "inputModes": ["text, text/plain, application/json"],
                "outputModes": ["text, text/plain, application/json"],
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
                                            "text": {"type": "string", "description": "The user text to embed and search for, potentially containing source and destination."}
                                        },
                                        "required": ["type", "text"]
                                    },
                                    "description": "The list of message parts, with at least one containing the text to process."
                                }
                            },
                            "required": ["parts"]
                        },
                        "keyspace": {
                            "type": "string",
                            "description": "Cassandra keyspace to search in.",
                            "default": DEFAULT_KEYSPACE
                        },
                        "table": {
                            "type": "string",
                            "description": "Cassandra table to search in.",
                            "default": DEFAULT_TABLE
                        },
                        "dimension": {
                            "type": "integer",
                            "description": "Expected embedding dimension (should match Cassandra table).",
                            "default": DEFAULT_EMBEDDING_DIM
                        }
                    },
                    "required": ["message"] # Only message is strictly required now with defaults
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "source_extracted": {
                            "type": "string",
                            "nullable": True,
                            "description": "The source location extracted from the user's input, if identified."
                        },
                        "destination_extracted": {
                            "type": "string",
                            "nullable": True,
                            "description": "The destination location extracted from the user's input, if identified."
                        },
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "routeid": {"type": "string"},
                                    "arrtime": {"type": "string", "format": "time", "description": "Arrival time (HH:MM)."},
                                    "bustype": {"type": "string"},
                                    "deptime": {"type": "string", "format": "time", "description": "Departure time (HH:MM)."},
                                    "destinationid": {"type": "integer"},
                                    "destinationname": {"type": "string"},
                                    "destinationstate": {"type": "string"},
                                    "isseater": {"type": "boolean"},
                                    "issleeper": {"type": "boolean"},
                                    "journeydurationinmin": {"type": "integer"},
                                    "serviceid": {"type": "integer"},
                                    "servicename": {"type": "string"},
                                    "slid": {"type": "string"},
                                    "sourceid": {"type": "integer"},
                                    "sourcename": {"type": "string"},
                                    "sourcestate": {"type": "string"},
                                    "travelsname": {"type": "string"},
                                    "cosine_similarity": {"type": "number", "format": "float", "description": "Cosine similarity score for the route."}
                                },
                                "required": ["routeid", "sourcename", "destinationname", "deptime", "arrtime", "cosine_similarity"]
                            },
                            "maxItems": 5,
                            "description": (
                                "An array containing the top 5 most similar data points found in Cassandra, "
                                "optionally filtered by source/destination, ordered by similarity score (descending)."
                            )
                        }
                    },
                    "required": ["matches"]
                }
            }
        ]
    }

    return response_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("route-a2a-agent:app", host="localhost", port=8053, reload=True)
