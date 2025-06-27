from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.http import HttpTool

def create_reviews_agent() -> Agent:
    
    clickhouse_tool = HttpTool(
        name="clickhouse_query",
        description="Query ClickHouse database for bus reviews",
        base_url="http://localhost:8024",  # Your ClickHouse HTTP interface
        # Add any required headers, authentication, etc.
    )
    return Agent(
        name="BusReviewAgent",
        description="Fetches user reviews for bus routes based on RouteID from ClickHouse.",
        model=LiteLlm(
            name= "ollama_chat/mistral:7b-instruct",
            api_key="http://10.166.8.126:11434/api/chat"
        ),
        instruction="You are a bus review agent. Your task is to fetch user reviews for bus routes based on RouteID from ClickHouse. If the RouteID is not found, return an appropriate error message.",
        tools=[http://localhost:8024]
    )
root_agent = create_reviews_agent()

