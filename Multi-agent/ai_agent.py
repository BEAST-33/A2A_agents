"""
Google ADK-Compatible Bus Rating Agent

This agent fetches average user ratings for bus routes from ClickHouse database.
It's structured according to Google's Agent Development Kit (ADK) specifications.
"""

import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# ADK imports (install with: pip install google-cloud-adk)
from google.adk.agents import BaseAgent, AgentContext, AgentResult
from google.adk.tools import BaseTool

# ClickHouse connection
try:
    from clickhouse_connect import get_client as ch_get_client
except ImportError:
    raise ImportError("Please install clickhouse-connect: pip install clickhouse-connect")


class BusRatingTool(BaseTool):
    """Tool for fetching bus ratings from ClickHouse database."""
    
    def __init__(self):
        super().__init__(
            name="get_bus_rating",
            description="Fetches the average user rating for a given bus route ID from ClickHouse database",
            parameters={
                "type": "object",
                "properties": {
                    "route_id": {
                        "type": "string",
                        "description": "The bus route ID to fetch the rating for"
                    }
                },
                "required": ["route_id"]
            }
        )
    
    def execute(self, route_id: str) -> Dict[str, Any]:
        """Execute the tool to get bus rating."""
        try:
            rating = self._get_bus_rating_from_db(route_id)
            if rating is not None:
                return {
                    "success": True,
                    "route_id": route_id,
                    "rating": rating,
                    "message": f"Average bus rating for RouteID {route_id}: {rating}"
                }
            else:
                return {
                    "success": False,
                    "route_id": route_id,
                    "rating": None,
                    "message": f"No rating found for RouteID {route_id}"
                }
        except Exception as e:
            logging.error(f"Error in BusRatingTool: {e}")
            return {
                "success": False,
                "route_id": route_id,
                "rating": None,
                "message": f"Error fetching rating: {str(e)}"
            }
    
    def _get_bus_rating_from_db(self, route_id: str) -> Optional[float]:
        """Internal method to fetch rating from ClickHouse."""
        try:
            client = ch_get_client(
                host='10.5.40.193',
                port=8123,
                username='ugc_readonly',
                password='ugc@readonly!',
                secure=False
            )
            
            query = "SELECT Value FROM UGC.UserRatings WHERE RouteID = %s"
            result = client.query(query, (route_id,))
            logging.debug("Results of the ratings: =>> %s", result)
            
            rows = result.result_rows
            if rows:
                logging.debug("Individual ratings: %s", rows)
                ratings = [row[0] for row in rows]
                average_rating = round(sum(ratings) / len(ratings), 1)
                logging.info(f"Average bus score for RouteID {route_id}: {average_rating}")
                return average_rating
            else:
                logging.info("No data found for RouteID %s", route_id)
                return None
                
        except Exception as e:
            logging.error(f"Error fetching bus rating from ClickHouse: {e}")
            return None


class BusRatingAgent(BaseAgent):
    """
    ADK-compatible agent for fetching bus ratings.
    
    This agent extends BaseAgent and provides functionality to:
    1. Extract route IDs from natural language input
    2. Fetch average ratings from ClickHouse database
    3. Return structured responses compatible with ADK workflows
    """
    
    def __init__(self, name: str = "BusRatingAgent"):
        # Initialize the tool
        self.bus_rating_tool = BusRatingTool()
        
        # Initialize BaseAgent with tools
        super().__init__(
            name=name,
            description=(
                "Agent that accepts a bus route ID and returns the average user rating "
                "for that route, as stored in ClickHouse. Useful for retrieving feedback "
                "and quality metrics for bus services."
            ),
            tools=[self.bus_rating_tool]
        )
        
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(name)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Main execution method required by ADK BaseAgent.
        
        Args:
            context: AgentContext containing the user input and session state
            
        Returns:
            AgentResult with the rating information or error message
        """
        try:
            # Extract user input from context
            user_input = context.get_user_input()
            self.logger.debug(f"Received user input: {user_input}")
            
            # Extract route ID from the input
            route_id = self._extract_route_id(user_input)
            
            if not route_id:
                return AgentResult(
                    success=False,
                    message="No valid route ID found in the input. Please provide a bus route ID.",
                    data={"error": "No route ID found"}
                )
            
            # Use the tool to get the rating
            tool_result = self.bus_rating_tool.execute(route_id)
            
            if tool_result["success"]:
                return AgentResult(
                    success=True,
                    message=tool_result["message"],
                    data={
                        "route_id": route_id,
                        "rating": tool_result["rating"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    message=tool_result["message"],
                    data={
                        "route_id": route_id,
                        "error": "Rating not found"
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error in BusRatingAgent execution: {e}")
            return AgentResult(
                success=False,
                message=f"An error occurred while processing your request: {str(e)}",
                data={"error": str(e)}
            )
    
    def _extract_route_id(self, user_input: str) -> Optional[str]:
        """Extract route ID from user input using regex."""
        if not user_input:
            return None
            
        # Look for numeric patterns in the input
        match = re.search(r'\d+', user_input)
        if match:
            return match.group()
        
        return None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities for ADK framework."""
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "tools": [tool.name for tool in self.tools],
            "input_types": ["text", "natural_language"],
            "output_types": ["structured_data", "text"],
            "skills": [
                {
                    "id": "extract_route_id",
                    "name": "Extract Route ID",
                    "description": "Extracts bus route ID from natural language input"
                },
                {
                    "id": "fetch_rating",
                    "name": "Fetch Bus Rating",
                    "description": "Retrieves average user rating from ClickHouse database"
                }
            ]
        }


# ADK Application wrapper for deployment
class BusRatingApp:
    """
    ADK Application wrapper for the Bus Rating Agent.
    This allows the agent to be deployed and managed within ADK workflows.
    """
    
    def __init__(self):
        self.agent = BusRatingAgent()
        self.logger = logging.getLogger("BusRatingApp")
    
    async def run(self, input_text: str) -> Dict[str, Any]:
        """
        Main entry point for the application.
        
        Args:
            input_text: User input containing route ID request
            
        Returns:
            Dictionary with result data
        """
        try:
            # Create context (in real ADK deployment, this would be handled by the framework)
            context = AgentContext(user_input=input_text)
            
            # Execute the agent
            result = await self.agent.execute(context)
            
            return {
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "agent": self.agent.name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in BusRatingApp: {e}")
            return {
                "success": False,
                "message": f"Application error: {str(e)}",
                "data": {"error": str(e)},
                "agent": self.agent.name,
                "timestamp": datetime.utcnow().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        """Test the ADK-compatible agent."""
        app = BusRatingApp()
        
        # Test cases
        test_inputs = [
            "Get the rating for route ID 123",
            "What is the average rating for bus route 456?",
            "Route 789 rating please",
            "No route ID here",  # Should fail gracefully
        ]
        
        for test_input in test_inputs:
            print(f"\n--- Testing: {test_input} ---")
            result = await app.run(test_input)
            print(f"Success: {result['success']}")
            print(f"Message: {result['message']}")
            print(f"Data: {result['data']}")
    
    # Run the test
    asyncio.run(test_agent())