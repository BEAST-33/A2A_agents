# red_deals_agent.py (FINAL - No Mock Data)

import requests
import json
from datetime import datetime
import logging

# Configure logger
logger = logging.getLogger(__name__)

class RedDealsAgent:
    """
    Agent responsible for fetching RedBus campaign data and preparing it for LLM analysis.
    This version now exclusively fetches data from the live API.
    """
    def __init__(self, api_base_url: str = "http://api-rbplus-prod.redbus.in/api/Campaign/GetAllCampaignDetails"):
        self.api_base_url = api_base_url
        logger.info(f"RedDealsAgent initialized with API URL: {api_base_url}")
        # Removed: self.use_mock_data = use_mock_data
        # Removed: self.mock_data_path = mock_data_path

    def _fetch_campaign_data(self, operator_id: int, is_international: bool) -> dict | None:
        """
        Internal method to fetch raw campaign details from the RedBus API.
        This method now only performs API calls.
        """
        params = {
            "operatorID": operator_id,
            "isInternational": str(is_international).lower()
        }
        try:
            logger.info(f"Fetching campaign data for operator_id: {operator_id}, is_international: {is_international}")
            start_time = datetime.now()
            
            response = requests.get(self.api_base_url, params=params, timeout=1500)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"API call completed in {duration:.2f} seconds with status: {response.status_code}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("API request timed out after 1500 seconds")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch API data: {str(e)}")
            return None

    def get_red_deals_data(self, operator_id: int, is_international: bool, filter_keyword: str = None, include_expired: bool = False) -> str:
        """
        Public method to fetch and prepare RedBus deal data based on parameters.
        This is the "tool" function Ollama will "call".

        Args:
            operator_id (int): The operator ID for the campaigns.
            is_international (bool): Flag to specify if international campaigns are needed.
            filter_keyword (str, optional): A keyword to filter deals by their title or description. Defaults to None.
            include_expired (bool, optional): If True, expired deals will also be included. Defaults to False.

        Returns:
            str: A formatted string summarizing the relevant RedDeals.
        """
        logger.info(f"Getting red deals data - operator_id: {operator_id}, is_international: {is_international}, filter_keyword: {filter_keyword}, include_expired: {include_expired}")
        
        campaign_data = self._fetch_campaign_data(operator_id, is_international)

        if not campaign_data or not campaign_data.get("Data"):
            logger.warning("No campaign data available")
            return "No campaign data was provided or available to analyze."

        red_deals_list = []
        for campaign in campaign_data["Data"]:
            # Prioritize 'CampaignTitle' but fallback to 'campaignDesc'
            campaign_title = campaign.get("CampaignTitle")
            if not campaign_title:
                campaign_title = campaign.get("campaignDesc", "Unnamed Deal")

            campaign_description = campaign.get("campaignDesc", "No description available.")
            
            # Check deal status
            status = campaign.get("status", "UNKNOWN").upper()
            if status != "ACTIVE" and not include_expired:
                continue # Skip non-active deals unless explicitly asked

            # Parse campaignNote for more details
            validity_str = "N/A"
            min_fare = "N/A"
            max_discount_limit = "N/A"
            try:
                notes_str = campaign.get("campaignNote", "[]")
                if notes_str.strip() and notes_str.startswith('[') and notes_str.endswith(']'):
                    notes = json.loads(notes_str)
                    for note in notes:
                        if "Validity :" in note:
                            validity_str = note.replace("Validity : ", "").strip()
                        elif "Minimum ticket fare :" in note:
                            min_fare = note.replace("Minimum ticket fare : ", "").strip()
                        elif "Maximum discount limit :" in note:
                            max_discount_limit = note.replace("Maximum discount limit : ", "").strip()
                else:
                    pass
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse campaign notes for {campaign_title}: {str(e)}")

            discount_value = campaign.get("discountValue", 0.0)
            discount_percent = campaign.get("discountPercent", 0.0)
            
            discount_info = ""
            if discount_percent > 0:
                discount_info = f"{discount_percent}%"
            elif discount_value > 0:
                discount_info = f"{campaign.get('CurrencyCode', 'INR')} {discount_value} OFF"

            # Apply filter if keyword is provided (case-insensitive)
            if filter_keyword:
                if filter_keyword.lower() not in campaign_title.lower() and \
                   filter_keyword.lower() not in campaign_description.lower():
                    continue

            deal_status_text = ""
            if status != "ACTIVE":
                deal_status_text = f" (Status: {status})"

            red_deals_list.append(
                f"Deal Name: {campaign_title}{deal_status_text}\n"
                f"Description: {campaign_description}\n"
                f"Discount: {discount_info if discount_info else 'N/A'}\n"
                f"Validity: {validity_str}\n"
                f"Min Fare: {min_fare}\n"
                f"Max Discount: {max_discount_limit}"
            )
        
        if not red_deals_list:
            if filter_keyword:
                logger.info(f"No deals found matching filter keyword: {filter_keyword}")
                return f"No specific 'RedDeals' found matching '{filter_keyword}'."
            else:
                logger.info("No deals found")
                return "No current 'RedDeals' were found." if not include_expired else "No RedDeals (including expired) were found."

        logger.info(f"Found {len(red_deals_list)} deals")
        return (
            "Available RedBus Deals:\n\n" +
            "\n---\n\n".join(red_deals_list) +
            "\n\n---End of Deals Data---"
        )