import json
from typing import List, Dict
import requests
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bus_analyzer.log'),
        logging.StreamHandler()
    ]
)

# Set urllib3 and requests logging to WARNING level to reduce noise
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class TagAnalysis:
    tag: str
    positive_count: int
    negative_count: int
    total_count: int
    sentiment_score: float

class BusServiceAnalyzer:
    def __init__(self):
        logger.info("Initializing BusServiceAnalyzer")
        self.base_url = "http://rbplus-api-prod.redbus.in:8085/rpwapp/v1/rnr/servicewiseratings"
        self.ollama_url = os.getenv('OLLAMA_API_URL')
        self.ollama_model = os.getenv('OLLAMA_MODEL_NAME', 'mistral:7b-instruct')
        if not self.ollama_url:
            raise ValueError("OLLAMA_API_URL not found in environment variables")
        logger.info(f"Using Ollama model: {self.ollama_model}")
    def fetch_data(self, operator_id: str, source_id: str, dest_id: str, limit: int = 10, page: int = 1) -> List[Dict]:
        """
        Fetch data from the RedBus API
        """
        
        
        params = {
            'limit': limit,
            'page': page,
            'sourceId': source_id,
            'destId': dest_id,
            'sortOrder': 'desc',
            'sortColumn': 'sales'
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/{operator_id}",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status_key') != 'suc':
                raise Exception(f"API returned error: {data.get('status_key')}")
                
            return data.get('data', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing API response: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return []
    
    
    def process_tags(self, tags_json: List[Dict]) -> Dict[str, TagAnalysis]:
        logger.info(f"Processing {len(tags_json)} tags")
        start_time = time.time()
        
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
        
        end_time = time.time()
        logger.info(f"Processed {len(tag_analysis)} unique tags in {end_time - start_time:.2f} seconds")
        return dict(tag_analysis)
    
    def analyze_with_llm(self, tag_analysis: Dict[str, TagAnalysis]) -> str:
        logger.info("Starting LLM analysis")
        start_time = time.time()
        
        # Prepare the prompt for the LLM
        prompt = "Analyze the following bus service feedback data and provide recommendations:\n\n"
        for tag, analysis in tag_analysis.items():
            prompt += f"Tag: {tag}\n"
            prompt += f"Positive feedback: {analysis.positive_count}\n"
            prompt += f"Negative feedback: {analysis.negative_count}\n"
            prompt += f"Total feedback: {analysis.total_count}\n"
            prompt += f"Sentiment score: {analysis.sentiment_score:.2f}%\n\n"
        
        prompt += "Focussing on the feedback for each option, mainly taking sentiment score as priority such that, more is the score, more likely it is better, give the user the option in decreasing order of preference."
        
        logger.debug(f"Prepared prompt of length: {len(prompt)}")
        
        # Get LLM analysis from Ollama
        try:
            logger.info("Sending request to Ollama API")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": "You are a transportation service analyst. Analyze the feedback data and provide actionable insights."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 500
                    }
                }
            )
            response.raise_for_status()
            result = response.json()['message']['content']
            end_time = time.time()
            logger.info(f"LLM analysis completed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error getting analysis from Ollama: {str(e)}")
            return "Error: Could not get analysis from LLM"

def main():
    logger.info("Starting bus analyzer")
    start_time = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the analyzer
    try:
        analyzer = BusServiceAnalyzer()
    except ValueError as e:
        logger.error(f"Initialization error: {str(e)}")
        return
    
    # Fetch data from API
    operator_id = "22"  # Example operator ID
    source_id = "122"   # Example source ID
    dest_id = "124"     # Example destination ID
    
    logger.info(f"Fetching data for operator: {operator_id}, source: {source_id}, dest: {dest_id}")
    data = analyzer.fetch_data(
        operator_id=operator_id,
        source_id=source_id,
        dest_id=dest_id,
        limit=10,
        page=1
    )
    
    if not data:
        logger.warning("No data received from API")
        return
    
    logger.info(f"Processing {len(data)} services")
    
    # Process each bus service
    for i, service in enumerate(data, 1):
        logger.info(f"Processing service {i}/{len(data)}")
        print(f"\nAnalyzing service from {service['ServiceDetails']['SrcNm']} to {service['ServiceDetails']['DestNm']}")
        print(f"Bus Type: {service['BusTypelist'][0]}")
        print(f"Overall Rating: {service['overallRating']}")
        
        # Process tags
        tag_analysis = analyzer.process_tags(service['TagsJson'])
        
        # Get LLM analysis
        analysis = analyzer.analyze_with_llm(tag_analysis)
        print("\nAnalysis and Recommendations:")
        print(analysis)
        print("\n" + "="*80)
    
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 