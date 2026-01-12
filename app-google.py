"""
Autonomous Vision Agent using Google Custom Search API for reverse image search
"""

import os
import requests
from typing import List, Dict, Any
from google.oauth2 import service_account
from googleapiclient.discovery import build
from urllib.parse import urlencode
import base64
from pathlib import Path


class GoogleImageSearchAgent:
    """Agent for reverse image search using Google Custom Search API"""
    
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        """
        Initialize the Google Image Search Agent
        
        Args:
            api_key: Google Custom Search API key. If not provided, reads from GOOGLE_API_KEY env var
            search_engine_id: Google Custom Search Engine ID. If not provided, reads from GOOGLE_SEARCH_ENGINE_ID env var
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
        if not self.search_engine_id:
            raise ValueError("Google Search Engine ID not provided and GOOGLE_SEARCH_ENGINE_ID environment variable not set")
    
    def reverse_image_search(self, image_path: str = None, image_url: str = None, 
                            num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform reverse image search using Google Custom Search API
        
        Args:
            image_path: Local file path to the image
            image_url: URL of the image to search
            num_results: Number of results to return (max 10 per request)
            
        Returns:
            List of search results containing title, link, snippet, and image info
        """
        if not image_path and not image_url:
            raise ValueError("Either image_path or image_url must be provided")
        
        if image_path:
            image_url = self._upload_image_to_google(image_path)
        
        return self._search_by_image_url(image_url, num_results)
    
    def _upload_image_to_google(self, image_path: str) -> str:
        """
        Upload local image to a temporary hosting service and return URL
        For production, consider using Google Cloud Storage or similar
        
        Args:
            image_path: Path to the image file
            
        Returns:
            URL of the uploaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # For this implementation, we'll use a simple approach
        # In production, use Google Cloud Storage or ImgBB API for temporary hosting
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Using ImgBB API as a free alternative for temporary image hosting
        imgbb_api_key = os.getenv('IMGBB_API_KEY')
        if not imgbb_api_key:
            raise ValueError("ImgBB API key not provided. Set IMGBB_API_KEY environment variable for local image uploads")
        
        url = "https://api.imgbb.com/1/upload"
        files = {'image': image_data}
        data = {'key': imgbb_api_key}
        
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()['data']['url']
    
    def _search_by_image_url(self, image_url: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using image URL via Google Custom Search API
        
        Args:
            image_url: URL of the image to search
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        params = {
            'q': image_url,
            'key': self.api_key,
            'cx': self.search_engine_id,
            'searchType': 'image',
            'num': min(num_results, 10),  # Google API max is 10 per request
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if 'items' in data:
            for item in data['items']:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'image': {
                        'url': item.get('image', {}).get('thumbnailUrl', ''),
                        'width': item.get('image', {}).get('width', ''),
                        'height': item.get('image', {}).get('height', ''),
                        'byteSize': item.get('image', {}).get('byteSize', ''),
                    },
                    'source_url': item.get('image', {}).get('contextLink', ''),
                }
                results.append(result)
        
        return results
    
    def search_similar_images(self, image_path: str = None, image_url: str = None, 
                             query_suffix: str = None, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar images with optional query suffix
        
        Args:
            image_path: Local file path to the image
            image_url: URL of the image to search
            query_suffix: Additional query terms to refine search
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        results = self.reverse_image_search(image_path, image_url, num_results)
        
        if query_suffix:
            # Refine results with additional query
            refined_params = {
                'q': query_suffix,
                'key': self.api_key,
                'cx': self.search_engine_id,
                'searchType': 'image',
                'num': min(num_results, 10),
            }
            
            response = requests.get(self.base_url, params=refined_params)
            response.raise_for_status()
            
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'image': {
                            'url': item.get('image', {}).get('thumbnailUrl', ''),
                            'width': item.get('image', {}).get('width', ''),
                            'height': item.get('image', {}).get('height', ''),
                            'byteSize': item.get('image', {}).get('byteSize', ''),
                        },
                        'source_url': item.get('image', {}).get('contextLink', ''),
                    }
                    results.append(result)
        
        return results


class VisionAgent:
    """Main Autonomous Vision Agent integrating Google Image Search"""
    
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        """
        Initialize the Vision Agent
        
        Args:
            api_key: Google Custom Search API key
            search_engine_id: Google Custom Search Engine ID
        """
        self.search_agent = GoogleImageSearchAgent(api_key, search_engine_id)
    
    def analyze_image(self, image_path: str = None, image_url: str = None) -> Dict[str, Any]:
        """
        Analyze an image and return reverse search results
        
        Args:
            image_path: Local file path to the image
            image_url: URL of the image
            
        Returns:
            Analysis results including reverse image search findings
        """
        if not image_path and not image_url:
            raise ValueError("Either image_path or image_url must be provided")
        
        results = self.search_agent.reverse_image_search(image_path, image_url, num_results=10)
        
        return {
            'image_source': image_path or image_url,
            'total_results': len(results),
            'search_results': results,
            'analysis_timestamp': None,  # Can be set with datetime.now()
        }
    
    def find_product_sources(self, image_path: str = None, image_url: str = None,
                            product_name: str = None) -> Dict[str, Any]:
        """
        Find sources and similar products for an image
        
        Args:
            image_path: Local file path to the image
            image_url: URL of the image
            product_name: Optional product name to refine search
            
        Returns:
            Product sources and similar items
        """
        query = product_name if product_name else ""
        results = self.search_agent.search_similar_images(
            image_path=image_path,
            image_url=image_url,
            query_suffix=query,
            num_results=10
        )
        
        return {
            'product': product_name or 'Unknown',
            'sources_found': len(results),
            'items': results,
        }


def main():
    """Example usage of the Vision Agent"""
    
    # Initialize the agent
    agent = VisionAgent()
    
    # Example 1: Reverse image search with URL
    print("Example 1: Reverse Image Search")
    print("-" * 50)
    image_url = "https://example.com/image.jpg"  # Replace with actual image URL
    try:
        results = agent.analyze_image(image_url=image_url)
        print(f"Found {results['total_results']} results")
        for i, result in enumerate(results['search_results'][:3], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Link: {result['link']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Find product sources
    print("\n\nExample 2: Product Source Finder")
    print("-" * 50)
    try:
        products = agent.find_product_sources(
            image_url=image_url,
            product_name="smartphone"
        )
        print(f"Product: {products['product']}")
        print(f"Sources found: {products['sources_found']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
