"""
Autonomous Vision Agent - Free Reverse Image Search Implementation
Uses free APIs: SauceNAO, Yandex, and DuckDuckGo for reverse image searching
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote
import os
from pathlib import Path


class SearchEngine(Enum):
    """Supported reverse image search engines"""
    SAUCENAO = "saucenao"
    YANDEX = "yandex"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchResult:
    """Data class for search results"""
    url: str
    title: str
    source: str
    similarity: Optional[float] = None
    thumbnail: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "similarity": self.similarity,
            "thumbnail": self.thumbnail,
            "metadata": self.metadata
        }


class SauceNAOSearcher:
    """SauceNAO reverse image search implementation"""
    
    BASE_URL = "https://saucenao.com/api/lookup"
    FREE_API_LIMIT = 200  # Free tier has 200 searches per 24 hours
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SauceNAO searcher
        
        Args:
            api_key: Optional API key (free searches work without key, but limited)
        """
        self.api_key = api_key or os.environ.get("SAUCENAO_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def search_by_url(self, image_url: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search by image URL
        
        Args:
            image_url: URL of the image to search
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        params = {
            "url": image_url,
            "output_type": 2,  # JSON output
            "numres": min(num_results, 16)  # Max 16 per request
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "results" in data:
                for result in data["results"][:num_results]:
                    header = result.get("header", {})
                    similarity = float(header.get("similarity", 0))
                    
                    # Extract relevant URLs
                    url = result.get("data", {}).get("source", "")
                    title = result.get("data", {}).get("title", "Unknown")
                    
                    if url:
                        results.append(SearchResult(
                            url=url,
                            title=title,
                            source=SearchEngine.SAUCENAO.value,
                            similarity=similarity,
                            thumbnail=result.get("data", {}).get("thumbnail", ""),
                            metadata={
                                "db": header.get("index_name", "Unknown DB"),
                                "index_id": header.get("index_id"),
                                "similarity_percentage": f"{similarity}%"
                            }
                        ))
            
            return results
            
        except requests.RequestException as e:
            print(f"SauceNAO search error: {e}")
            return []
    
    def search_by_file(self, file_path: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search by local image file
        
        Args:
            file_path: Path to local image file
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            with open(file_path, "rb") as f:
                files = {"image": f}
                params = {
                    "output_type": 2,
                    "numres": min(num_results, 16)
                }
                
                if self.api_key:
                    params["api_key"] = self.api_key
                
                response = self.session.post(
                    self.BASE_URL,
                    files=files,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                if "results" in data:
                    for result in data["results"][:num_results]:
                        header = result.get("header", {})
                        similarity = float(header.get("similarity", 0))
                        url = result.get("data", {}).get("source", "")
                        title = result.get("data", {}).get("title", "Unknown")
                        
                        if url:
                            results.append(SearchResult(
                                url=url,
                                title=title,
                                source=SearchEngine.SAUCENAO.value,
                                similarity=similarity,
                                metadata={
                                    "db": header.get("index_name", "Unknown DB"),
                                    "similarity_percentage": f"{similarity}%"
                                }
                            ))
                
                return results
                
        except (FileNotFoundError, requests.RequestException) as e:
            print(f"SauceNAO file search error: {e}")
            return []


class YandexSearcher:
    """Yandex reverse image search implementation"""
    
    BASE_URL = "https://yandex.com/images/search"
    API_URL = "https://yandex.com/images/api/v1/search"
    
    def __init__(self):
        """Initialize Yandex searcher"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    def search_by_url(self, image_url: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search by image URL using Yandex
        
        Args:
            image_url: URL of the image to search
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Yandex search via URL parameter
            params = {
                "rpt": "imageview",
                "url": image_url,
                "cbir_page": "similar"
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse HTML response for image results
            results = self._parse_yandex_response(response.text, num_results)
            return results
            
        except requests.RequestException as e:
            print(f"Yandex search error: {e}")
            return []
    
    def search_by_file(self, file_path: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search by local image file using Yandex
        
        Args:
            file_path: Path to local image file
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            with open(file_path, "rb") as f:
                files = {"upfile": f}
                data = {"rpt": "imageview"}
                
                response = self.session.post(
                    self.BASE_URL,
                    files=files,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                
                results = self._parse_yandex_response(response.text, num_results)
                return results
                
        except (FileNotFoundError, requests.RequestException) as e:
            print(f"Yandex file search error: {e}")
            return []
    
    def _parse_yandex_response(self, html: str, num_results: int) -> List[SearchResult]:
        """
        Parse Yandex HTML response for image results
        
        Args:
            html: HTML response from Yandex
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Simple extraction of image links from response
        # In production, use BeautifulSoup for more robust parsing
        import re
        
        # Extract image results from JSON data embedded in page
        json_patterns = [
            r'"thumbUrl":"([^"]+)"',
            r'"url":"([^"]+)"',
        ]
        
        urls = set()
        for pattern in json_patterns:
            for match in re.finditer(pattern, html):
                url = match.group(1)
                if url and "yandex" not in url:
                    urls.add(url)
        
        for url in list(urls)[:num_results]:
            results.append(SearchResult(
                url=url,
                title="Image result",
                source=SearchEngine.YANDEX.value,
                metadata={"search_type": "yandex_reverse_image"}
            ))
        
        return results


class DuckDuckGoSearcher:
    """DuckDuckGo reverse image search implementation"""
    
    BASE_URL = "https://duckduckgo.com/"
    API_URL = "https://duckduckgo.com/api/search"
    
    def __init__(self):
        """Initialize DuckDuckGo searcher"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })
    
    def search_by_url(self, image_url: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search by image URL using DuckDuckGo
        
        Args:
            image_url: URL of the image to search
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # DuckDuckGo image search
            params = {
                "q": f"!ri {image_url}",  # Reverse image search operator
                "format": "json"
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            results = self._extract_results_from_page(response.text, num_results)
            return results
            
        except requests.RequestException as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def search_by_file(self, file_path: str, num_results: int = 5) -> List[SearchResult]:
        """
        DuckDuckGo doesn't support file uploads directly via free API
        This is a placeholder for consistency with other searchers
        
        Args:
            file_path: Path to local image file (not used)
            num_results: Number of results to return
            
        Returns:
            Empty list (not supported)
        """
        print("DuckDuckGo free API does not support direct file uploads")
        print("Please use a publicly accessible image URL instead")
        return []
    
    def _extract_results_from_page(self, html: str, num_results: int) -> List[SearchResult]:
        """
        Extract results from DuckDuckGo response
        
        Args:
            html: HTML response from DuckDuckGo
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        import re
        
        # Extract URLs from page
        url_pattern = r'(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$\-._+!*\'(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
        
        urls = set()
        for match in re.finditer(url_pattern, html):
            url = match.group(0)
            if url and "duckduckgo" not in url:
                urls.add(url)
        
        for url in list(urls)[:num_results]:
            results.append(SearchResult(
                url=url,
                title="Image result",
                source=SearchEngine.DUCKDUCKGO.value,
                metadata={"search_type": "duckduckgo_image_search"}
            ))
        
        return results


class UniversalImageSearcher:
    """
    Universal image searcher that combines multiple free search engines
    """
    
    def __init__(self, use_saucenao: bool = True, use_yandex: bool = True, use_duckduckgo: bool = True):
        """
        Initialize universal searcher
        
        Args:
            use_saucenao: Enable SauceNAO search
            use_yandex: Enable Yandex search
            use_duckduckgo: Enable DuckDuckGo search
        """
        self.searchers = {}
        
        if use_saucenao:
            self.searchers[SearchEngine.SAUCENAO] = SauceNAOSearcher()
        if use_yandex:
            self.searchers[SearchEngine.YANDEX] = YandexSearcher()
        if use_duckduckgo:
            self.searchers[SearchEngine.DUCKDUCKGO] = DuckDuckGoSearcher()
    
    def search_by_url(self, image_url: str, num_results: int = 5, engines: Optional[List[SearchEngine]] = None) -> Dict[str, List[SearchResult]]:
        """
        Search image across multiple engines
        
        Args:
            image_url: URL of the image to search
            num_results: Number of results per engine
            engines: Specific engines to use (None = all available)
            
        Returns:
            Dictionary with engine names as keys and result lists as values
        """
        if engines is None:
            engines = list(self.searchers.keys())
        
        all_results = {}
        
        for engine in engines:
            if engine in self.searchers:
                print(f"Searching with {engine.value}...")
                searcher = self.searchers[engine]
                results = searcher.search_by_url(image_url, num_results)
                all_results[engine.value] = results
                time.sleep(0.5)  # Rate limiting between requests
        
        return all_results
    
    def search_by_file(self, file_path: str, num_results: int = 5, engines: Optional[List[SearchEngine]] = None) -> Dict[str, List[SearchResult]]:
        """
        Search image file across multiple engines
        
        Args:
            file_path: Path to local image file
            num_results: Number of results per engine
            engines: Specific engines to use (None = all available)
            
        Returns:
            Dictionary with engine names as keys and result lists as values
        """
        if engines is None:
            engines = list(self.searchers.keys())
        
        all_results = {}
        
        for engine in engines:
            if engine in self.searchers:
                print(f"Searching with {engine.value}...")
                searcher = self.searchers[engine]
                
                # Skip DuckDuckGo for file uploads (not supported in free API)
                if engine == SearchEngine.DUCKDUCKGO:
                    print(f"Skipping {engine.value} (file uploads not supported in free API)")
                    continue
                
                results = searcher.search_by_file(file_path, num_results)
                all_results[engine.value] = results
                time.sleep(0.5)  # Rate limiting between requests
        
        return all_results
    
    def format_results(self, results: Dict[str, List[SearchResult]], max_per_engine: int = 3) -> str:
        """
        Format search results for display
        
        Args:
            results: Results dictionary from search_by_url or search_by_file
            max_per_engine: Maximum results to display per engine
            
        Returns:
            Formatted string representation
        """
        formatted = "=== Reverse Image Search Results ===\n\n"
        
        for engine_name, result_list in results.items():
            formatted += f"\n📌 {engine_name.upper()} Results:\n"
            formatted += "-" * 50 + "\n"
            
            if not result_list:
                formatted += "No results found.\n"
            else:
                for i, result in enumerate(result_list[:max_per_engine], 1):
                    formatted += f"\n{i}. {result.title}\n"
                    formatted += f"   URL: {result.url}\n"
                    if result.similarity:
                        formatted += f"   Similarity: {result.similarity}%\n"
                    if result.metadata:
                        for key, value in result.metadata.items():
                            formatted += f"   {key}: {value}\n"
        
        return formatted


# Example usage and CLI
def main():
    """Main function demonstrating usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Free Reverse Image Search using multiple engines"
    )
    parser.add_argument("image", help="Image URL or file path")
    parser.add_argument(
        "--engine",
        choices=["saucenao", "yandex", "duckduckgo", "all"],
        default="all",
        help="Search engine to use"
    )
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of results per engine"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = UniversalImageSearcher()
    
    # Determine if input is URL or file
    is_url = args.image.startswith(("http://", "https://"))
    
    # Select engines
    engines = None
    if args.engine != "all":
        engines = [SearchEngine(args.engine)]
    
    # Perform search
    if is_url:
        results = searcher.search_by_url(args.image, args.results, engines)
    else:
        results = searcher.search_by_file(args.image, args.results, engines)
    
    # Format output
    if args.output == "json":
        # Convert to JSON-serializable format
        json_results = {}
        for engine, res_list in results.items():
            json_results[engine] = [r.to_dict() for r in res_list]
        print(json.dumps(json_results, indent=2))
    else:
        print(searcher.format_results(results))


if __name__ == "__main__":
    main()
