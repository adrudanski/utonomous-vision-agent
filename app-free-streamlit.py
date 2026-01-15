"""
Autonomous Vision Agent - Streamlit App with Free Services
Uses SauceNAO, Yandex, and DuckDuckGo for reverse image search
NO API KEYS REQUIRED - Works on Streamlit Cloud
"""

import streamlit as st
import requests
import time
import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from PIL import Image
import tempfile
import os

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Free Reverse Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .search-result {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ===== SEARCH ENGINE CLASSES =====

class SauceNAOSearcher:
    """SauceNAO reverse image search - FREE tier available"""
    
    BASE_URL = "https://saucenao.com/api/lookup"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SAUCENAO_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def search_by_file(self, file_path: str, num_results: int = 5) -> List[Dict]:
        """Search by local image file"""
        try:
            with open(file_path, "rb") as f:
                files = {"image": f}
                params = {
                    "output_type": 2,  # JSON output
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
                            results.append({
                                "engine": "SauceNAO",
                                "url": url,
                                "title": title,
                                "similarity": similarity,
                                "metadata": {
                                    "db": header.get("index_name", "Unknown DB"),
                                    "similarity_percentage": f"{similarity}%"
                                }
                            })
                
                return results
                
        except Exception as e:
            st.error(f"SauceNAO search error: {str(e)}")
            return []


class YandexSearcher:
    """Yandex reverse image search - No API key required"""
    
    BASE_URL = "https://yandex.com/images/search"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    def search_by_file(self, file_path: str, num_results: int = 5) -> List[Dict]:
        """Search by local image file"""
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
                
                # Parse HTML response for image results
                results = self._parse_yandex_response(response.text, num_results)
                return results
                
        except Exception as e:
            st.warning(f"Yandex search may have encountered a CAPTCHA: {str(e)}")
            return []
    
    def _parse_yandex_response(self, html: str, num_results: int) -> List[Dict]:
        """Parse Yandex HTML response for image results"""
        results = []
        
        # Extract image URLs from JSON data embedded in page
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
            results.append({
                "engine": "Yandex",
                "url": url,
                "title": "Image result",
                "similarity": None,
                "metadata": {"search_type": "yandex_reverse_image"}
            })
        
        return results


class DuckDuckGoSearcher:
    """DuckDuckGo image search - No API key required"""
    
    BASE_URL = "https://duckduckgo.com/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })
    
    def search_by_url(self, image_url: str, num_results: int = 5) -> List[Dict]:
        """Search by image URL"""
        try:
            params = {
                "q": f"!ri {image_url}",
                "format": "json"
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            results = self._extract_results_from_page(response.text, num_results)
            return results
            
        except Exception as e:
            st.warning(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def _extract_results_from_page(self, html: str, num_results: int) -> List[Dict]:
        """Extract results from DuckDuckGo response"""
        results = []
        
        url_pattern = r'(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$\-._+!*\'(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
        
        urls = set()
        for match in re.finditer(url_pattern, html):
            url = match.group(0)
            if url and "duckduckgo" not in url:
                urls.add(url)
        
        for url in list(urls)[:num_results]:
            results.append({
                "engine": "DuckDuckGo",
                "url": url,
                "title": "Image result",
                "similarity": None,
                "metadata": {"search_type": "duckduckgo_image_search"}
            })
        
        return results


class UniversalImageSearcher:
    """Universal image searcher combining multiple free engines"""
    
    def __init__(self, use_saucenao: bool = True, use_yandex: bool = True, use_duckduckgo: bool = True):
        self.searchers = {}
        
        if use_saucenao:
            self.searchers["saucenao"] = SauceNAOSearcher()
        if use_yandex:
            self.searchers["yandex"] = YandexSearcher()
        if use_duckduckgo:
            self.searchers["duckduckgo"] = DuckDuckGoSearcher()
    
    def search_by_file(self, file_path: str, num_results: int = 5, engines: Optional[List[str]] = None) -> Dict[str, List]:
        """Search image file across multiple engines"""
        if engines is None:
            engines = list(self.searchers.keys())
        
        all_results = {}
        
        for engine_name in engines:
            if engine_name in self.searchers:
                st.info(f"🔍 Searching with {engine_name.upper()}...")
                searcher = self.searchers[engine_name]
                
                try:
                    if engine_name == "duckduckgo":
                        # DuckDuckGo doesn't support file uploads directly
                        st.warning(f"Skipping {engine_name} (file uploads not supported)")
                        continue
                    
                    results = searcher.search_by_file(file_path, num_results)
                    all_results[engine_name] = results
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    st.error(f"Error with {engine_name}: {str(e)}")
                    all_results[engine_name] = []
        
        return all_results


# ===== STREAMLIT APP =====

def main():
    """Main Streamlit application"""
    
    st.title("🔍 Free Reverse Image Search")
    st.markdown("Upload an image to find similar images using **FREE services** - NO API keys required!")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    num_results = st.sidebar.slider(
        "Number of Results per Engine",
        min_value=1,
        max_value=10,
        value=5,
        help="How many results to fetch from each search engine"
    )
    
    # Search engine selection
    st.sidebar.subheader("🌐 Search Engines")
    use_saucenao = st.sidebar.checkbox("SauceNAO", value=True, help="Best for anime/manga, similarity scores")
    use_yandex = st.sidebar.checkbox("Yandex", value=True, help="Good for general images")
    use_duckduckgo = st.sidebar.checkbox("DuckDuckGo", value=True, help="Basic reverse search")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Information")
    st.sidebar.info("""
    **Free Services Used:**
    - SauceNAO (200 searches/day free)
    - Yandex (no official limits)
    - DuckDuckGo (no limits)
    
    **Note:** Yandex may occasionally show CAPTCHA. If that happens, try SauceNAO instead.
    """)
    
    # Main upload section
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "gif", "bmp"],
        help="Upload an image to search for similar images"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption=f"Size: {image.size}")
        
        with col2:
            st.subheader("Search Settings")
            
            # Add optional search query to refine results
            search_query = st.text_input(
                "Additional Context (Optional)",
                placeholder="e.g., 'red sports car', 'vintage watch'",
                help="Add keywords to help refine the search results"
            )
            
            # Search button with key to ensure it works
            search_button = st.button(
                "🔍 Search Now",
                key="search_button",
                use_container_width=True,
                type="primary"
            )
            
            if search_button:
                # Save uploaded file to temporary location
                with st.spinner("🔍 Searching across multiple engines..."):
                    try:
                        # Create temporary file
                        suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.read())
                            temp_path = tmp.name
                        
                        # Initialize searcher
                        searcher = UniversalImageSearcher(
                            use_saucenao=use_saucenao,
                            use_yandex=use_yandex,
                            use_duckduckgo=use_duckduckgo
                        )
                        
                        # Perform search
                        results = searcher.search_by_file(temp_path, num_results=num_results)
                        
                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                        # Display results
                        st.success(f"✅ Search completed!")
                        
                        total_results = sum(len(r) for r in results.values())
                        st.info(f"📊 Found {total_results} results across {len(results)} search engines")
                        
                        # Display results by engine
                        for engine_name, engine_results in results.items():
                            if engine_results:
                                st.subheader(f"🔎 {engine_name.upper()} Results")
                                
                                for i, result in enumerate(engine_results, 1):
                                    with st.container(border=True):
                                        # Title and similarity
                                        col_title, col_sim = st.columns([3, 1])
                                        with col_title:
                                            st.markdown(f"**{i}. {result['title'][:50]}**" if len(result['title']) > 50 else f"**{i}. {result['title']}**")
                                        with col_sim:
                                            if result.get('similarity'):
                                                st.markdown(f"📊 {result['similarity']:.1f}%")
                                        
                                        # URL
                                        st.caption(f"🔗 {result['url']}")
                                        
                                        # Metadata
                                        if result.get('metadata'):
                                            metadata_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()])
                                            st.caption(f"📋 {metadata_str}")
                                        
                                        # Action button
                                        st.markdown(f"[🌐 View Full Result]({result['url']})", unsafe_allow_html=True)
                                        st.markdown("---")
                            else:
                                st.warning(f"⚠️ No results from {engine_name.upper()}")
                    
                    except Exception as e:
                        st.error(f"""
                        ❌ **Error during search:**
                        
                        {str(e)}
                        
                        **Troubleshooting tips:**
                        1. Try a different image format (JPG, PNG)
                        2. Reduce image size (under 5MB)
                        3. Try fewer search engines at once
                        4. Check your internet connection
                        """)
                        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Free Reverse Image Search • Powered by SauceNAO, Yandex & DuckDuckGo
        <br>
        No API keys required • Works on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()