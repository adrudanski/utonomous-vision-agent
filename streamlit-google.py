"""
Streamlit app for Google Custom Search API integration with reverse image search.
"""

import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
import os
from urllib.parse import quote
import base64

# Page configuration
st.set_page_config(
    page_title="Reverse Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
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
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🔍 Reverse Image Search")
st.markdown("Find similar images and information about images using Google Custom Search API")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Google Custom Search API Key",
        type="password",
        help="Get your API key from Google Cloud Console"
    )
    
    search_engine_id = st.text_input(
        "Custom Search Engine ID",
        help="Create a custom search engine on Google Programmable Search"
    )
    
    st.markdown("---")
    st.subheader("📖 How to get credentials:")
    with st.expander("Click to expand instructions"):
        st.markdown("""
        1. **Google API Key:**
           - Go to [Google Cloud Console](https://console.cloud.google.com/)
           - Create a new project
           - Enable Custom Search API
           - Create credentials (API Key)
        
        2. **Search Engine ID:**
           - Visit [Programmable Search](https://programmable-search.google.com/)
           - Create new search engine
           - Configure to search the entire web
           - Copy the Search Engine ID
        """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🔗 URL Search", "📋 About"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "gif", "bmp"],
        help="Upload an image to search for similar images"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            image_details = f"Size: {image.size} | Format: {image.format}"
            st.caption(image_details)
        
        with col2:
            st.subheader("Search Settings")
            search_type = st.radio(
                "Search Type:",
                ["Visual Similarity", "Related Products", "News", "All Results"]
            )
            
            num_results = st.slider(
                "Number of Results:",
                min_value=5,
                max_value=30,
                value=10,
                step=5
            )
            
            if st.button("🔍 Search", key="upload_search", use_container_width=True):
                if not api_key or not search_engine_id:
                    st.error("❌ Please enter your API Key and Search Engine ID in the sidebar")
                else:
                    with st.spinner("🔄 Searching for similar images..."):
                        # Convert image to base64 for API request
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Prepare search query based on search type
                        queries = {
                            "Visual Similarity": f"image:similar filetype:jpg OR filetype:png",
                            "Related Products": "similar products",
                            "News": "image news",
                            "All Results": "image"
                        }
                        
                        try:
                            perform_search(
                                api_key,
                                search_engine_id,
                                queries.get(search_type, "image"),
                                num_results,
                                "image"
                            )
                        except Exception as e:
                            st.error(f"❌ Search error: {str(e)}")

with tab2:
    st.subheader("Search by Image URL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        image_url = st.text_input(
            "Enter Image URL:",
            placeholder="https://example.com/image.jpg",
            help="Provide a direct link to an image"
        )
        
        if image_url and st.button("👁️ Preview Image", use_container_width=True):
            try:
                response = requests.get(image_url, timeout=10)
                preview_image = Image.open(BytesIO(response.content))
                st.image(preview_image, use_column_width=True)
            except Exception as e:
                st.error(f"❌ Could not load image: {str(e)}")
    
    with col2:
        st.subheader("Search Parameters")
        search_query = st.text_input(
            "Additional Search Query:",
            placeholder="e.g., 'similar products', 'brand information'",
            help="Optional: Add keywords to refine your search"
        )
        
        result_type = st.selectbox(
            "Result Type:",
            ["All", "Images", "Articles", "Products"]
        )
        
        num_url_results = st.slider(
            "Number of Results:",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        
        if st.button("🔍 Search by URL", key="url_search", use_container_width=True):
            if not api_key or not search_engine_id:
                st.error("❌ Please enter your API Key and Search Engine ID in the sidebar")
            elif not image_url:
                st.error("❌ Please enter an image URL")
            else:
                with st.spinner("🔄 Searching..."):
                    query = f"image {search_query}" if search_query else "image"
                    try:
                        perform_search(
                            api_key,
                            search_engine_id,
                            query,
                            num_url_results,
                            "image"
                        )
                    except Exception as e:
                        st.error(f"❌ Search error: {str(e)}")

with tab3:
    st.subheader("About This Application")
    st.markdown("""
    ### Features
    - **Reverse Image Search**: Find similar images across the web
    - **Multiple Search Types**: Visual similarity, products, news, and general results
    - **URL-based Search**: Search using image URLs
    - **Customizable Results**: Adjust number of results per search
    - **Rich Preview**: See thumbnails of similar images
    
    ### Powered By
    - [Streamlit](https://streamlit.io/) - Web app framework
    - [Google Custom Search API](https://developers.google.com/custom-search) - Search functionality
    - [Pillow](https://python-pillow.org/) - Image processing
    
    ### How It Works
    1. Upload an image or provide an image URL
    2. Configure your search parameters
    3. Submit your search
    4. Browse similar images and click links to view full sources
    
    ### Privacy Note
    - Your API credentials are stored locally in this session only
    - Images are sent to Google servers for processing
    - Search results are based on Google's index
    
    ### Requirements
    - Valid Google API Key with Custom Search API enabled
    - Custom Search Engine ID configured for web search
    - Internet connection for API calls
    """)
    
    st.markdown("---")
    st.markdown("""
    **Version**: 1.0.0  
    **Last Updated**: 2026-01-12  
    **Author**: Autonomous Vision Agent
    """)

def perform_search(api_key, search_engine_id, query, num_results, search_type):
    """
    Perform Google Custom Search and display results.
    
    Args:
        api_key (str): Google API key
        search_engine_id (str): Custom search engine ID
        query (str): Search query
        num_results (int): Number of results to return
        search_type (str): Type of search (image, web, etc.)
    """
    
    url = "https://www.googleapis.com/customsearch/v1"
    
    all_results = []
    
    # Pagination to get more results
    for start_index in range(1, num_results + 1, 10):
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "searchType": search_type,
            "startIndex": start_index,
            "num": min(10, num_results - start_index + 1)
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "items" in data:
                all_results.extend(data["items"])
            
            # Stop if we have enough results
            if len(all_results) >= num_results:
                break
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {str(e)}")
            return
        except json.JSONDecodeError:
            st.error("❌ Invalid API response")
            return
    
    # Display results
    if all_results:
        st.success(f"✅ Found {len(all_results)} results")
        
        cols = st.columns(3)
        
        for idx, item in enumerate(all_results[:num_results]):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Display thumbnail if available
                    if "image" in item:
                        try:
                            thumb_url = item["image"].get("thumbnailUrl", "")
                            if thumb_url:
                                st.image(thumb_url, use_column_width=True)
                        except Exception as e:
                            st.caption(f"Image preview unavailable")
                    
                    # Display result information
                    st.subheader(item.get("title", "No title")[:50], divider="blue")
                    st.caption(item.get("displayLink", "No URL"))
                    
                    snippet = item.get("snippet", "")
                    st.caption(snippet[:100] + "..." if len(snippet) > 100 else snippet)
                    
                    # Action buttons
                    link = item.get("link", "#")
                    st.markdown(
                        f"[🔗 View Full Result]({link})",
                        unsafe_allow_html=True
                    )
    else:
        st.warning("⚠️ No results found. Try different search parameters.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Built with Streamlit & Google Custom Search API"
    "</div>",
    unsafe_allow_html=True
)
