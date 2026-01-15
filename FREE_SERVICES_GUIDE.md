# Free Services Guide for Autonomous Vision Agent

## Overview

This guide explains how to use the **Free Reverse Image Search** implementation that requires **NO API keys** and works perfectly on **Streamlit Cloud**.

## ✅ Features

- **Zero API Keys Required**: Works out-of-the-box without any configuration
- **Multiple Search Engines**: Uses SauceNAO, Yandex, and DuckDuckGo
- **Streamlit Cloud Ready**: Designed specifically for Streamlit Cloud deployment
- **User-Friendly Interface**: Clean, intuitive UI with real-time feedback
- **Error Handling**: Comprehensive error messages and troubleshooting tips

## 🔍 Supported Free Services

### 1. SauceNAO (Primary)
- **Type**: Reverse Image Search API
- **Free Tier**: 200 searches per day
- **Strengths**: 
  - High accuracy for anime/manga images
  - Provides similarity scores
  - Good for general images
  - Reliable API
- **Limitations**: 
  - 200 searches/day limit on free tier
  - May rate-limit if exceeded
- **No API Key Required**: Works without key (with reduced limits)

### 2. Yandex Images (Secondary)
- **Type**: Web-based reverse search
- **Free Tier**: No official limits
- **Strengths**:
  - No API key required
  - Excellent for general images
  - Good for product images
  - No rate limits (officially)
- **Limitations**:
  - May occasionally show CAPTCHA
  - HTML parsing can be less reliable
  - No similarity scores
- **No API Key Required**: Fully free

### 3. DuckDuckGo (Fallback)
- **Type**: Image search via URL only
- **Free Tier**: No limits
- **Strengths**:
  - Privacy-focused
  - No authentication needed
  - Simple to use
- **Limitations**:
  - URL-based only (no file uploads)
  - Limited reverse search capabilities
  - Less accurate results
- **No API Key Required**: Fully free

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the App**:
```bash
streamlit run app-free-streamlit.py
```

3. **Open Browser**: Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**:
```bash
git add app-free-streamlit.py
git commit -m "Add free services Streamlit app"
git push origin main
```

2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app-free-streamlit.py` as the main file
   - Click "Deploy"

3. **No Configuration Needed**: The app works immediately without any environment variables or secrets!

## 📝 Usage Instructions

### Basic Search

1. **Upload an Image**:
   - Click "Browse files" or drag-and-drop
   - Supported formats: JPG, JPEG, PNG, WEBP, GIF, BMP
   - Recommended size: Under 5MB

2. **Adjust Settings** (Optional):
   - Number of results per engine (1-10)
   - Select which search engines to use
   - Add optional search context

3. **Click "Search Now"**:
   - The app searches across all enabled engines
   - Results appear as they're found
   - Each result shows title, URL, and similarity (if available)

### Advanced Features

#### Search Context
Add keywords to refine your search:
- Example: Upload a watch and type "vintage Rolex"
- This helps engines find more relevant results

#### Engine Selection
Enable/disable specific engines:
- Use SauceNAO for anime/manga or similarity scores
- Use Yandex for general images
- Use DuckDuckGo as fallback

#### Result Interpretation
- **Similarity Score** (SauceNAO only): Higher = more similar
- **URL**: Click to visit the source page
- **Metadata**: Additional info about the match

## 🔧 Troubleshooting

### Issue: "Button does nothing"

**Solution**: This is the original bug you mentioned. The new app fixes this by:
- Using unique button keys
- Proper event handling
- Clear visual feedback during search

### Issue: "Yandex shows CAPTCHA"

**Solution**: 
- Yandex may occasionally require CAPTCHA verification
- Try disabling Yandex and using only SauceNAO
- Wait a few minutes and try again
- Use a different IP address if on VPN

### Issue: "SauceNAO limit exceeded"

**Solution**:
- Free tier allows 200 searches/day
- If exceeded, wait 24 hours or get a free API key
- Sign up at [SauceNAO](https://saucenao.com/user.php) for higher limits

### Issue: "No results found"

**Solution**:
- Try a different image format (JPG, PNG)
- Reduce image size (under 5MB)
- Enable more search engines
- Check internet connection
- Try with a different image

### Issue: "Error during search"

**Solution**:
1. Check the error message for specifics
2. Verify image format is supported
3. Try with fewer search engines
4. Check your internet connection
5. Look at the troubleshooting tips shown in the app

## 📊 Comparison with Paid Services

| Feature | Free Services | Bing API | Google API |
|---------|--------------|----------|------------|
| Cost | Free | Paid | Paid |
| API Key | Not required | Required | Required |
| Daily Limits | 200 (SauceNAO) | Varies | Varies |
| Similarity Scores | Yes (SauceNAO) | Yes | No |
| Setup Time | 0 minutes | 30+ minutes | 30+ minutes |
| Streamlit Cloud | Works out-of-box | Requires secrets | Requires secrets |
| Accuracy | Good | Very Good | Excellent |

## 🎯 Best Practices

### For Best Results:

1. **Use High-Quality Images**: Clear, well-lit images work best
2. **Multiple Engines**: Enable all engines for comprehensive results
3. **Add Context**: Use the search context field for better relevance
4. **Try Different Formats**: If one format fails, try another
5. **Be Patient**: Some engines take longer than others

### For Performance:

1. **Limit Results**: Use 3-5 results per engine for faster searches
2. **Disable Unused Engines**: Turn off engines you don't need
3. **Smaller Images**: Under 2MB uploads process faster
4. **Stable Connection**: Ensure good internet connectivity

### For Reliability:

1. **SauceNAO First**: Use SauceNAO as primary engine
2. **Yandex as Backup**: Enable Yandex for additional results
3. **Handle Errors**: Read error messages carefully
4. **Retry Mechanism**: Try again if first attempt fails

## 🛠️ Technical Details

### How It Works

1. **Image Upload**: Streamlit file uploader saves image to temporary file
2. **Search Execution**: Each search engine processes the image independently
3. **Result Aggregation**: Results are collected and displayed by engine
4. **Cleanup**: Temporary files are deleted after search

### Rate Limiting

- Built-in 0.5 second delay between engine searches
- Prevents overwhelming free APIs
- Respects service limitations

### Error Handling

- Individual engine failures don't stop other engines
- Clear error messages for each failure
- Graceful degradation when engines are unavailable

## 📈 Future Improvements

Potential enhancements:

- [ ] Add more free search engines
- [ ] Implement result caching
- [ ] Add image preview in results
- [ ] Support for batch uploads
- [ ] Export results to CSV/JSON
- [ ] Advanced filtering options
- [ ] Similarity score visualization

## 🤝 Contributing

To contribute improvements:

1. Test thoroughly with various image types
2. Ensure no API keys are required
3. Test on Streamlit Cloud
4. Update documentation
5. Submit pull request

## 📄 License

Same as parent project (MIT License)

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [SauceNAO API](https://saucenao.com/api.php)
- [Yandex Images](https://yandex.com/images/)
- [DuckDuckGo](https://duckduckgo.com/)

---

**Last Updated**: 2026-01-15  
**Version**: 1.0.0  
**Status**: ✅ Production Ready