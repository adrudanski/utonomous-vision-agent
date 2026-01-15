# Autonomous Vision Agent - Free Services Version

## 🚀 Quick Start - NO API Keys Required!

This version uses **100% free services** and works immediately on Streamlit Cloud without any configuration.

### ✨ What's New

- ✅ **No API Keys Needed**: Works out-of-the-box
- ✅ **Fixed Button Issue**: Upload button now works properly
- ✅ **Multiple Search Engines**: SauceNAO, Yandex, DuckDuckGo
- ✅ **Streamlit Cloud Ready**: Deploy in seconds
- ✅ **Clean UI**: Modern, intuitive interface

## 🎯 How to Use

### Option 1: Streamlit Cloud (Recommended)

1. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub repository
   - Select `app-free-streamlit.py` as main file
   - Click "Deploy"

2. **That's it!** The app works immediately.

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app-free-streamlit.py

# Open http://localhost:8501
```

## 📸 How to Search

1. **Upload an image** (JPG, PNG, WEBP, etc.)
2. **Adjust settings** (optional - number of results, search engines)
3. **Click "Search Now"** - Results appear automatically!
4. **Browse results** by engine, click links to view sources

## 🔍 Search Engines

| Engine | Free Tier | Best For | Similarity Scores |
|--------|-----------|----------|-------------------|
| **SauceNAO** | 200/day | Anime, Art, General | ✅ Yes |
| **Yandex** | Unlimited | General, Products | ❌ No |
| **DuckDuckGo** | Unlimited | Basic Search | ❌ No |

## 🛠️ Troubleshooting

### "Button does nothing" 
✅ **Fixed!** This version has a working button with proper event handling.

### "Yandex shows CAPTCHA"
- Disable Yandex in settings
- Use SauceNAO instead
- Wait a few minutes and try again

### "No results found"
- Try a different image format
- Enable more search engines
- Check internet connection
- Upload a smaller image

### "Search takes too long"
- Reduce number of results (try 3 instead of 10)
- Disable one or two search engines
- Use a smaller image file

## 📚 Full Documentation

See [FREE_SERVICES_GUIDE.md](FREE_SERVICES_GUIDE.md) for:
- Detailed feature descriptions
- Advanced usage tips
- Best practices
- Technical details
- Troubleshooting guide

## 🆚 Compare Versions

| Version | API Keys | Setup Time | Streamlit Cloud | Button Issue |
|---------|----------|------------|-----------------|--------------|
| **app-free-streamlit.py** | ❌ None | 0 min | ✅ Works | ✅ Fixed |
| app.py | Bing Key | 30+ min | Needs secrets | ❌ Broken |
| app-google.py | Google Key | 30+ min | Needs secrets | ❌ Broken |

## 🎉 Key Benefits

1. **Zero Configuration**: No API keys, no secrets, no setup
2. **Instant Deployment**: Works on Streamlit Cloud immediately
3. **Fixed Issues**: Upload button now works properly
4. **Multiple Engines**: Get results from 3 different sources
5. **Free Forever**: No hidden costs or subscriptions

## 📦 Files Included

- `app-free-streamlit.py` - Main Streamlit app (USE THIS ONE)
- `FREE_SERVICES_GUIDE.md` - Comprehensive guide
- `app-free.py` - Command-line version (original)
- `requirements.txt` - Dependencies

## 🔧 Technical Stack

- **Streamlit** - Web UI framework
- **Requests** - HTTP library for search APIs
- **Pillow** - Image processing
- **Python 3.8+** - Runtime environment

## 🤝 Feedback

If you encounter any issues:
1. Check [FREE_SERVICES_GUIDE.md](FREE_SERVICES_GUIDE.md) for solutions
2. Review the troubleshooting section above
3. Open an issue on GitHub

## 📄 License

MIT License - Same as parent project

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: 2026-01-15

**Happy Searching! 🎉**