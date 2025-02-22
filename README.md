# 🌱 AgriBrain - Intelligent Agricultural Assistant

AgriBrain is a multilingual AI-powered agricultural assistant that helps farmers with crop recommendations, farming techniques, and agricultural best practices. Built using Flask and Google's Gemini API, it provides personalized farming advice in multiple languages including English, Urdu, and Roman Urdu.

## ✨ Features

- 🤖 AI-powered agricultural assistance
- 🌍 Multilingual support (English, Urdu, Roman Urdu)
- 🎯 ML-based crop recommendations
- 📚 RAG (Retrieval Augmented Generation) for accurate information
- 💬 Interactive chat interface
- 📊 Soil parameter analysis
- 🔄 Contextual responses with memory
- 📱 Mobile-responsive design

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: Google Gemini API, Langchain
- **Vector Store**: Chroma DB
- **Embeddings**: HuggingFace Sentence Transformers
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **ML Models**: Scikit-learn (for crop recommendation)

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Required Python packages (see requirements.txt)
- Minimum 4GB RAM
- Storage space for vector database

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Engr-Jalal-Saleem/RAG-Based-Agricultural-Chatbot-Flask-App-Using-Gemini-API
cd AgriBrain
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a .env file with:
```env
SECRET_KEY="your_secret_key"
GEMINI_API_KEY="your_gemini_api_key"
FLASK_ENV=development
```

5. Place ML models in the correct directory:
```
ML_Recommendation_Models/
├── model.pkl
├── standscaler.pkl
└── minmaxscaler.pkl
```

6. Add agricultural documents:
```
AgriBrain Books/
└── [your agricultural PDF documents]
```

## 🚀 Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## 💡 Usage

### Chat Interface
- Type your agricultural questions in any supported language
- Get AI-powered responses in the same language
- View references and sources used

### Crop Recommendation
1. Enter soil parameters:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - Temperature
   - Humidity
   - pH
   - Rainfall

2. Get personalized recommendations:
   - Suitable crop suggestions
   - Cultivation practices
   - Care schedule
   - Expected yield
   - Challenges and solutions

## 🌟 Features in Detail

### Multilingual Support
- Automatic language detection
- Consistent language responses
- Support for:
  - English
  - Urdu (Nastaliq script)
  - Roman Urdu

### RAG Implementation
- Document chunking and embedding
- Semantic search
- Context-aware responses
- Source citations

### ML-Based Recommendations
- Soil parameter analysis
- Weather condition consideration
- Crop suitability prediction
- Detailed cultivation guidance

## 🔧 Configuration

### Vector Store
- Located in: `persistent_storage/vectordb/`
- Automatically initialized on first run
- Persists embeddings for faster responses

### Agricultural Documents
- Place PDF documents in: `AgriBrain Books/`
- Supports multiple documents
- Automatically processed on startup

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for AI capabilities
- Langchain for RAG implementation
- HuggingFace for embeddings
- Flask community for the web framework
- Agricultural experts for domain knowledge

## 📞 Support

For support, email jalalsaleem786@gmail.com or open an issue in the repository.

## 🔮 Future Enhancements

- Additional language support
- Image-based disease detection
- Weather integration
- Market price predictions
- Mobile app development

---
Made with ❤️ for farmers worldwide 
