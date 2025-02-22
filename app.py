from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize global variables for vector store
vector_store = None
conversation_history = {}

# Define persistent storage paths
PERSIST_DIRECTORY = Path("persistent_storage")
DB_DIRECTORY = PERSIST_DIRECTORY / "vectordb"

# Create directories if they don't exist
PERSIST_DIRECTORY.mkdir(exist_ok=True)
DB_DIRECTORY.mkdir(exist_ok=True)

# Load ML models for crop recommendation
try:
    crop_model = pickle.load(open(r'ML_Recommendation_Models\model.pkl', 'rb'))
    sc = pickle.load(open(r'ML_Recommendation_Models\standscaler.pkl', 'rb'))
    ms = pickle.load(open(r'ML_Recommendation_Models\minmaxscaler.pkl', 'rb'))
except:
    print("Warning: Crop recommendation models not found")
    crop_model = None

def get_crop_recommendation(N, P, K, temp, humidity, ph, rainfall):
    """Get crop recommendation based on soil and weather parameters"""
    if crop_model is None:
        return "Crop recommendation model not available"
    
    try:
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)
        
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        
        predicted_crop = crop_dict.get(prediction[0], "Unknown crop")
        
        # Generate LLM response for the crop recommendation
        prompt = f"""Based on the following soil and weather parameters:
- Nitrogen (N): {N} mg/kg
- Phosphorus (P): {P} mg/kg
- Potassium (K): {K} mg/kg
- Temperature: {temp}°C
- Humidity: {humidity}%
- pH: {ph}
- Rainfall: {rainfall} mm

The ML model has predicted {predicted_crop} as the most suitable crop.

LANGUAGE DETECTION AND RESPONSE RULES:
1. First, analyze the script used in the question:
   * Latin script (abc) → Respond in English
   * Nastaliq script (ابت) → Respond in Urdu
   * Roman Urdu (abc + Urdu words) → Respond in Roman Urdu


2. Response Rules:
   * Respond ONLY in the detected language
   * Use consistent script throughout
   * Use local crop names and terms
   * Convert measurements to local standards
   * Do not mix languages

Provide this information:
1. Why this crop suits these conditions
2. Key cultivation practices
3. Care schedule
4. Main challenges and solutions
5. Expected yield
6. Sustainable methods

Use:
- Headings in detected language
- Bullet points
- Emojis
- Local farming terms
- Consistent script

Verify before responding:
- Single language/script used
- Local terminology correct
- Measurements in local format
- Farmer-friendly language."""

        llm_response = model.generate_content(prompt)
        
        return {
            "predicted_crop": predicted_crop,
            "detailed_recommendation": llm_response.text
        }
        
    except Exception as e:
        return f"Error in crop recommendation: {str(e)}"

def initialize_vector_store():
    """Initialize or load existing vector store"""
    global vector_store
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # If vector store exists, load it
    if os.path.exists(DB_DIRECTORY) and os.listdir(DB_DIRECTORY):
        print("Loading existing vector store...")
        vector_store = Chroma(persist_directory=str(DB_DIRECTORY), embedding_function=embeddings)
        return True
    return False

def get_relevant_chunks(query, k=5):
    """Get relevant chunks from vector store"""
    global vector_store
    
    if vector_store is None:
        initialize_vector_store()
        if vector_store is None:
            return []
    
    # Search for relevant chunks
    results = vector_store.similarity_search(query, k=k)
    return [(doc.page_content, doc.metadata) for doc in results]

def format_conversation_history(history):
    """Format conversation history for context"""
    if not history:
        return ""
    
    formatted = "\nPrevious conversation:\n"
    for entry in history:
        formatted += f"User: {entry['user']}\n"
        formatted += f"Assistant: {entry['assistant']}\n"
    return formatted

@app.route('/')
def home():
    # Initialize vector store when app starts
    initialize_vector_store()
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    try:
        data = request.json
        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['Ph'])
        rainfall = float(data['Rainfall'])
        
        result = get_crop_recommendation(N, P, K, temp, humidity, ph, rainfall)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = request.remote_addr
        
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Get relevant chunks with metadata
        relevant_chunks = get_relevant_chunks(user_message)
        context = ""
        references = []
        
        for content, metadata in relevant_chunks:
            context += content + "\n"
            if metadata.get('source'):
                references.append(metadata.get('source'))
        
        history_context = format_conversation_history(conversation_history[session_id][-3:])
        
        # Enhanced prompt with specific instructions
        prompt = f"""You are AgriBrain, an intelligent agricultural assistant powered by advanced AI and comprehensive agricultural knowledge. You help farmers with their questions about agriculture, farming techniques, crop management, and related topics.

Context from agricultural documents:
{context}

Previous Conversation:
{history_context}

Current Question: {user_message}

LANGUAGE DETECTION AND RESPONSE RULES:
1. First, analyze the script used in the question:
   * English Letters (abc) → Respond in English
   * Nastaliq script (ابت) → Respond in Urdu
   * Roman Urdu (abc + Urdu words) → Respond in Roman Urdu

2. Language Matching Rules:
   * Use ONLY the detected language
   * Do not mix scripts or languages
   * Maintain consistent script throughout
   * Use language-appropriate terminology
   * Keep formatting in the same language

Response Structure:
1. Content Organization:
   - Clear headings in detected language
   - Organized bullet points
   - Tables if needed
   - Emojis for visual aid
   - Step-by-step instructions

2. Information Quality:
   - Scientific accuracy
   - Practical advice
   - Local context
   - Current best practices
   - Safety considerations

3. Technical Details:
   - Specific measurements
   - Timing information
   - Cost considerations
   - Resource requirements
   - Implementation steps

4. Sources and References:
   - Cite relevant sources
   - Include case studies
   - Reference local practices

References to cite: {references}

FINAL CHECK:
1. Is the response in the EXACT SAME SCRIPT as the question?
2. Are all terms and measurements in the appropriate language?
3. Is the language consistent throughout?
4. Are local/regional terms used correctly?

Respond following these guidelines while maintaining strict language consistency."""
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        # Update conversation history
        conversation_history[session_id].append({
            'user': user_message,
            'assistant': response.text
        })
        
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
        
        return jsonify({
            'response': response.text,
            'history': conversation_history[session_id],
            'references': references
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Process the agriculture PDFs directory before starting the app
    agri_pdf_dir = r"D:\Chatbot-Flask-App-Using-Gemini-API\AgriBrain Books"
    print("Processing agriculture PDFs...")
    
    # Initialize vector store
    if not initialize_vector_store():
        # Only process PDFs if vector store doesn't exist
        loader = PyPDFDirectoryLoader(agri_pdf_dir)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=str(DB_DIRECTORY)
        )
        vector_store.persist()
        print("Vector store created and persisted successfully!")
    
    # Start the Flask app
    app.run(debug=True) 