from flask import Flask, render_template, request, jsonify, send_file, url_for
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
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import io
import base64
import torch
from ultralytics import YOLO
import time
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
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

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ML models
try:
    # Load crop recommendation model
    crop_model = pickle.load(open(r'ML_Recommendation_Models\model.pkl', 'rb'))
    sc = pickle.load(open(r'ML_Recommendation_Models\standscaler.pkl', 'rb'))
    ms = pickle.load(open(r'ML_Recommendation_Models\minmaxscaler.pkl', 'rb'))
    print("Crop recommendation models loaded successfully!")
except Exception as e:
    print(f"Error loading crop models: {e}")
    crop_model = None

# Load YOLO model
try:
    yolo_model = YOLO('CV_Model/best.pt')
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Page routes
@app.route('/')
def home():
    return render_template('pages/home.html')

@app.route('/chat')
def chat():
    return render_template('pages/chat.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('pages/crop_recommendation.html')

@app.route('/leaf-analysis')
def leaf_analysis():
    return render_template('pages/leaf_health.html')

@app.route('/about')
def about():
    return render_template('pages/about.html')


def process_image_for_yolo(image_path):
    """
    Process image using YOLO model for leaf health detection
    """
    output_path = None
    try:
        if yolo_model is None:
            raise Exception("YOLO model not loaded")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read image")

        # Run YOLO detection
        results = yolo_model(image)  # Get predictions
        
        # Process results
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence
                confidence = box.conf[0].cpu().numpy()
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                class_name = r.names[cls]
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

        # Plot the results
        plotted_image = results[0].plot()
        
        # Generate a unique filename for the processed image
        timestamp = int(time.time() * 1000)
        output_filename = f'processed_{timestamp}_{os.path.basename(image_path)}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Save the processed image
        cv2.imwrite(output_path, plotted_image)

        # Get the most confident detection
        if detected_objects:
            best_detection = max(detected_objects, key=lambda x: x['confidence'])
            condition = best_detection['class']
            confidence = best_detection['confidence']
        else:
            condition = "No issues detected"
            confidence = 1.0

        return {
            'condition': condition,
            'confidence': confidence,
            'processed_image_path': output_path,
            'detections': detected_objects
        }
    except Exception as e:
        print(f"Error in YOLO processing: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback
        # Return a basic response if YOLO fails
        return {
            'condition': 'Error in processing',
            'confidence': 0,
            'processed_image_path': image_path,
            'detections': []
        }



# API routes
@app.route('/api/chat', methods=['POST'])
def chat_api():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
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
        
        # Generate response using Gemini
        prompt = f"""You are AgriBrain, an intelligent agricultural assistant. Help with this query:

Context: {context}
History: {history_context}
Query: {user_message}

Respond in the same language as the query (English/Urdu/Roman Urdu).
Use markdown formatting.
Include relevant references if available."""

        response = model.generate_content(prompt)
        
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
        print(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop-recommendation', methods=['POST'])
def crop_recommendation_api():
    try:
        data = request.json
        if not all(key in data for key in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']):
            return jsonify({'error': 'Missing required parameters'}), 400

        if crop_model is None:
            return jsonify({'error': 'Crop recommendation model not available'}), 503

        # Process input data
        feature_list = [
            float(data['Nitrogen']), 
            float(data['Phosphorus']), 
            float(data['Potassium']),
            float(data['Temperature']), 
            float(data['Humidity']), 
            float(data['Ph']), 
            float(data['Rainfall'])
        ]
        
        # Make prediction
        single_pred = np.array(feature_list).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)
        
        # Get crop name
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        predicted_crop = crop_dict.get(prediction[0], "Unknown crop")
        
        # Get detailed recommendation from Gemini
        prompt = f"""Based on these soil and weather parameters:
- Nitrogen: {data['Nitrogen']} mg/kg
- Phosphorus: {data['Phosphorus']} mg/kg
- Potassium: {data['Potassium']} mg/kg
- Temperature: {data['Temperature']}°C
- Humidity: {data['Humidity']}%
- pH: {data['Ph']}
- Rainfall: {data['Rainfall']} mm

The model recommends growing {predicted_crop}.

Provide:
1. Why this crop is suitable
2. Cultivation practices
3. Expected yield
4. Potential challenges
5. Best practices

Use markdown formatting with clear sections."""

        recommendation = model.generate_content(prompt)
        
        return jsonify({
            'predicted_crop': predicted_crop,
            'detailed_recommendation': recommendation.text
        })
        
    except Exception as e:
        print(f"Error in crop recommendation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-leaf', methods=['POST'])
def analyze_leaf_api():
    if yolo_model is None:
        return jsonify({'error': 'Leaf analysis model not available'}), 503

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_filepath)
        
        # Process with YOLO
        image = cv2.imread(uploaded_filepath)
        results = yolo_model(image)
        
        # Save processed image
        plotted_image = results[0].plot()
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(processed_filepath, plotted_image)
        
        # Get detection
        condition = "No issues detected"
        confidence = 1.0
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                condition = r.names[cls]
                confidence = conf
                break
        
        # Convert images to base64
        with open(uploaded_filepath, 'rb') as f:
            original_base64 = base64.b64encode(f.read()).decode()
        
        with open(processed_filepath, 'rb') as f:
            processed_base64 = base64.b64encode(f.read()).decode()
        
        # Clean up files
        os.remove(uploaded_filepath)
        os.remove(processed_filepath)
        
        return jsonify({
            'condition': condition,
            'confidence': confidence,
            'analysis': f"Detected {condition} with {confidence*100:.1f}% confidence",
            'recommendations': "Please consult an agricultural expert for detailed recommendations.",
            'original_image': f"data:image/jpeg;base64,{original_base64}",
            'processed_image': f"data:image/jpeg;base64,{processed_base64}"
        })
        
    except Exception as e:
        print(f"Error in leaf analysis: {e}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/analyze_leaf', methods=['POST'])
def analyze_leaf():
    uploaded_filepath = None
    processed_filepath = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'})
        
        # Save the uploaded file with a secure filename
        filename = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        unique_filename = f"{timestamp}_{filename}"
        uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Ensure the uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(uploaded_filepath)
        
        # Process the image with YOLO
        results = process_image_for_yolo(uploaded_filepath)
        processed_filepath = results['processed_image_path']
        
        # Read both original and processed images
        try:
            with open(uploaded_filepath, 'rb') as img_file:
                original_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            with open(processed_filepath, 'rb') as img_file:
                processed_image = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading images: {e}")
            return jsonify({'error': 'Error processing images'})
        
        # Generate analysis text based on results
        analysis = f"""## Leaf Health Analysis
- **Condition**: {results['condition']}
- **Confidence**: {results['confidence']*100:.1f}%"""

        # Generate LLM response based on detection
        prompt = f"""Based on the leaf health analysis:
- Condition: {results['condition']}
- Confidence: {results['confidence']*100:.1f}%

Please provide:
1. Detailed explanation of the detected condition
2. Specific treatment recommendations
3. Preventive measures
4. Long-term care instructions
5. Warning signs to watch for
6. When to seek expert help

Format the response using:
- Clear headings
- Bullet points
- Emojis for better readability
- Step-by-step instructions where needed
"""
        
        llm_response = model.generate_content(prompt)
        recommendations = llm_response.text
        
        response_data = {
            'condition': results['condition'],
            'analysis': analysis,
            'recommendations': recommendations,
            'original_image': f'data:image/jpeg;base64,{original_image}',
            'processed_image': f'data:image/jpeg;base64,{processed_image}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_leaf: {e}")
        return jsonify({'error': str(e)})
        
    finally:
        # Clean up files after sending the response
        try:
            if uploaded_filepath and os.path.exists(uploaded_filepath):
                os.remove(uploaded_filepath)
        except Exception as e:
            print(f"Error removing uploaded file: {e}")
            
        try:
            if processed_filepath and os.path.exists(processed_filepath):
                os.remove(processed_filepath)
        except Exception as e:
            print(f"Error removing processed file: {e}")
            
if __name__ == '__main__':
    # Initialize vector store
    if not initialize_vector_store():
        agri_pdf_dir = r"D:\Chatbot-Flask-App-Using-Gemini-API\AgriBrain Books"
        print(f"Processing PDFs from: {agri_pdf_dir}")
        
        loader = PyPDFDirectoryLoader(agri_pdf_dir)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=str(DB_DIRECTORY)
        )
        vector_store.persist()
        print("Vector store created successfully!")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 