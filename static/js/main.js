let conversationHistory = [];
marked.setOptions({
    breaks: true,
    gfm: true
});

function appendMessage(message, isUser, references = []) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'user-message' : 'bot-message';
    
    let messageContent = '';
    if (isUser) {
        messageContent = `<i class="fas fa-user"></i><div class="message-content">${message}</div>`;
    } else {
        messageContent = `
            <i class="fas fa-robot"></i>
            <div class="message-content">
                ${marked.parse(message)}
                ${references && references.length > 0 ? `
                    <div class="references">
                        <h4>📚 References:</h4>
                        <ul>
                            ${references.map(ref => `<li>${ref}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageContent;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    
    if (!message) return;

    // Add user message
    appendMessage(message, true);
    userInput.value = '';
    userInput.disabled = true;

    // Add typing indicator
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'bot-message typing-message';
    typingDiv.innerHTML = `
        <i class="fas fa-robot"></i>
        <div class="typing">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': Date.now().toString()
            },
            body: JSON.stringify({ 
                message: message,
                history: conversationHistory
            })
        });
        
        const result = await response.json();
        
        // Remove typing indicator
        typingDiv.remove();
        
        if (result.error) {
            appendMessage('Error: ' + result.error, false);
        } else {
            appendMessage(result.response, false, result.references);
            conversationHistory = result.history;
        }
    } catch (error) {
        // Remove typing indicator
        typingDiv.remove();
        
        console.error('Error:', error);
        appendMessage('Sorry, I encountered an error. Please try again.', false);
    } finally {
        userInput.disabled = false;
        userInput.focus();
    }
}

async function getCropRecommendation(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/crop_recommendation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        const resultDiv = document.getElementById('recommendationResult');
        
        if (result.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
            appendMessage(`Error getting crop recommendation: ${result.error}`, false);
        } else {
            form.reset();
            
            const recommendationMessage = `# 🌱 Crop Recommendation Results

## Soil and Weather Parameters:
- Nitrogen (N): ${data.Nitrogen} mg/kg
- Phosphorus (P): ${data.Phosphorus} mg/kg
- Potassium (K): ${data.Potassium} mg/kg
- Temperature: ${data.Temperature}°C
- Humidity: ${data.Humidity}%
- pH: ${data.Ph}
- Rainfall: ${data.Rainfall} mm

## 🎯 Recommended Crop: ${result.predicted_crop}

${result.detailed_recommendation}`;

            appendMessage(recommendationMessage, false);
            
            document.getElementById('chat-section').scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error('Error:', error);
        appendMessage('Sorry, I encountered an error while getting the crop recommendation. Please try again.', false);
    }
}

function askQuestion(question) {
    document.getElementById('userInput').value = question;
    sendMessage();
}

function previewImage(input) {
    const preview = document.getElementById('imagePreview');
    const previewBox = document.querySelector('.preview-box');
    const placeholder = document.querySelector('.upload-placeholder');

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewBox.classList.remove('d-none');
            placeholder.style.display = 'none';
        }
        reader.readAsDataURL(input.files[0]);
    }
}

function removePreview() {
    const preview = document.getElementById('imagePreview');
    const previewBox = document.querySelector('.preview-box');
    const placeholder = document.querySelector('.upload-placeholder');
    const input = document.getElementById('leafImage');

    preview.src = '#';
    previewBox.classList.add('d-none');
    placeholder.style.display = 'block';
    input.value = '';
}

// Simplified version for quick fix
function handleImageUpload(input) {
    const preview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.preview-container');
    const placeholder = document.querySelector('.upload-placeholder');

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
            placeholder.style.display = 'none';
        };
        reader.readAsDataURL(input.files[0]);
    }
}

async function analyzeLeaf(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitButton = form.querySelector('button[type="submit"]');
    const resultSection = document.querySelector('.result-section');
    const analysisResult = document.getElementById('analysisResult');
    const placeholder = document.getElementById('placeholderResult');
    
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        const formData = new FormData(form);
        const response = await fetch('/api/analyze-leaf', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            placeholder.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
            placeholder.style.display = 'block';
            analysisResult.classList.add('d-none');
        } else {
            // Show results
            placeholder.style.display = 'none';
            analysisResult.classList.remove('d-none');
            
            // Set images directly
            document.getElementById('originalImage').src = result.original_image;
            document.getElementById('processedImage').src = result.processed_image;
            
            // Update text
            document.querySelector('.detection-title').textContent = 
                `Detected: ${result.condition} (${(result.confidence * 100).toFixed(1)}% confidence)`;
            document.querySelector('.detection-description').innerHTML = result.analysis;
            document.querySelector('.recommendations-content').innerHTML = result.recommendations;
            
            resultSection.style.display = 'block';
            analysisResult.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        placeholder.innerHTML = '<div class="alert alert-danger">Error during analysis. Please try again.</div>';
        placeholder.style.display = 'block';
        analysisResult.classList.add('d-none');
    } finally {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-microscope"></i> Analyze Leaf';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Chat input Enter key handler
    document.getElementById('userInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Leaf analysis form submit handler
    document.getElementById('leafHealthForm').addEventListener('submit', analyzeLeaf);

    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    const imageUpload = document.querySelector('#imageUpload');
    const leafHealthForm = document.querySelector('#leafHealthForm');
    const removePreviewBtn = document.querySelector('.remove-preview');
    
    if (imageUpload) {
        imageUpload.addEventListener('change', handleImageUpload);
    }
    
    if (leafHealthForm) {
        leafHealthForm.addEventListener('submit', analyzeLeaf);
    }
    
    if (removePreviewBtn) {
        removePreviewBtn.addEventListener('click', removePreview);
    }

    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    const uploadBox = document.getElementById('uploadBox');
    const imageUpload = document.getElementById('imageUpload');

    if (uploadBox) {
        uploadBox.addEventListener('click', () => {
            imageUpload.click();
        });

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('drag-over');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('drag-over');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                imageUpload.files = e.dataTransfer.files;
                handleImageUpload(imageUpload);
            }
        });
    }

    if (imageUpload) {
        imageUpload.addEventListener('change', function() {
            handleImageUpload(this);
        });
    }
}); 