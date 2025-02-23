// Image Upload and Analysis Functions
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewContainer = document.querySelector('.preview-container');
            const previewImage = previewContainer.querySelector('img');
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            
            // Hide upload text and show preview
            document.querySelector('.upload-text').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

function removePreview() {
    const previewContainer = document.querySelector('.preview-container');
    const uploadInput = document.querySelector('#imageUpload');
    const uploadText = document.querySelector('.upload-text');
    
    previewContainer.style.display = 'none';
    uploadInput.value = '';
    uploadText.style.display = 'block';
}

function analyzeImage(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const analyzing = document.querySelector('.analyzing');
    const resultSection = document.querySelector('.result-section');
    
    // Show loading state
    analyzing.style.display = 'block';
    resultSection.style.display = 'none';
    
    fetch('/analyze_leaf', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        analyzing.style.display = 'none';
        resultSection.style.display = 'block';
        
        // Update results
        document.querySelector('#originalImage').src = data.original_image;
        document.querySelector('#processedImage').src = data.processed_image;
        document.querySelector('#detectionTitle').textContent = data.condition;
        document.querySelector('#detectionDescription').innerHTML = marked.parse(data.description);
        document.querySelector('#recommendations').innerHTML = marked.parse(data.recommendations);
    })
    .catch(error => {
        analyzing.style.display = 'none';
        alert('An error occurred during analysis. Please try again.');
        console.error('Error:', error);
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.querySelector('#imageUpload');
    const leafHealthForm = document.querySelector('#leafHealthForm');
    const removePreviewBtn = document.querySelector('.remove-preview');
    
    if (imageUpload) {
        imageUpload.addEventListener('change', handleImageUpload);
    }
    
    if (leafHealthForm) {
        leafHealthForm.addEventListener('submit', analyzeImage);
    }
    
    if (removePreviewBtn) {
        removePreviewBtn.addEventListener('click', removePreview);
    }
}); 