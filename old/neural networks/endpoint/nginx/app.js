const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// API endpoint URL
const API_ENDPOINT = '/api/predict/';

// Initialize canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 25; 
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

let drawing = false;

// Function to display status messages
function setStatus(message, isError = false) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = isError ? 'error' : 'success';
}

// Drawing events
function startDrawing(e) {
    e.preventDefault();
    drawing = true;
    draw(e);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;
    
    let x, y;
    
    if (e.type === 'mousemove') {
        const rect = canvas.getBoundingClientRect();
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    } else if (e.type === 'touchmove') {
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        x = touch.clientX - rect.left;
        y = touch.clientY - rect.top;
    } else {
        return;
    }
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Function to send the image to the server
async function predictDigit() {
    // Validate that something has been drawn
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let hasDrawing = false;
    
    for (let i = 0; i < imageData.length; i += 4) {
        if (imageData[i] > 0 || imageData[i+1] > 0 || imageData[i+2] > 0) {
            hasDrawing = true;
            break;
        }
    }
    
    if (!hasDrawing) {
        clearCanvas()
        setStatus('Please draw a digit first.', true);
        return;
    }
    
    // Show loading indicator
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('prediction').textContent = '';
    setStatus('Sending image...');
    
    try {
        // Convert canvas to image
        const dataURL = canvas.toDataURL('image/png');
        
        // Convert dataURL to Blob
        const response = await fetch(dataURL);
        const blob = await response.blob();
        
        // Show the image being sent (for debugging)
        const debugImg = document.getElementById('debug-img');
        debugImg.src = dataURL;
        debugImg.classList.remove('hidden');
        
        // Create FormData and add the image
        const formData = new FormData();
        formData.append('file', blob, 'digit.png');
        
        // Send the request with improved error handling
        const fetchResponse = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (fetchResponse.ok) {
            const data = await fetchResponse.json();
            document.getElementById('prediction').textContent = `${data.prediction}`;
            setStatus('Prediction completed.');
        } else {
            const errorText = await fetchResponse.text();
            throw new Error(`Server error (${fetchResponse.status}): ${errorText}`);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        setStatus(`Error: ${error.message}`, true);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
    clearWork()
}

function clearWork()
{
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Function to clear the canvas
function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = '';
    setStatus('');
    document.getElementById('debug-img').classList.add('hidden');
}

// Add events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Button events
document.getElementById('predict-btn').addEventListener('click', predictDigit);
document.getElementById('clear-btn').addEventListener('click', clearCanvas);