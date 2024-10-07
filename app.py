import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
import asyncio
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your trained model
model = None
def load_model_on_demand():
    global model
    if model is None:
        try:
            model = load_model('leaf_disease_model2.h5')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Mapping of class indices to class names
classes = {0: 'healthy', 1: 'powdery', 2: 'rusty'}

# Function to detect leaves in the image
def detect_leaves(image: np.ndarray) -> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([80, 255, 255]))
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_green_area = sum(cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000)
    
    return total_green_area > 0.01 * image.shape[0] * image.shape[1]

# Function to make predictions
async def predict_class(image: Image.Image) -> str:
    try:
        image_array = np.array(image)
        
        if not detect_leaves(image_array):
            return "no_leaf"
        
        image = image.resize((225, 225))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        load_model_on_demand()
        predictions = await asyncio.to_thread(model.predict, image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return classes[predicted_class]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise ValueError(f"Error during prediction: {str(e)}")

# Create a FastAPI app
app = FastAPI()

# Route for prediction
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {'image/jpeg', 'image/png', 'image/gif'}:
        raise HTTPException(status_code=400, detail='Invalid image file type. Allowed types are JPEG, PNG, and GIF.')

    try:
        image_data = await file.read()
        if len(image_data) > 5 * 1024 * 1024:  # 5 MB limit
            raise HTTPException(status_code=400, detail='Image file too large. Max size is 5 MB.')
        
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error opening image: {str(e)}")
        raise HTTPException(status_code=400, detail=f'Invalid image: {str(e)}')

    try:
        predicted_class = await predict_class(image)
        
        if predicted_class == "no_leaf":
            return JSONResponse(content={'message': 'No leaf detected in the image.'})
        
        return JSONResponse(content={'predicted_class': predicted_class})
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# To run the app: uvicorn app:app --reload
# To post image: curl -X POST http://127.0.0.1:8000/predict -F "file=@2.jpeg"