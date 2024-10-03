import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your trained model
try:
    model = load_model('leaf_disease_model2.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    exit(1)

# Mapping of class indices to class names
classes = {'healthy': 0, 'rusty': 2, 'powdery': 1}
icd = {v: k for k, v in classes.items()}

# Function to make predictions
async def predict_class(image: Image.Image) -> str:
    try:
        # Resize and preprocess the image to 225x225
        image = image.resize((225, 225))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions asynchronously
        predictions = await asyncio.to_thread(model.predict, image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return icd[predicted_class]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise ValueError(f"Error during prediction: {str(e)}")

# Create a FastAPI app
app = FastAPI()

# Route for prediction
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Check if the file is an image
    allowed_types = {'image/jpeg', 'image/png', 'image/gif'}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail='Invalid image file type. Allowed types are JPEG, PNG, and GIF.')

    try:
        # Open the image from the file (in memory)
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=400, detail='Image file too large. Max size is 10 MB.')
        
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error opening image: {str(e)}")
        raise HTTPException(status_code=400, detail=f'Invalid image: {str(e)}')

    try:
        # Predict the class of the image
        predicted_class = await predict_class(image)
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    # Return the result as JSON
    return JSONResponse(content={'predicted_class': predicted_class})

# To run the app, use the following command in your terminal:
# uvicorn app:app --reload
# use this command in your terminal to post image to the server curl -X POST http://127.0.0.1:8000/predict \  -F "file=@image_name.jpg"