import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sqlite3
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained emotion model
print("Loading the trained emotion model...")
model = load_model('face_emotionModel.h5')
print("Model loaded successfully!")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Personalized messages for each emotion
emotion_messages = {
    'Angry': 'You are angry. Would you like to take a break and cool off?',
    'Disgust': 'You look disgusted. What\'s bothering you?',
    'Fear': 'You appear fearful. Remember, you can handle this!',
    'Happy': 'You are happy! That\'s wonderful to see!',
    'Sad': 'You seem sad. Is everything okay?',
    'Surprise': 'You look surprised! What\'s the matter?',
    'Neutral': 'You have a neutral expression. How are you feeling?'
}


# Initialize SQLite database
def init_db():
    """Create database and tables if they don't exist"""
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     name
                     TEXT
                     NOT
                     NULL,
                     email
                     TEXT
                     NOT
                     NULL,
                     student_id
                     TEXT
                     NOT
                     NULL,
                     emotion
                     TEXT
                     NOT
                     NULL,
                     confidence
                     REAL
                     NOT
                     NULL,
                     timestamp
                     DATETIME
                     DEFAULT
                     CURRENT_TIMESTAMP,
                     image_path
                     TEXT
                 )''')

    conn.commit()
    conn.close()


# Initialize database on startup
init_db()


def preprocess_image(image_array):
    """Convert image to grayscale 48x48 for model prediction"""
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))

    # Normalize to 0-1
    normalized = resized / 255.0

    # Add channel dimension
    img_array = np.expand_dims(normalized, axis=-1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_emotion(image_array):
    """Predict emotion from image array"""
    processed_img = preprocess_image(image_array)
    predictions = model.predict(processed_img, verbose=0)
    emotion_index = np.argmax(predictions[0])
    confidence = float(predictions[0][emotion_index])
    emotion = emotion_labels[emotion_index]

    return emotion, confidence


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and emotion prediction"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Get form data
        file = request.files['image']
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        student_id = request.form.get('student_id', '').strip()

        # Validate inputs
        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if not name or not email or not student_id:
            return jsonify({'error': 'Please fill in all fields'}), 400

        # Read image file
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)

        # Ensure it's in RGB/BGR format
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

        # Predict emotion
        emotion, confidence = predict_emotion(image_array)
        message = emotion_messages[emotion]

        # Save image to uploads folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the image
        image.save(filepath)

        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        c.execute('''INSERT INTO users (name, email, student_id, emotion, confidence, image_path)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (name, email, student_id, emotion, confidence, filepath))

        conn.commit()
        conn.close()

        # Return prediction result
        return jsonify({
            'emotion': emotion,
            'confidence': f'{confidence * 100:.2f}%',
            'message': message,
            'name': name,
            'email': email,
            'student_id': student_id
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Retrieve user submission history"""
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        c.execute('''SELECT name, email, student_id, emotion, confidence, timestamp
                     FROM users
                     ORDER BY timestamp DESC LIMIT 10''')

        rows = c.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'name': row[0],
                'email': row[1],
                'student_id': row[2],
                'emotion': row[3],
                'confidence': f'{row[4] * 100:.2f}%',
                'timestamp': row[5]
            })

        return jsonify(history)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask web server...")
    app.run(debug=False, host='0.0.0.0', port=10000)
