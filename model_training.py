import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import os

# Emotion labels mapping
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Loading FER2013 dataset...")

# Load the CSV file
data = pd.read_csv('fer2013.csv')

# Separate the data
X = []
y = []

# Process each row in the dataset
for index, row in data.iterrows():
    # Convert pixel string to numpy array
    pixels = np.array(row['pixels'].split(), dtype='float32')

    # Reshape to 48x48 image
    image = pixels.reshape(48, 48)

    # Normalize pixels to 0-1 range
    image = image / 255.0

    X.append(image)
    y.append(row['emotion'])

    # Print progress every 5000 images
    if (index + 1) % 5000 == 0:
        print(f"Processed {index + 1} images...")

print(f"Total images loaded: {len(X)}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension (48, 48, 1) for grayscale
X = X.reshape(X.shape[0], 48, 48, 1)

print(f"Dataset shape: {X.shape}")
print(f"Building CNN model...")

# Build the CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Second convolutional block
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Third convolutional block
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten and dense layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

# Data augmentation to improve model robustness
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

print("Training the model...")

# Train the model
history = model.fit(
    train_datagen.flow(X, y, batch_size=32),
    epochs=50,
    steps_per_epoch=len(X) // 32,
    verbose=1
)

print("Saving the trained model...")

# Save the model
model.save('face_emotionModel.h5')
print("Model saved as 'face_emotionModel.h5'")

# Print training summary
print(f"\nTraining completed!")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
