import cv2
import numpy as np
import tensorflow as tf

# Load the JSON model architecture
with open('network_emotions.json', 'r') as json_file:
    json_saved_model = json_file.read()

# Load the model architecture from JSON
network_loaded = tf.keras.models.model_from_json(json_saved_model)

# Load the model weights
network_loaded.load_weights('weights_emotions.hdf5')

# Compile the loaded model
network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Print model summary
network_loaded.summary()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert grayscale image to RGB
        roi = cv2.cvtColor(image_gray[y:y + h, x:x + w], cv2.COLOR_GRAY2RGB)
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)  # Add a batch dimension
        prediction = network_loaded.predict(roi)

        if prediction is not None:
            result = np.argmax(prediction)
            print(emotions[result])
            cv2.putText(frame, emotions[result], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
