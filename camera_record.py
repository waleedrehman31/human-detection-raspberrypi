import cv2
from datetime import datetime
import time
import threading
import requests

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for default camera, change to 1 for a USB camera

# Load pre-trained human detection model (Haar cascade)
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Server configuration
server_url = "https://yourhostingerdomain.com/upload_endpoint"  # Replace with your server endpoint
api_key = "your_api_key_here"  # Replace with your actual API key if needed

# Function to upload a file to the server
def upload_file(file_path, server_url, api_key):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.post(server_url, files=files, headers=headers)
        
    if response.status_code == 200:
        print(f"File uploaded successfully: {file_path}")
    else:
        print(f"Failed to upload file. Status code: {response.status_code}, Response: {response.text}")

# Function to detect human and capture image/video
def detect_and_record():
    recording = False
    out = None
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        # If humans are detected, capture image and start recording
        if len(humans) > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = f"/home/pi/detected_image_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image captured: {image_path}")
            upload_file(image_path, server_url, api_key)
            
            if not recording:
                # Start recording video
                video_path = f"/home/pi/detected_video_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
                recording = True
                start_time = time.time()
            
            # Draw rectangle around detected humans
            for (x, y, w, h) in humans:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # Stop recording if no human is detected for more than 5 seconds
            if recording and time.time() - start_time > 5:
                recording = False
                if out:
                    out.release()
                    print(f"Video recording stopped: {video_path}")
                    upload_file(video_path, server_url, api_key)
        
        if recording and out:
            out.write(frame)
        
        # Display the streaming video
        cv2.imshow('Live Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    camera.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Run the human detection and recording in a separate thread
if __name__ == "__main__":
    threading.Thread(target=detect_and_record).start()