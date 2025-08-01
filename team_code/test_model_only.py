import cv2
import os
import time
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import socket
import json

# UDP socket setup
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ('10.227.68.63', 9999)  # Update with your PC's IP

# Class definitions for 4 classes
class_dict = {
    0: "safe driving",
    1: "using phone", 
    2: "reaching back",
    3: "sleepy"
}

class_names = {
    0: "c0",
    1: "c1", 
    2: "c2",
    3: "c3"
}

# Model setup
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
print("Loading model...")
model.load_state_dict(torch.load("best_model_4class_improved.pth", map_location=torch.device('cpu')))
print("Model loaded successfully")
model.eval()

if torch.cuda.is_available():
    print("CUDA available, moving model to GPU")
    model.cuda()
else:
    print("CUDA not available, using CPU")
print("Model setup complete")

# Transforms
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def show_camera():
    print("Starting camera function...")
    video_capture = cv2.VideoCapture(0)
    print("VideoCapture object created")
    
    if video_capture.isOpened():
        print("Camera opened successfully, starting capture loop...")
        print("Press 'q' to quit")
        try:
            frame_count = 0
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("Failed to capture frame")
                    break
                
                print("Processing frame...")
                im = transform(Image.fromarray(frame)).unsqueeze(0)
                print("Running inference...")
                
                if torch.cuda.is_available():
                    output = model(im.cuda())
                else:
                    output = model(im)
                print("Inference completed")
                
                proba = nn.Softmax(dim=1)(output)
                proba = [round(float(elem), 4) for elem in proba[0]]
                pr = max(proba)
                idx = proba.index(pr)
                
                # Show all probabilities
                print(f"All probabilities: {proba}")
                print(f"Predicted class: {idx}")
                
                cl = class_dict[idx]
                cn = class_names[idx]
                print(f"Prediction: {cn}({cl}) @ prob {pr}")
                
                # Create proper JSON message for unified_risk_manager
                message = {
                    "driver_state": cl.replace(" ", "_"),  # Convert "safe driving" to "safe_driving"
                    "confidence": float(pr),
                    "timestamp": time.time()
                }
                
                # Send JSON-encoded message
                json_message = json.dumps(message)
                s.sendto(json_message.encode(), addr)
                print(f"Sent: {json_message}")
                
                # Display the frame with predictions
                display_frame = frame.copy()
                
                # Add prediction text to the frame
                prediction_text = f"Class {idx}: {cl}"
                confidence_text = f"Confidence: {pr:.3f}"
                frame_count_text = f"Frame: {frame_count}"
                
                # Show all class probabilities
                prob_text = f"P0:{proba[0]:.3f} P1:{proba[1]:.3f} P2:{proba[2]:.3f} P3:{proba[3]:.3f}"
                
                # Draw text on frame
                cv2.putText(display_frame, prediction_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, confidence_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, frame_count_text, (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, prob_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Show the frame
                cv2.imshow('Model Predictions Only', display_frame)
                
                frame_count += 1
                
                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed, exiting...")
                    break

        finally:
            print("Releasing camera...")
            video_capture.release()
            cv2.destroyAllWindows()
            print("Camera released")
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    show_camera() 