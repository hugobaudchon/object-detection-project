import mss
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# Custom display names for YOLO labels
DISPLAY_NAMES = {
    'rc car': 'duckiebot',
    'yellow rubber duckie': 'duckie',
    'small orange traffic cone': 'cone',
    'wooden toy house': 'house',
    'qr code': 'qr code',
    'road sign': 'road sign'
}

# Colors for each class (in BGR format)
CLASS_COLORS = {
    'duckiebot': (255, 255, 0),      # Cyan (BGR)
    'duckie': (0, 165, 255),         # Orange (BGR)
    'cone': (0, 0, 255),             # Red (BGR)
    'house': (255, 255, 255),        # White (BGR)
    'qr code': (255, 255, 255),      # White (BGR)
    'road sign': (0, 0, 0)           # Black (BGR)
}

def main():
    # Initialize YOLO
    model = YOLO('/home/hugo/PycharmProjects/object-detection-project/packages/object_detection/include/weights/best.pt')
    if torch.cuda.is_available():
        device = f"cuda:0"
        torch.cuda.set_device(device)
        print(f"Using GPU device: {device}")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU.")
    model.to(device)
    
    # Initialize screen capture
    sct = mss.mss()
    
    # Initialize display labels flag
    show_labels = True
    print("Press 'l' to toggle labels on/off")
    
    # Get list of monitors
    print("\nAvailable monitors:")
    for i, monitor in enumerate(sct.monitors[1:], 1):
        print(f"Monitor {i}: {monitor}")
    
    print("\nPosition your camera feed window where you want it.")
    print("Click and drag to select the region, then press ENTER.")
    
    frame = np.array(sct.grab(sct.monitors[1]))
    height, width = frame.shape[:2]
    
    scale = 0.5
    small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    
    roi = cv2.selectROI("Select Camera Feed Region", small_frame)
    cv2.destroyWindow("Select Camera Feed Region")
    
    region = {
        "top": int(roi[1] / scale),
        "left": int(roi[0] / scale),
        "width": int(roi[2] / scale),
        "height": int(roi[3] / scale)
    }
    
    print(f"\nCapturing region: {region}")
    
    try:
        while True:
            start_time = time.time()
            
            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            results = model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    conf = float(box.conf)
                    cls = int(box.cls)
                    yolo_label = result.names[cls]
                    
                    display_name = DISPLAY_NAMES.get(yolo_label, yolo_label)
                    color = CLASS_COLORS.get(display_name, (0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label if enabled
                    if show_labels:
                        label = f"{display_name} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                        # Use black text for light colors, white text for dark colors
                        text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)  # 382 is threshold (255*1.5)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Detections', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels {'enabled' if show_labels else 'disabled'}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
