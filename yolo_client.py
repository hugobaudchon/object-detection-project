#!/usr/bin/env python3
import socket
import cv2
import numpy as np
import base64
import json
import struct
import threading
import queue
import time
from ultralytics import YOLO


class YOLOClient:
    def __init__(self, robot_ip, port=8765, model_path='yolov8n.pt'):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
        self.connected = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.running = True

        # Initialize YOLO
        print(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)

        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._receive_frames)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def connect(self):
        """Connect to robot"""
        try:
            if self.socket:
                self.socket.close()

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.robot_ip, self.port))
            self.connected = True
            print(f"Connected to {self.robot_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False

    def _receive_frames(self):
        """Receive frames from robot"""
        while self.running:
            if not self.connected:
                if not self.connect():
                    time.sleep(1)
                    continue

            try:
                # Read message length (4 bytes)
                length_data = self.socket.recv(4)
                if not length_data:
                    raise ConnectionError("Connection closed by robot")

                message_length = struct.unpack('!I', length_data)[0]
                print(f"Expecting message of length: {message_length}")

                # Read the actual message
                message = b''
                while len(message) < message_length:
                    chunk = self.socket.recv(message_length - len(message))
                    if not chunk:
                        raise ConnectionError("Connection closed while reading message")
                    message += chunk

                # Decode frame
                encoded_frame = message.decode()
                frame_data = base64.b64decode(encoded_frame)
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    print("Got new frame")
                    # Update frame queue
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame)

            except Exception as e:
                print(f"Error receiving frame: {e}")
                self.connected = False
                time.sleep(1)

    def send_detections(self, detections):
        """Send detection results back to robot"""
        if not self.connected:
            return False

        try:
            # Convert detections to JSON and encode
            message = json.dumps(detections).encode()

            # Send message length first, then the message
            message_length = struct.pack('!I', len(message))
            self.socket.sendall(message_length + message)
            return True

        except Exception as e:
            print(f"Error sending detections: {e}")
            self.connected = False
            return False

    def process_frames(self):
        """Main processing loop"""
        try:
            while self.running:
                try:
                    print("Waiting for frame from queue...")
                    # Get the latest frame
                    frame = self.frame_queue.get(timeout=1.0)
                    print(f"Got frame from queue, shape: {frame.shape}")

                    # Create a copy of frame for visualization
                    display_frame = frame.copy()

                    # Run YOLO inference
                    print("Running YOLO inference...")
                    results = self.model(frame)
                    print(f"Got inference results")

                    # Process results
                    detections = []
                    for r in results:
                        boxes = r.boxes
                        print(f"Found {len(boxes)} boxes")
                        for box in boxes:
                            detection = {
                                'bbox': box.xyxy[0].tolist(),
                                'conf': float(box.conf),
                                'cls': int(box.cls)
                            }
                            detections.append(detection)

                    # Send detections back to robot
                    self.send_detections(detections)

                    # Visualize results on the copy
                    print("Drawing detections...")
                    for det in detections:
                        bbox = det['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            display_frame,
                            f"Class {det['cls']}: {det['conf']:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                    # Show frame
                    print("Displaying frame...")
                    cv2.imshow('YOLO Detections', display_frame)
                    cv2.waitKey(1)
                    print("Frame displayed, loop complete")

                except queue.Empty:
                    print("Queue empty, continuing...")
                    continue
                except Exception as e:
                    print(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.socket:
            self.socket.close()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO client for Duckietown')
    parser.add_argument('robot_ip', help='IP address of the robot')
    parser.add_argument('--port', type=int, default=8765, help='Port number')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model')
    args = parser.parse_args()

    client = YOLOClient(args.robot_ip, args.port, args.model)
    try:
        client.process_frames()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.cleanup()


if __name__ == '__main__':
    main()