#!/usr/bin/env python3
import os
import time
import socket
import json
import cv2
import threading
import struct
import signal
import numpy as np
import asyncio
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from dtps import context, ContextConfig, DTPSContext
from dt_robot_utils import get_robot_name
from duckietown_messages.actuators.differential_pwm import DifferentialPWM

class ObjectDetectionNode:
    def __init__(self):
        self._shutdown = False
        self._robot_name = get_robot_name()
        self._vehicle_name = self._robot_name
        
        # ROS initialization
        rospy.init_node('object_detection_node', anonymous=True)
        self._bridge = CvBridge()
        
        # TCP server settings
        self.server_port = 8766
        self.client_socket = None
        self.client_lock = threading.Lock()

        # Frame rate control
        self.last_frame_time = 0
        self.min_frame_interval = 0.33  # 3 FPS max
        self.frames_sent = 0
        self.last_stats_time = time.time()

        # Detection settings
        self.latest_detections = []
        self.detections_lock = threading.Lock()
        self.duckie_class_id = 1  # Class ID for duckie
        self.min_duckie_area = 5000  # minimum area to trigger stop
        self.last_detection_time = 0
        self.detection_timeout = 1.0  # seconds

        # DTPS context
        self.pwm_publisher = None

        # Register sigint handler
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Start TCP server in a separate thread
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Start detection receiver thread
        self.receiver_thread = threading.Thread(target=self.receive_detections)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

        # Start stats thread
        self.stats_thread = threading.Thread(target=self._print_stats)
        self.stats_thread.daemon = True
        self.stats_thread.start()

        # Subscribe to camera feed
        self.sub = rospy.Subscriber(
            f'/{self._robot_name}/camera_node/image/compressed',
            CompressedImage,
            self.camera_callback,
            queue_size=1
        )

        self.print_network_info()
        print(f"Object detection node initialized for robot: {self._robot_name}")

    def _print_stats(self):
        """Print performance statistics periodically"""
        while not self._shutdown:
            time.sleep(5.0)  # Print every 5 seconds
            current_time = time.time()
            elapsed = current_time - self.last_stats_time
            fps = self.frames_sent / elapsed
            print(f"\nPerformance Stats:")
            print(f"Frames sent: {fps:.1f} fps\n")
            self.frames_sent = 0
            self.last_stats_time = current_time

    def get_ip_addresses(self):
        """Get all IP addresses for the robot"""
        ips = []
        try:
            interfaces = socket.getaddrinfo(host=socket.gethostname(), port=None, family=socket.AF_INET)
            all_ips = set(item[4][0] for item in interfaces)
            ips = [ip for ip in all_ips if not ip.startswith('127.')]

            if not ips:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(('10.255.255.255', 1))
                    ip = s.getsockname()[0]
                    if ip and not ip.startswith('127.'):
                        ips.append(ip)
                except Exception:
                    pass
                finally:
                    s.close()
        except Exception as e:
            print(f"Error getting IP addresses: {e}")

        return ips

    def print_network_info(self):
        """Print network information to help with connections"""
        ips = self.get_ip_addresses()
        
        print("\nNetwork Information:")
        print("=" * 50)
        print(f"Robot name: {self._vehicle_name}")
        print(f"Server port: {self.server_port}")
        print("IP Addresses:")
        for ip in ips:
            print(f"  - {ip}")
        print("=" * 50)

    def calculate_bbox_area(self, bbox):
        """Calculate area of bounding box"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def get_largest_duckie_area(self):
        """Get the area of the largest duckie detection"""
        with self.detections_lock:
            duckie_detections = [det for det in self.latest_detections 
                               if det['cls'] == self.duckie_class_id]
            
            if not duckie_detections:
                return 0
                
            areas = [self.calculate_bbox_area(det['bbox']) for det in duckie_detections]
            return max(areas) if areas else 0

    async def motor_control(self):
        """Main control loop for motor commands"""
        try:
            # Initialize connection to DTPS
            switchboard = (await context("switchboard")).navigate(self._robot_name)
            self.pwm_publisher = await (switchboard / "actuator" / "wheels" / "base" / "pwm").until_ready()
            
            # Initial state - stop
            stop_msg = DifferentialPWM(left=0.0, right=0.0)
            await self.pwm_publisher.publish(stop_msg.to_rawdata())
            print("Motor control initialized - Robot stopped")

            while not self._shutdown:
                try:
                    current_time = time.time()
                    time_since_last_detection = current_time - self.last_detection_time
                    
                    if time_since_last_detection > self.detection_timeout:
                        # Haven't received detections recently, stay stopped
                        if self.pwm_publisher:
                            stop_msg = DifferentialPWM(left=0.0, right=0.0)
                            await self.pwm_publisher.publish(stop_msg.to_rawdata())
                            print("No recent detections - Stopping")
                    else:
                        # Process latest detection
                        largest_duckie_area = self.get_largest_duckie_area()
                        
                        if largest_duckie_area > self.min_duckie_area:
                            # Duckie detected and close - stop
                            pwm = DifferentialPWM(left=0.0, right=0.0)
                            print(f"Duckie detected! Area: {largest_duckie_area:.1f} - Stopping")
                        else:
                            # No duckie or too far - move forward
                            rads_left = 0.1
                            rads_right = 0.1
                            pwm_left = 0.4 * rads_left
                            pwm_right = 0.4 * rads_right
                            pwm = DifferentialPWM(left=pwm_left, right=pwm_right)
                            print(f"Moving forward, largest duckie area: {largest_duckie_area:.1f}")
                        
                        if self.pwm_publisher:
                            await self.pwm_publisher.publish(pwm.to_rawdata())
                    
                except Exception as e:
                    print(f"Error in motor control loop: {e}")
                    
                await asyncio.sleep(0.1)  # 10Hz control rate
                
        except Exception as e:
            print(f"Error in motor control setup: {e}")
            import traceback
            traceback.print_exc()

    def camera_callback(self, msg):
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < self.min_frame_interval:
                return
            
            self.last_frame_time = current_time

            # Decode the JPEG data
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Resize to smaller resolution (e.g., 640x480 or even 320x240)
            # img = cv2.resize(img, (640, 480))
            
            # Re-encode with lower quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, compressed_jpeg = cv2.imencode('.jpg', img, encode_params)
            
            self.send_frame(compressed_jpeg.tobytes())
            self.frames_sent += 1
                
        except Exception as e:
            print(f"Error in camera callback: {e}")
    
    def run_server(self):
        """Run TCP server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.server_port))
        server_socket.listen(1)

        print(f"Starting TCP server on port {self.server_port}")

        while not self._shutdown:
            try:
                client, addr = server_socket.accept()
                print(f"Client connected from {addr}")
                # Set TCP_NODELAY for low latency
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.client_lock:
                    if self.client_socket is not None:
                        self.client_socket.close()
                    self.client_socket = client
            except Exception as e:
                print(f"Server error: {e}")
                time.sleep(1)

    def receive_detections(self):
        """Receive and process detections from client"""
        while not self._shutdown:
            with self.client_lock:
                if self.client_socket is None:
                    time.sleep(0.1)
                    continue

                try:
                    # Make socket non-blocking for initial read
                    self.client_socket.setblocking(False)
                    try:
                        length_data = self.client_socket.recv(4)
                    except socket.error:
                        time.sleep(0.01)
                        continue
                    finally:
                        self.client_socket.setblocking(True)

                    if not length_data:
                        print("Received empty length data, client may have disconnected")
                        self.client_socket = None
                        continue

                    message_length = struct.unpack('!I', length_data)[0]

                    # Receive message
                    message = b''
                    remaining = message_length
                    while remaining > 0:
                        chunk = self.client_socket.recv(remaining)
                        if not chunk:
                            raise ConnectionError("Connection closed while reading message")
                        message += chunk
                        remaining -= len(chunk)

                    # Update detections and timestamp
                    detections = json.loads(message.decode())
                    with self.detections_lock:
                        self.latest_detections = detections
                    self.last_detection_time = time.time()
                    
                    print(f"Received {len(detections)} detections:")
                    for det in detections:
                        bbox = det['bbox']
                        area = self.calculate_bbox_area(bbox)
                        print(f"  Class {det['cls']}: conf={det['conf']:.2f}, bbox={bbox}, area={area:.1f}")

                except socket.error as e:
                    if e.errno == socket.EAGAIN or e.errno == socket.EWOULDBLOCK:
                        continue
                    else:
                        print(f"Socket error: {e}")
                        self.client_socket = None
                except Exception as e:
                    print(f"Error receiving detections: {e}")
                    import traceback
                    traceback.print_exc()
                    self.client_socket = None

    def send_frame(self, jpeg_data):
        """Send frame to client"""
        with self.client_lock:
            if self.client_socket is None:
                return

            try:
                # Send the length of the JPEG data followed by the data itself
                message_length = struct.pack('!I', len(jpeg_data))
                self.client_socket.sendall(message_length)
                print(f"Size of frame sent: {len(jpeg_data)} bytes")
                self.client_socket.sendall(jpeg_data)

            except Exception as e:
                print(f"Error sending frame: {e}")
                self.client_socket = None

    async def worker(self):
        """Main worker function"""
        try:
            # Start motor control
            motor_task = asyncio.create_task(self.motor_control())
            
            # Wait until shutdown
            await self.join()
            
        except Exception as e:
            print(f"Error in worker: {e}")
            import traceback
            traceback.print_exc()

    async def join(self):
        """Wait until shutdown"""
        while not self._shutdown:
            await asyncio.sleep(1)

    def _sigint_handler(self, _, __):
        """Handle SIGINT"""
        self._shutdown = True

    @property
    def is_shutdown(self):
        return self._shutdown

    def spin(self):
        """Start the node"""
        try:
            asyncio.run(self.worker())
        except RuntimeError as e:
            if not self.is_shutdown:
                print(f"An error occurred while running the event loop: {e}")
                raise

if __name__ == "__main__":
    node = ObjectDetectionNode()
    node.spin()