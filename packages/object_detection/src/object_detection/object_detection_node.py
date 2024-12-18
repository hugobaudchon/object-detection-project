#!/usr/bin/env python3
import os
import time
import rospy
import socket
import json
import cv2
import base64
import threading
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
# from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from object_detection.msg import BoundingBox, DetectionArray
import struct


class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # TCP server settings
        self.server_port = 8765
        self.client_socket = None
        self.client_lock = threading.Lock()

        # Processing settings
        self.last_inference_time = time.time()
        self.inference_delay = 0.01

        # # Publisher for detections
        self.detection_pub = rospy.Publisher(
            f'/{self._vehicle_name}/object_detection_node/detections',
            BoundingBox,
            queue_size=1
        )

        # Start TCP server in a separate thread
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # # # Start detection receiver thread
        # self.receiver_thread = threading.Thread(target=self.receive_detections)
        # self.receiver_thread.daemon = True
        # self.receiver_thread.start()

        # Subscribe to camera feed
        self.sub = rospy.Subscriber(
            self._camera_topic,
            CompressedImage,
            self.callback
        )

        self.print_network_info()

        rospy.loginfo(f"[{self._vehicle_name}] Object detection node initialized")

    def run_server(self):
        """Run TCP server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.server_port))
        server_socket.listen(1)

        rospy.loginfo(f"Starting TCP server on port {self.server_port}")

        while not rospy.is_shutdown():
            try:
                client, addr = server_socket.accept()
                rospy.loginfo(f"Client connected from {addr}")
                with self.client_lock:
                    if self.client_socket is not None:
                        self.client_socket.close()
                    self.client_socket = client
            except Exception as e:
                rospy.logerr(f"Server error: {e}")
                rospy.sleep(1)

    def get_ip_addresses(self):
        """Get all IP addresses for the robot"""
        ips = []
        try:
            # Get all network interfaces
            interfaces = socket.getaddrinfo(host=socket.gethostname(), port=None, family=socket.AF_INET)
            # Extract unique IPs
            all_ips = set(item[4][0] for item in interfaces)
            # Filter out localhost
            ips = [ip for ip in all_ips if not ip.startswith('127.')]

            # If no non-localhost IPs found, try alternative method
            if not ips:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Doesn't need to be reachable
                    s.connect(('10.255.255.255', 1))
                    ip = s.getsockname()[0]
                    if ip and not ip.startswith('127.'):
                        ips.append(ip)
                except Exception:
                    pass
                finally:
                    s.close()
        except Exception as e:
            rospy.logerr(f"Error getting IP addresses: {e}")

        return ips

    def print_network_info(self):
        """Print network information to help with connections"""
        ips = self.get_ip_addresses()

        rospy.loginfo("=" * 50)
        rospy.loginfo(f"Robot name: {self._vehicle_name}")
        rospy.loginfo(f"Server port: {self.server_port}")
        rospy.loginfo("IP Addresses:")
        for ip in ips:
            rospy.loginfo(f"  - {ip}")
        rospy.loginfo("=" * 50)

        # Also print to stdout for clarity
        print("\nNetwork Information:")
        print("=" * 50)
        print(f"Robot name: {self._vehicle_name}")
        print(f"Server port: {self.server_port}")
        print("IP Addresses:")
        for ip in ips:
            print(f"  - {ip}")
        print("=" * 50)

    # def receive_detections(self):
    #     """Receive and process detections from client"""
    #     buffer = ""
    #     while not rospy.is_shutdown():
    #         with self.client_lock:
    #             if self.client_socket is None:
    #                 rospy.sleep(0.1)
    #                 continue
    #
    #             try:
    #                 # First receive message length (4 bytes)
    #                 length_data = self.client_socket.recv(4)
    #                 if not length_data:
    #                     continue
    #
    #                 message_length = struct.unpack('!I', length_data)[0]
    #
    #                 # Then receive the actual message
    #                 message = self.client_socket.recv(message_length).decode()
    #
    #                 # Process detections
    #                 detections = json.loads(message)
    #                 self.publish_detections(detections)
    #
    #             except Exception as e:
    #                 rospy.logerr(f"Error receiving detections: {e}")
    #                 self.client_socket = None

    # def publish_detections(self, detections):
    #     """Publish detections to ROS topic"""
    #     detection_array = Detection2DArray()
    #     detection_array.header.stamp = rospy.Time.now()
    #     detection_array.header.frame_id = f"{self._vehicle_name}/camera_optical_frame"
    #
    #     for det in detections:
    #         detection = Detection2D()
    #
    #         # Set bounding box
    #         bbox = det['bbox']
    #         detection.bbox = BoundingBox2D()
    #         detection.bbox.center.x = (bbox[0] + bbox[2]) / 2
    #         detection.bbox.center.y = (bbox[1] + bbox[3]) / 2
    #         detection.bbox.size_x = bbox[2] - bbox[0]
    #         detection.bbox.size_y = bbox[3] - bbox[1]
    #
    #         # Set class and confidence
    #         hypothesis = ObjectHypothesisWithPose()
    #         hypothesis.id = det['cls']
    #         hypothesis.score = det['conf']
    #         detection.results.append(hypothesis)
    #
    #         detection_array.detections.append(detection)
    #
    #     self.detection_pub.publish(detection_array)

    def send_frame(self, encoded_frame):
        """Send frame to client, no response expected"""
        with self.client_lock:
            if self.client_socket is None:
                return

            try:
                # Make socket non-blocking for send
                self.client_socket.setblocking(0)

                # Prepare and send message
                message = encoded_frame.encode()
                message_length = struct.pack('!I', len(message))

                # Try to send with short timeout
                import select
                ready = select.select([], [self.client_socket], [], 0.1)[1]
                if ready:
                    # Send length first
                    self.client_socket.send(message_length)
                    # Then send data
                    self.client_socket.send(message)
                    rospy.logdebug(f"Sent frame of size {len(message)} bytes")
                else:
                    rospy.logwarn("Socket not ready for writing, skipping frame")

            except Exception as e:
                rospy.logerr(f"Error sending frame: {e}")
                self.client_socket = None

    def callback(self, msg):
        """Handle incoming camera frames"""
        current_time = time.time()

        if current_time - self.last_inference_time >= self.inference_delay:
            self.last_inference_time = current_time

            try:
                image = self._bridge.compressed_imgmsg_to_cv2(msg)
                image = cv2.resize(image, (416, 416))
                _, buffer = cv2.imencode('.jpg', image)
                encoded_frame = base64.b64encode(buffer).decode()
                self.send_frame(encoded_frame)

            except Exception as e:
                rospy.logerr(f"Error processing frame: {e}")
        else:
            rospy.logdebug("Skipping frame due to inference delay")

    def on_shutdown(self):
        """Cleanup when node is shutdown"""
        with self.client_lock:
            if self.client_socket:
                self.client_socket.close()
        rospy.loginfo("Object detection node is shutting down")


if __name__ == '__main__':
    node = ObjectDetectionNode(node_name='object_detection_node')
    rospy.spin()