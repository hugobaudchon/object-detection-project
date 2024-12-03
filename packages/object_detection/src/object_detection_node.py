#!/usr/bin/env python3

import os
import random
import time

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

# from object_detection.include.tensorrt_model import YoLov5TRT

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        print(f"Vehicle name: {self._vehicle_name}")
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        print("Creating CvBridge")
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        print(f"Creating window '{self._window}'")
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        print(f"Subscribing to {self._camera_topic}")
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        self.last_inference_time = time.time()  # Initialize the last inference time
        self.inference_delay = 0.5

    def detect_objects(self, image):
        # model = YoLov5TRT()
        # boxes = model.infer(image)

        height, width, _ = image.shape

        n_random_boxes = 3
        labels = ['duckie', 'robot', 'cone']
        labels_colors = {'duckie': (0, 255, 0), 'robot': (0, 0, 255), 'cone': (255, 0, 0)}

        # 3 random boxes with labels and corresponding colors
        boxes = []
        for i in range(n_random_boxes):
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + 20, min(x1 + 100, width))
            y2 = random.randint(y1 + 20, min(y1 + 100, height))
            label = labels[random.randint(0, len(labels) - 1)]
            boxes.append((x1, y1, x2, y2, label))

        for box in boxes:
            x1, y1, x2, y2, label = box
            color = labels_colors[label]  # Get the color for the label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Class {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

    def callback(self, msg):
        current_time = time.time()
        # Check if enough time has passed since the last inference
        if current_time - self.last_inference_time >= self.inference_delay:
            self.last_inference_time = current_time  # Update the last inference time
            image = self._bridge.compressed_imgmsg_to_cv2(msg)
            print(image.shape)
            image_with_boxes = self.detect_objects(image)
            cv2.imshow(self._window, image_with_boxes)
            cv2.waitKey(1)
        else:
            print("Skipping frame to simulate inference delay")

if __name__ == '__main__':
    # create the node
    node = ObjectDetectionNode(node_name='object_detection_node')
    # keep spinning
    rospy.spin()