#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract images from a rosbag."""

import os
import argparse
import rosbag
from sensor_msgs.msg import CompressedImage, Image
import numpy as np

def bag_to_images(bag_file, output_dir, image_topic):
    """Extract a folder of images from a rosbag."""
    print(f"bag_file: {bag_file}")
    print(f"output_dir: {output_dir}")
    print(f"image_topic: {image_topic}")

    print("Extract images from %s on topic %s into %s" % (bag_file, image_topic, output_dir))

    bag = rosbag.Bag(bag_file, "r")
    count = 0

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        print(type(msg))

        # Check if the message is CompressedImage
        # if isinstance(msg, CompressedImage):

        # Determine the file extension based on the compression format
        ext = "jpg" if "jpeg" in msg.format.lower() else "png"
        output_path = os.path.join(output_dir, f"frame{count:06d}.{ext}")
        with open(output_path, "wb") as f:
            f.write(msg.data)
        print(f"Wrote compressed image {count} to {output_path}")

        # # Handle uncompressed Image messages
        # elif isinstance(msg, Image):
        #     cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #     output_path = os.path.join(output_dir, f"frame{count:06d}.png")
        #     with open(output_path, "wb") as f:
        #         f.write(cv_img.tobytes())
        #     print(f"Wrote uncompressed image {count} to {output_path}")
        #
        # else:
        #     print(f"Skipped non-image message on topic {topic}")

        count += 1

    bag.close()
    print(f"Finished extracting {count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    bag_to_images(args.bag_file, args.output_dir, args.image_topic)
