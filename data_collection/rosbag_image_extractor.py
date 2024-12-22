#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime

import paramiko
import os
import argparse


def ssh_and_run_command(robot_name, command, password):
    """SSH into a host and run a command."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        host = f"{robot_name}.local"
        client.connect(hostname=host, username='duckie', password=password)
        print(f"SSH connection established with {host}.")

        stdin, stdout, stderr = client.exec_command(command)

        # Continuously read output to prevent blocking
        for line in iter(stdout.readline, ""):
            print(line, end="")  # Print each line of the output as it comes

        for line in iter(stderr.readline, ""):
            print(f"Error: {line}", end="")
    finally:
        client.close()
        print(f"SSH connection closed with {host}.")


def transfer_file_to_robot(local_path, robot_name, remote_path):
    """Copy a file from the local machine to the robot."""
    print(f"Copying {local_path} to {robot_name}:{remote_path}...")
    os.system(f"scp {local_path} duckie@{robot_name}.local:{remote_path}")
    print("File transfer complete.")


def transfer_images_to_local(robot_name, password, docker_container, docker_output_dir, robot_output_dir, local_output_dir):
    """Transfer the extracted images from the Docker container to the local machine."""
    print(f"Copying images from Docker container '{docker_container}' to robot's file system...")

    # Copy images from Docker to the robot's file system
    docker_copy_command = (
        f"docker cp {docker_container}:{docker_output_dir} {robot_output_dir} && "
        f"docker exec {docker_container} rm -rf {docker_output_dir}"
    )
    ssh_and_run_command(robot_name, docker_copy_command, password)

    print(f"Transferring extracted images from {robot_name}:{robot_output_dir} to {local_output_dir}...")

    # Ensure local output directory exists
    os.makedirs(local_output_dir, exist_ok=True)

    # Transfer images from the robot to the local machine
    remote_path = f"duckie@{robot_name}.local:{robot_output_dir}/"
    command = f"rsync -avz {remote_path} {local_output_dir}"
    result = os.system(command)

    if result != 0:
        print("rsync failed, falling back to scp.")
        remote_files = f"{remote_path}*"
        result = os.system(f"scp -r {remote_files} {local_output_dir}")

    if result == 0:
        print("Transfer complete.")

        # Remove images from the robot after successful transfer
        cleanup_command = f"ssh duckie@{robot_name}.local 'rm -r {robot_output_dir}'"
        cleanup_result = os.system(cleanup_command)

        if cleanup_result == 0:
            print("Remote images removed successfully.")
        else:
            print("Error: Failed to remove images from the robot.")
    else:
        print("Error: Failed to transfer images.")


def collect_data(args):
    docker_container_name = 'ros1'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Remote paths on the robot
    remote_bag_path = f"/home/duckie/camera_video.bag"
    remote_image_output_dir = f"/home/duckie/images_{timestamp}"
    remote_script_path = "/home/duckie/bag_to_images.py"

    # ROS image topic
    image_topic = f"/{args.robot_name}/camera_node/image/compressed"
    throttled_image_topic = f"/{args.robot_name}/camera_node/image/compressed_throttled"

    # Command to start a throttled image topic
    start_throttled_image_topic = (
        f"docker exec -i {docker_container_name} bash -c "
        "'source /opt/ros/noetic/setup.bash && "
        f"rosrun topic_tools throttle messages {image_topic} 1 {throttled_image_topic} &'"
    )

    # Command to start rosbag recording on the robot
    start_rosbag_command = (
        f"docker exec -i {docker_container_name} bash -c "
        "'source /opt/ros/noetic/setup.bash && "
        f"rm -f {remote_bag_path} && "
        f"rosbag record --bz2 {throttled_image_topic} -O {remote_bag_path} &'"
    )

    stop_rosbag_command = "docker exec -i ros1 bash -c 'pkill -f \"rosbag record\"'"

    # Command to extract images on the robot
    extract_images_command = (
        f"docker exec -i {docker_container_name} bash -c "
        f"'source /opt/ros/noetic/setup.bash && "
        f"ls /home/duckie && mkdir -p {remote_image_output_dir} && "
        f"python3 {remote_script_path} {remote_bag_path} {remote_image_output_dir} {throttled_image_topic}'"
    )

    copy_script_command = f"docker cp {remote_script_path} {docker_container_name}:/home/duckie/bag_to_images.py"

    # Step 1: Copy the `bag_to_images.py` script to the robot
    print("Copying the image extraction script to the robot...")
    bag_to_image_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bag_to_images.py")
    transfer_file_to_robot(bag_to_image_script, args.robot_name, remote_script_path)
    ssh_and_run_command(args.robot_name, copy_script_command, args.password)

    # Step 2: SSH into the robot and start rosbag recording
    input("Press Enter to start recording images:")

    print("Starting rosbag recording on the virtual robot...")
    ssh_and_run_command(args.robot_name, start_throttled_image_topic, args.password)
    ssh_and_run_command(args.robot_name, start_rosbag_command, args.password)
    print(f"Recording images from topic '{throttled_image_topic}'...")
    print("You should now move the robot around to capture different images!")

    # Step 3: Wait for rosbag to record data
    # Wait for user input to stop recording
    input("Press Enter to stop recording and download images:")

    print("Stopping rosbag recording...")
    ssh_and_run_command(args.robot_name, stop_rosbag_command, args.password)

    # Step 4: Extract images from the rosbag directly on the robot
    print("Running image extraction on the virtual robot...")
    ssh_and_run_command(args.robot_name, extract_images_command, args.password)

    # Step 5: Transfer the images back to the local machine
    transfer_images_to_local(
        args.robot_name,
        args.password,
        docker_container_name,
        remote_image_output_dir,
        remote_image_output_dir,
        f"{args.output_dir}/images_{timestamp}"
    )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="SSH into a virtual robot, run image extraction, and transfer images.")
    parser.add_argument("--robot_name", help="Robot's hostname.")
    parser.add_argument("--password", help="Robot's password.")
    parser.add_argument("--output_dir", help="Local directory to save extracted images.")

    args = parser.parse_args()

    collect_data(args)
