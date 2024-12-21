# Object Detection Project

**Authors:** Michell Payano, Aristides Milios, Hugo Baudchon

## Summary of Steps  
1. **Data Collection**: Gathering data with both the real & virtual robots to train our object detection model.  
2. **Automatic Labeling**: Automatically labeling the data using a pretrained Language-Vision model.
3. **Model Training**: Building and training the object detection model.  
4. **Integration with the robot**: The model sends images to the laptop via TCP, and the laptop returns the detections back to the robot, which publishes them as a ROS topic.

---

### Data Collection

Steps 1-3 are for virtual robot setup, while steps 4-6 are for running data collection on either a real or virtual robot.

#### Virtual Robot Setup (optional if you only want to use the real robot)

1. *Start the simulation*:  

Run the following shell command to start the duckiematrix. There is a custom loop in ```./assets/duckiematrix/loop``` with additional duckies and static robots.
```shell
dts matrix run --standalone -m [/path/to/dts/map]
```

2. *Start a virtual bot*:

First, create a virtual bot if you don't have one already:
```shell
dts duckiebot virtual create -t duckiebot [VBOT_NAME] -c DB21J
```

Then, start the virtual bot:
```shell
dts duckiebot virtual start [VBOT_NAME]
```

This might take a few seconds, you can check when the virtual bot is ready with the command ```dts fleet discover```

3. *Attach the virtual bot to the duckiematrix*:

Once the virtual bot is ready, you can attach it to the previously started duckiematrix:
```shell
dts matrix attach [BOT_NAME] "map_0/vehicle_0"
```

#### Data collection (real and virtual robot)
4. *Start keyboard controls*:

Run the following command to start the keyboard controls for the robot (real or virtual):
```shell
dts duckiebot keyboard_control [BOT_NAME]
```

5. 




### Model Training  

#### TODO

### Model Training

#### TODO

### Integration with the robot

for the robot script:
```
dts devel build -H didibot -f --verbose
dts devel run -H didibot -L object-detection -X
```

for the laptop script (running the detection model):
IMPORTANT: check the output of dts devel run for the IP and update it in the command below:
```
python yolo_client.py [tcp_ip_adress] --port 8765 --model [/path/to/model/weights.pt]
```