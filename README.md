# Object Detection Project

**Authors:** Michell Payano, Aristides Milios, Hugo Baudchon

## Quickstart

```
dts devel build -H [bot_name] -f --verbose
dts devel run -H [bot_name] -L object-detection -X
```

in a separate terminal (use a venv if you want but be careful about installing it in the project directory because it might be copied over to the robot accidentally and fill up its entire storage):

```
pip install ultralytics
pip install opencv-python
python client/yolo_client.py [tcp_ip_adress] --port 8765 --model [/path/to/model/weights.pt]
```

## Summary of Steps  
1. **Data Collection**: Gathering camera images data with both the real & virtual robots to train our object detection model.  
2. **Automatic Labelling**: Automatically labeling the data using a pretrained Vision LLM (VLLM).
3. **Model Training**: Building and training the object detection model.  
4. **Integration with the robot**: The model sends images to the laptop via TCP, and the laptop returns the detections back to the robot, which publishes them as a ROS topic.

---

## Data Collection

Steps 1-3 are for virtual robot setup only, while steps 4-5 are for running data collection on either a real or virtual robot.

### Virtual Robot Setup (optional if you only want to use the real robot)

1. *Start the simulation*:  

Run the following shell command to start the duckiematrix. There is a custom loop in ```./assets/duckiematrix/loop``` with additional duckies and static robots.
```shell
dts matrix run --standalone -m [/path/to/dts/map]
```

2. *Start a virtual bot*:

First, create a virtual bot if you don't have one already:
```shell
dts duckiebot virtual create -t duckiebot [BOT_NAME] -c DB21J
```

Then, start the virtual bot:
```shell
dts duckiebot virtual start [BOT_NAME]
```

This might take a few seconds, you can check when the virtual bot is ready with the command ```dts fleet discover```

3. *Attach the virtual bot to the duckiematrix*:

Once the virtual bot is ready, you can attach it to the previously started duckiematrix:
```shell
dts matrix attach [BOT_NAME] "map_0/vehicle_0"
```

### Data collection (real and virtual robot)
4. *Start keyboard controls*:

Run the following command to start the keyboard controls for the robot (real or virtual):
```shell
dts duckiebot keyboard_control [BOT_NAME]
```

5. *Collect data*:

We highly recommend setuping the automatic ssh key with the robot to avoid typing the password multiple times within the script, especially if you run the script multiple time under different environment conditions (lightning, objects placement...):

```shell
ssh-copy-id duckie@[BOT_NAME].local
```

Then, run the following command to start collecting data. You will be prompted to press "Enter" twice to 1) start recording and 2) stop recording. You should use the keyboard controls previously started to move the robot around while collecting.

--output_dir should be a path on your computer, as the script will automatically download the images from the robot to your computer.
```shell
python data_collection/rosbag_image_extractor.py \
 --robot_name [BOT_NAME] \
 --password quackquack \
 --output_dir [path/to/output/folder]
```


## Automatic Labelling  

The automated labelling is done through the [OWLv2 model](https://huggingface.co/docs/transformers/en/model_doc/owlv2), powered by the [Huggingface](https://huggingface.co/) library for inference. Specifically we use [this version](https://huggingface.co/google/owlv2-base-patch16-ensemble) of the model for our inference, as we found it struck a good balance between inference speed and annotation quality.

OWLv2 is an open-vocabulary (open-domain) detection model. That is to say, at inference time and with no training, it is able to accept a "class list" (effectively a short description of each class), and based on its understanding of these descriptions, produce bounding boxes around the corresponding objects in the image in zero-shot.

The labelling script is run as follows:

```
python automated_data_labelling/process_images.py
```

The key parameters of this script, located at the top of the file, are:

```
zip_file_images_folder = "duckietown_images" # directory name containing the images for labelling
DETECTION_THRESHOLD = 0.2 # the certainty threshold under which we discard bounding boxes
CONTAINMENT_THRESHOLD = 0.7 # see footnote about failure case with many similar objs
class_list = [
    "rc car",
    "yellow rubber duckie",
    "small orange traffic cone",
    "wooden toy house",
    "qr code",
    "road sign"
] 
```

The script produces a `labels` subdirectory in the original directory that contains the bounding boxes in YOLO format to be ingested by the subsequent training scripts.

The script contains a `visualize_detection` function that can be used to visually inspect the quality of the annotations for a given image in the folder. It is not currently being used by the script (the script used to be a notebook where the results were visualized interactively, but this version is meant to run on a cluster for faster inference).

### Footnote about `CONTAINMENT_THRESHOLD` parameter:

The `process_images.py` script also discards the failure case of OWLv2 where, when many small bounding boxes of the same class exist close together, OWLv2 creates another large bounding box that encapsulates them all together, along with the small bounding boxes of the individual objects. Some clever code in the script discards these larger bounding boxes. The `CONTAINMENT_THRESHOLD` parameter determines how much overlap there needs to be for the larger bounding box to be discarded. Specifically, it controls how much of the small box needs to be contained in the large box for discarding to occur. 

## Model Training

In this project, we used for object detection the models based on You Only Look Once [YOLO](https://docs.ultralytics.com/) version architectures. The YOLO architecture is renowned for its exceptional speed and accuracy in object detection. Unlike traditional methods that involve multiple stages for detecting and classifying objects, YOLO treats object detection as a single regression problem, directly predicting bounding boxes and class probabilities in one forward pass through the network. This design enables real-time performance, making YOLO particularly advantageous for applications requiring fast inference, such as autonomous vehicles, surveillance, and robotics.

For this case, we tried the pre-trained nano/tiny versions of the models versions 5, 8, 9, 10 and 11, where each was trained for 100 epochs. In the table below we can see some configuration each model and the result in the validation set. The MaP50 corresponds to the Mean Average Precision at an intersection over union (IoU) threshold of 0.5. It is a standard evaluation metric for object detection, summarizing precision and recall across all classes.


| Model      | Layers       | Parameters  | MaP50       |
|:-----------|:------------:|------------:|------------:|
| YOLOv5n    | 262          | 2,654,816   | 0.835       |
| YOLOv8n    | 225          | 3,157,200   | 0.848       |
| YOLOv9t    | 917          | 2,128,720   | 0.843       |
| YOLOv10n   | 385          | 2,775,520   | 0.836       |
| YOLOv11n   | 319          | 2,624,080   | 0.846       |

Since these models show almost a similar performance, for simplicty, we decided to use the YOLOv5n as it is widly used in the duckietown community. In the plot below we can see the loss associated with the bounding box detection and the loss associated with the class prediction for both the training and validation sets, where each one present a downward trend. Also we can see for the validation, the Map50 and Map50-95, which are the mean average precision at IoU 0.5 and the mean average precision averaged across multiple IoU (from 0.5 to 0.95). Both lines present an upward trend, showing that the model in improving its detection performance acrosss all classes, especially for tighter bounding box overlaps.


![results_training](https://github.com/hugobaudchon/object-detection-project/blob/v3/images/results_plot.png?raw=true)


## Integration with the robot

First, you need to build and run the object detection package on the robot:
```
dts devel build -H [bot_name] -f --verbose
dts devel run -H [bot_name] -L object-detection -X
```

This will start streaming camera images to the laptop through a TCP connection.
The laptop will run the object detection model and send the detections back to the robot, which will publish them as a ROS topic.

Once the object detection package is running on the robot, you can run the client on the laptop to receive the images and send back the detections.

IMPORTANT: check the output of ```dts devel run [...]``` for the IP and update it in the command below.

You will also have to point the script to the YOLO model weights .pt file.
```
python client/yolo_client.py [tcp_ip_adress] --port 8765 --model [/path/to/model/weights.pt]
```
After integrating all these components we can see that as result our duckiebot successfully stops whenever it sees a duckie (or any object specified to detect in the [object_detection_node.py](packages/object_detection/src/object_detection/object_detection_node.py) in the self.duckie_class_id):

![video](images/video.gif)

## Common Issues

1. Make sure that extraneous directories aren't being copied over to fill up the robot's storage (e.g. venvs)
