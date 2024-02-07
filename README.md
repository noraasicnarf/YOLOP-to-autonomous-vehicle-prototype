
# YOLOP-to-autonomous-vehicle-prototype
An implementation of the YOLOP model into an autonomous vehicle prototype with lane-keeping and collision avoidance. Some file are cloned from [hustvl/yolop](https://github.com/noraasicnarf/YOLOP-to-autonomous-vehicle-prototype)
### Custom Trained Yolop
The network was trained on Nvidia Geforce RTX 3050. YOLOP inference was lowered down from 640 x 640 to 320 x 320 and added 13 class for Object Detection
### Prototype System Flow
![yolop](pictures/Prototype.jpg)
To implement the described system, you'll need to have the Raspberry Pi Cam v2 (for capturing images), on the Host PC (for processing YOLOP inference), on the Jetson Nano (for communication with Raspberry Pi Camera and Host PC, and for sending commands to the Arduino), and on the Arduino (for controlling the DC motor based on received commands). Below, I'll provide a high-level overview of the steps involved and then outline the code for each component separately.

# Raspberry Pi Cam v2:

Capture images using the Raspberry Pi Camera V2.
Stream the images over Wi-Fi to the Host PC.
# Host PC:

Receive the streamed images from the Raspberry Pi.
Process the images using the YOLOP model for lane keeping.
Send the resulting commands (characters) back to the Jetson Nano.
# Jetson Nano:

Receive the commands from the Host PC.
Forward the commands to the Arduino via USB serial connection.
# Arduino:

Receive commands from the Jetson Nano via USB serial connection.
Interpret the commands and control the DC motor accordingly.
### Prototype Sample Run
![sample run](pictures/sample.gif)
