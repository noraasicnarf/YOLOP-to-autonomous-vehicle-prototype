import cv2

port = 5555

def gpipe():
    # Create the GStreamer pipeline string
    pipeline = f"udpsrc port={port} ! application/x-rtp, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

    # Create a VideoCapture object with the GStreamer pipeline
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    return cap