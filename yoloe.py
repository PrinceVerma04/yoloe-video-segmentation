#prompt free segmentation mode

from ultralytics import YOLOE

# model=YOLOE("yoloe-26l-seg-pf.pt")
# results=model.predict("/home/coral/Desktop/yolo working/YOLOE/video folder/16859876_1920_1080_30fps.mp4",
#                       show=True,
#                       save=True,
#                       show_conf=False,conf=0.25)

# with prompt model
# from ultralytics import YOLOE
# # Load text-prompt segmentation model
# model = YOLOE("yoloe-26x-seg.pt")
# # Define target classes
# names = ["Man", "Woman"]
# # Generate text embeddings and set classes
# text_embeddings = model.get_text_pe(names)
# model.set_classes(names, text_embeddings)
# # Input video
# video_path = "/home/coral/Desktop/yolo working/YOLOE/video folder/16859876_1920_1080_30fps.mp4"
# # Run prediction
# results = model.predict(
#     source=video_path,
#     conf=0.25,
#     show=True,
#     save=True,
#     show_conf=False
# )

#with visual prompt model
import numpy as np
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
# Load YOLOE segmentation model
model = YOLOE("yoloe-26x-seg.pt")
# Reference image
reference_image = "/home/coral/Desktop/yolo working/YOLOE/referance/deer.jpg"
# Target video
video_path = "/home/coral/Desktop/yolo working/YOLOE/video folder/12491353_2160_3840_30fps.mp4"
# Select objects using bounding boxes
visual_prompts = {
    "bboxes": np.array([[115, 45, 420, 365]]),  # whole deer in deer.jpg (x1,y1,x2,y2)
    "cls": np.array([0]),
}
# Your class names (one per "cls" id in visual_prompts)
class_names = ["deer"]
# Step 1: learn the class from the box in the reference image
model.predict(
    source=reference_image,
    refer_image=reference_image,
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    verbose=False,
)
# Step 2: replace the default "object0" label with your name
model.model.names = class_names
model.predictor = None  # use a normal predictor for the video
# Step 3: run on the video
results = model.predict(
    source=video_path,
    show=True,
    save=True,
    conf=0.1,
)