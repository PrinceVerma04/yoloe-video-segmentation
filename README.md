# YOLOE Open-Vocabulary Video Segmentation
<img width="1932" height="504" alt="snapshot_00338" src="https://github.com/user-attachments/assets/463b5aa1-5b79-425b-85c9-b8093cf17a85" />



https://github.com/user-attachments/assets/31eab4e5-e468-49ee-a9dc-1017e80daf8c







Instance segmentation on video with [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/), using three different ways of telling the model what to look for:

| Mode | Model | How the target class is defined |
|---|---|---|
| **Prompt-free** | `yoloe-26l-seg-pf.pt` | Built-in vocabulary, segments everything it recognises |
| **Text prompt** | `yoloe-26x-seg.pt` + `mobileclip2_b.ts` | Class names as text, e.g. `["Man", "Woman"]` |
| **Visual prompt** (default) | `yoloe-26x-seg.pt` | A bounding box drawn on a reference image |

The active example finds a **deer** in a video by using a box around the deer in `referance/deer.jpg`, and labels the detections with a custom class name.

---
[YOLOE LinkedIn carousel.pdf](https://github.com/user-attachments/files/35275892/YOLOE.LinkedIn.carousel.pdf)

## Project structure


```
YOLOE/
├── yoloe.py               # Main script (all three modes; two are commented out)
├── referance/
│   └── deer.jpg           # Reference image for visual prompting (612×407)
├── requirements.txt
├── README.md
│
│   # Not committed (see .gitignore):
├── video folder/          # Input videos
├── runs/segment/predict*/ # Saved output videos
├── yoloe-26x-seg.pt       # Promptable model (text / visual prompts)
├── yoloe-26l-seg-pf.pt    # Prompt-free model
└── mobileclip2_b.ts       # MobileCLIP2-B text encoder (text-prompt mode only)
```

---

## Setup

Tested with Python 3.14, PyTorch 2.15 and Ultralytics 8.4.216 on an NVIDIA RTX 5080.

```bash
git clone https://github.com/PrinceVerma04/yoloe-video-segmentation.git
cd yoloe-video-segmentation

python3 -m venv yolo
source yolo/bin/activate
pip install -r requirements.txt
```

**Model weights** don't need to be downloaded by hand. Ultralytics fetches `yoloe-26x-seg.pt`, `yoloe-26l-seg-pf.pt` and `mobileclip2_b.ts` on first use.

**Videos** aren't included in the repository. Put your own `.mp4` files in a `video folder/` directory, or change `video_path` in `yoloe.py`.

---

## Usage

```bash
python yoloe.py
```

A live window shows the predictions, and the annotated video is saved to `runs/segment/predict-N/`.

To change mode, uncomment the block you want in `yoloe.py` and comment out the others.

### 1. Visual prompt (default)

You draw a box on a reference image, and YOLOE finds similar objects in the video.

```python
visual_prompts = {
    "bboxes": np.array([[115, 45, 420, 365]]),  # x1, y1, x2, y2 in deer.jpg pixels
    "cls": np.array([0]),
}
class_names = ["deer"]
```

The script works in three steps:

1. **Learn the class:** run the visual prompt once on the reference image with `YOLOEVPSegPredictor`.
2. **Rename it:** with visual prompts, Ultralytics names classes `object0`, `object1`… and has no argument to change that. Setting `model.model.names = class_names` replaces them, and `model.predictor = None` switches back to a normal predictor.
3. **Run on the video** with a plain `model.predict(...)`.

**Multiple classes:** add one box per object, give each a `cls` id, and list the names in id order:

```python
visual_prompts = {
    "bboxes": np.array([[115, 45, 420, 365], [x1, y1, x2, y2]]),
    "cls": np.array([0, 1]),
}
class_names = ["deer", "dog"]
```

### 2. Text prompt

```python
model = YOLOE("yoloe-26x-seg.pt")
names = ["Man", "Woman"]
model.set_classes(names, model.get_text_pe(names))
model.predict(source=video_path, conf=0.25, show=True, save=True)
```

`get_text_pe()` uses the **MobileCLIP2-B** text encoder (`mobileclip2_b.ts`) to turn the class names into embeddings. This is the only mode that uses that file.

### 3. Prompt-free

```python
model = YOLOE("yoloe-26l-seg-pf.pt")
model.predict(source=video_path, conf=0.25, show=True, save=True)
```

---

## Tips for visual prompting

- **Box the whole object.** The model learns the class from what is inside the box. In testing, a box that covered only the deer's chest and legs detected nothing, while a full-body box found the deer in every test frame.
- **Coordinates are in reference-image pixels**, as `[x1, y1, x2, y2]` (top-left and bottom-right corners).
- **Use a low confidence threshold.** Visual-prompt matches often score low (0.15–0.4 here), so this script uses `conf=0.1`.
- **Pick a reference that looks like the target.** A reference image with a similar pose, viewpoint and lighting to the video gives much higher confidence. A frame cropped from the video itself works well.
- **4K video with `show=True` is slow.** Set `show=False` for faster processing, or resize the video first.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing detected with visual prompt | Check that the box covers the whole object and lower `conf` |
| Labels show `object0` | Make sure `model.model.names = class_names` and `model.predictor = None` run before the video `predict()` |
| `refer_image` box ignored | Pass both `refer_image=` and `predictor=YOLOEVPSegPredictor` |
| VS Code: *Import "ultralytics..." could not be resolved* | Ctrl+Shift+P → **Python: Select Interpreter** → `yolo/bin/python` |

---

## Credits

- [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip)
