# 🩺 YOLOv12 Polyp Detection

![Model](https://img.shields.io/badge/Model-YOLOv12-blue)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-black)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Dataset](https://img.shields.io/badge/Dataset-Roboflow-orange)
![Notebook](https://img.shields.io/badge/Environment-Jupyter-informational)
![License](https://img.shields.io/badge/License-Ultralytics-green)

Automatic **polyp detection from endoscopic images** using **YOLOv12**, trained on a **Roboflow Polyp Detection dataset**.
The full pipeline—dataset loading, training, evaluation, and inference—is implemented in a Jupyter Notebook.

---

## 📌 Overview

Colorectal polyp detection is a crucial step in early colorectal cancer prevention.
This project applies **YOLOv12**, a next-generation real-time object detection model from **Ultralytics**, to accurately localize polyps in medical images.

### ✨ Highlights

* YOLOv12 real-time object detection
* Transfer learning with pretrained weights
* Roboflow-hosted annotated medical dataset
* End-to-end notebook-based workflow

---

## 🧠 Model Details

* **Architecture:** YOLOv12
* **Framework:** Ultralytics (PyTorch-based)
* **Task:** Object Detection
* **Classes:** Polyp (single class)
* **Training Strategy:** Fine-tuning pretrained YOLOv12 weights

---

## 📦 Dataset

This project uses the **Polyp Detection dataset** hosted on **Roboflow**.

🔗 **Dataset & Model Link**
[https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models](https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models)

### Dataset Features

* Annotated colonoscopy/endoscopy images
* Bounding-box annotations for polyps
* Exported in **YOLO format**
* Train / Validation / Test splits

### Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Annotation Format (YOLO)

```
<class_id> <x_center> <y_center> <width> <height>
```

(All values normalized between 0 and 1)

---

## 📂 Project Structure

```
.
├── Copy_of_yolov12polypdetection.ipynb   # Main YOLOv12 notebook
├── dataset/                              # Roboflow polyp dataset
├── runs/                                 # Training & inference outputs
├── weights/                              # Saved YOLOv12 model weights
└── README.md                             # Project documentation
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* Ultralytics (YOLOv12)
* OpenCV
* NumPy
* Matplotlib
* Jupyter Notebook

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## 🚀 How to Run

1. Open the notebook:

   ```bash
   jupyter notebook Copy_of_yolov12polypdetection.ipynb
   ```
2. Download the dataset from Roboflow in **YOLO format**.
3. Update dataset paths if required.
4. Run the notebook cells sequentially to:

   * Train the YOLOv12 model
   * Evaluate performance
   * Run inference and visualize results

---

## 📊 Training Outputs

YOLOv12 automatically generates:

* Training & validation loss curves
* Precision, Recall, and mAP metrics
* Best and last model checkpoints

Saved under:

```
runs/train/
```

---

## 🔍 Inference Results

* Bounding boxes around detected polyps
* Confidence scores for each detection
* Output images saved in:

```
runs/detect/
```

---

## 🧪 Applications

* Medical image analysis
* Colonoscopy screening assistance
* Healthcare AI research
* Benchmarking with YOLOv8, YOLOv10, YOLOv11, and Faster R-CNN

---

## ⚠️ Disclaimer

🚨 **For research and educational purposes only**
This project is **not approved for clinical or diagnostic use**.

---

## 📜 License

* **YOLOv12:** Ultralytics License
* **Dataset:** Roboflow Dataset License

Please review the respective platforms for full licensing terms.

---

## 🙌 Acknowledgements

* **Roboflow Polyp Detection Dataset**
* **Ultralytics YOLO**
* Open-source medical imaging community

---

## ⭐ Support

If you find this project useful, please consider **starring the repository** ⭐
Contributions and suggestions are welcome!
