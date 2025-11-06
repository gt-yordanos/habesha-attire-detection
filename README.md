<p align="center">
  <img src="HabeshaAttireLogo.jpg" alt="Habesha Attire Logo" width="70" />
</p>

<h1 align="center">Habesha Attire Detection</h1>

<p align="center">
  <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/tensorflow.svg" alt="TensorFlow" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> 
  A deep learning project built in <strong>Jupyter Notebook</strong> using <strong>TensorFlow</strong> and <strong>Keras</strong> to identify Habesha traditional clothing from images.
</p>

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/info.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Overview
This project focuses on detecting **Habesha traditional attire (Habesha libs)** from images using a **Convolutional Neural Network (CNN)**.  
It was created as part of my personal learning journey in computer vision and deep learning.

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/database.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Dataset
I gathered around **1,000 images** of both Habesha and non-Habesha traditional clothes using the **Chrome Bulk Image Downloader** extension.  
The data was split into **training** and **validation** sets, with image augmentation applied to make the model more robust.

---

## <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/python.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Tools & Libraries
- **TensorFlow / Keras** – Model building and training  
- **OpenCV (cv2)** – Image preprocessing  
- **scikit-learn** – Evaluation metrics (classification report, confusion matrix)  
- **Matplotlib** – Visualization of accuracy and loss  
- **ImageDataGenerator** – Data augmentation (rotation, shift, zoom, flip, etc.)

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/code.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Installation Guidelines

```bash
pip install tensorflow
```
```bash
pip install opencv-python
```
```bash
pip install scikit-learn
```
```bash
pip install matplotlib
```

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/layers.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Model Summary
The CNN includes:
- 3 Convolutional layers (with ReLU and MaxPooling)  
- 1 Fully connected layer with dropout for regularization  
- 1 Sigmoid output layer for binary classification  

**Compilation Details:**  
- Optimizer: `Adam`  
- Loss: `Binary Crossentropy`  
- Metrics: `Accuracy`

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/bar-chart-2.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Results
After 20 epochs of training:

| Metric | Value |
|--------|--------|
| **Validation Accuracy** | **71%** |
| **Precision (Habesha Libs)** | 0.73 |
| **Recall (Habesha Libs)** | 0.80 |
| **F1-Score** | 0.76 |

The model performs decently, showing it has learned general visual patterns of Habesha attire.

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/alert-circle.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Limitations
The model sometimes **confuses white suits and other cultural white attires (like Indian traditional clothing)** with Habesha clothes.  
This is mainly because my dataset lacked enough variation in non-Habesha white clothing.

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/rocket.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Future Plans
To improve the model:
- Expand the dataset with more white traditional attires from other cultures  
- Train the model on **diverse backgrounds** and lighting  
- Possibly fine-tune with **transfer learning (e.g., VGG16 or MobileNet)** for better feature extraction  

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/lightbulb.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Reflection
This was a hands-on learning experience in:
- Image collection and preprocessing  
- CNN architecture design  
- Evaluating and improving model performance  

It’s a solid foundation that I plan to build upon in my AI/ML journey.

---

## <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/external-link.svg" width="20" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:5px;"/> Links
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YordanosTefera/habesha-attire-detection)  
- **LinkedIn:** [Yordanos Tefera](https://www.linkedin.com/in/yordanosgtefera/)

---
<div style="background-color:red;">Test</div>
<p align="center">
  <img src="https://cdn.jsdelivr.net/npm/lucide-static/icons/award.svg" alt="Thank You" width="120" style="fill:#FFD700; background-color:#ffffff; border-radius:50%; padding:10px;"/><br/>
  <strong>Thank you for checking out my project!</strong>
</p>