# 🚦 Traffic Sign Detection

This repository contains a project focused on **object detection of traffic signs** using deep learning techniques. It was created as part of the **Global AI Hub bootcamp** and follows the required submission template structure.

---

## 📌 Introduction

The primary goal of this project is to develop an **Object Detection model** capable of detecting and accurately classifying traffic signs from images. This technology is crucial for applications such as **autonomous vehicles**, intelligent transportation systems, and enhancing road safety.

For this project, I performed the following key steps:

1.  Selected the **Traffic Signs Preprocessed Dataset** from Kaggle.
2.  Applied essential data preprocessing techniques (normalization, augmentation, label encoding).
3.  Implemented and trained a deep learning model for simultaneous **bounding box detection and classification**.
4.  Evaluated the model's performance using standard object detection metrics.

Full technical explanations and step-by-step implementations are available in the included Jupyter notebooks: `data-preprocessing.ipynb` and `model-training.ipynb`.

---

## 📊 Dataset

- **Name:** Traffic Signs Preprocessed Dataset
- **Size:** Approximately 50,000 images
- **Classes:** 43 distinct traffic sign categories
- **Type:** Preprocessed images with bounding box annotations

**Note:** Due to its size, the dataset is not included in this repository. Please download it directly from the [Kaggle Dataset](#links) link below.

---

## ⚙️ Methods

### Data Preprocessing (`data-preprocessing.ipynb`)

- Normalized image pixel values.
- Encoded categorical labels for classification.
- Split the dataset into training, validation, and test sets.
- Applied data augmentation techniques (e.g., rotation, horizontal flipping, scaling) to improve model generalization.
- **Technical Constraint:** Due to **memory constraints** encountered during the preprocessing of 50,000 images, it was not possible to resize and convert the images to a high-resolution RGB format. Consequently, all image inputs were kept at a low resolution of **32x32 pixels**.

### Model Training (`model-training.ipynb`)

- Built a **CNN-based deep learning model** using **TensorFlow/Keras**.
- Implemented a multi-task learning approach combining **bounding box regression** and **classification**.
- The **32x32 input resolution** prevented the effective use of **pretrained models** (like VGG or ResNet) which require higher input dimensions.
- Utilized the **Adam optimizer** along with techniques like **learning rate scheduling** and **early stopping**.
- Evaluated model performance using metrics such as **accuracy**, **confusion matrix**, and **mAP (mean Average Precision)**.

---

## 📈 Metrics & Results

The trained model successfully detected and classified traffic signs across 43 categories.

- **Performance:** The model achieved a satisfactory performance, with the **accuracy falling within the 70%-80% range** (exact results are detailed in the `model-training.ipynb` notebook).
- **Limitation & Impact:** This result was significantly influenced by the **32x32 pixel input resolution** (as detailed in the Data Preprocessing section). This small size drastically restricted the model's capacity to extract rich, distinguishing features, thereby limiting the maximum achievable accuracy.
- **Observations:** Some expected misclassifications occurred between visually similar traffic signs, which is common given the dataset's complexity.

**Conclusion:** The current model is functional and demonstrates the feasibility of the detection task. However, overcoming the memory constraint to allow for high-resolution input and the use of pretrained models is necessary to reach state-of-the-art performance metrics.

---

```markdown
📂 Repository Structure

Traffic-Sign-Detection/
┣ 📂 doc/ # Project documentation
┣ 📓 data-preprocessing.ipynb # Scripts for data cleaning, preparation, and augmentation
┣ 📓 model-training.ipynb # Scripts for model building, training, and evaluation
┣ 📓 tensorboard-n-heatmap.ipynb # Scripts for visualizing training metrics and interpreting model predictions using Grad-CAM
┣ 📄 README.md # Project documentation

---

# 🔮 Future Work

The project can be significantly extended in the following ways:

- **Platform Transition and Hardware Upgrade:** To overcome the current **memory constraint**, future work should involve migrating the process to a system with higher RAM and GPU memory (e.g., a powerful local machine or a cloud-based VM). This platform change is essential to enable the conversion of all 50,000 images to **higher-resolution RGB format** and the effective use of powerful **pretrained deep learning models**, leading to a substantial boost in accuracy.
- **Deployment:** Develop a Streamlit or Flask web application for real-time, interactive traffic sign detection.
- **End-to-End GPU Pipeline:** Optimize the training and inference pipeline for dedicated GPU environments to achieve faster performance.
- **Advanced Models:** Once the resolution constraint is solved via a hardware upgrade, experiment with state-of-the-art detection architectures such as **YOLOv8**, **Faster R-CNN**, or **transformer-based models**.

---

## 🔗 Links

- **Kaggle Dataset:** [Traffic Signs Preprocessed Dataset](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed/)
- **Kaggle Notebook – Data Preprocessing:** [https://www.kaggle.com/code/melisacagilgann/data-preprocessing](https://www.kaggle.com/code/melisacagilgann/data-preprocessing)
- **Kaggle Notebook – Model Training:** [https://www.kaggle.com/code/melisacagilgann/model-training](https://www.kaggle.com/code/melisacagilgann/model-training)
- **Kaggle Notebook – Tensorboard and Heatmap:** [https://www.kaggle.com/code/melisacagilgann/tensorboard-and-heatmap](https://www.kaggle.com/code/melisacagilgann/tensorboard-and-heatmap)
