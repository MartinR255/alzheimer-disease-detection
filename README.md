# Alzheimer Disease Detection using 3D CNNs

## Project Overview

In this work, we aim to explore methods for the detection of Alzheimer's disease, with a focus on deep learning techniques such as convolutional neural networks (CNNs). Our objectives include analysing existing deep learning approaches for Alzheimer's disease detection using MRI images and implementing a neural network that achieves optimal performance in the classification task. Since neural networks can sometimes rely on irrelevant patterns that may not align with the diagnostic reasoning of medical professionals, we also aim to incorporate interpretability methods such as Grad-CAM. The final outcome of this thesis will be a classification pipeline that is capable of detecting Alzheimer's disease from MRI scans while also providing some degree of interpretability to help better understand the decision making process of the trained neural network.

---

## Installation

---

## Usage

- **Training:**  

- **Testing and Evaluation:**  
  
- **Pipeline Execution:**  


---
## Data
Folder data contains data splits in JSON format used in our four classification experiments. Each experiment has training, validation and testing split. Each key is a file name in a format **imageID-subjectID.nii.gz : label.**

---

## References

- [MONAI: Medical Open Network for AI](https://monai.io/)
- [PyTorch](https://pytorch.org/)
- [ANTsPy](https://antspy.readthedocs.io/en/latest/)
- [Advanced AI explainability for PyTorch](https://github.com/jacobgil/pytorch-grad-cam)
- [Captum](https://captum.ai/)
- [M3D-CAM](https://arxiv.org/abs/2007.00453)
---

