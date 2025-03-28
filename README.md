# Swin Transformer Demo

This project performs real-time video frame classification using a **pretrained Swin Transformer** model and a webcam feed. It predicts and displays the most probable label from the **ImageNet** class set (1000 classes) on each frame.

---

## 🔍 Why Swin Transformer?

The [Swin Transformer](https://arxiv.org/abs/2103.14030) is a powerful **vision transformer** architecture that achieves **state-of-the-art results** on many computer vision benchmarks. It’s selected for this project due to the following reasons:

- ✅ **Hierarchical Representation**: Swin Transformer computes visual features in a hierarchical manner (like CNNs), making it efficient for image classification tasks.
- ✅ **Window-based Self-Attention**: Reduces computation while maintaining accuracy, allowing real-time inference on GPUs.
- ✅ **Pretrained on ImageNet**: The `swin_tiny_patch4_window7_224` model from the `timm` library is already fine-tuned on ImageNet-1K, making it ideal for out-of-the-box classification.
- ✅ **Compact & Fast**: The "Tiny" variant balances **speed and accuracy**, allowing smooth real-time predictions on most modern machines with GPU support.

---

## 📷 What This Project Does

- Captures **live video feed** using OpenCV.
- Applies **image preprocessing**: resizing, normalization, and tensor conversion.
- Uses a **Swin Transformer** model to classify each frame.
- Annotates the frame with the **top prediction** and **confidence score**.
- Displays the annotated frame in real time.

---

## 🧰 Dependencies

Install the following packages (Python ≥ 3.8 recommended):

    pip install torch torchvision timm opencv-python pillow

When it starts running: 
- Loads a pretrained Swin Transformer model for image classification.
- Captures real-time video from the webcam.

For each frame:
  - Applies preprocessing.
  - Performs classification using the model.
  - Displays the frame with the predicted label and confidence score.

Press 'q' to stop the application.


### Model Name: swin_tiny_patch4_window7_224
### Pretrained On: ImageNet-1K
### Input Size: 224x224 RGB images
### Framework: PyTorch + TIMM (PyTorch Image Models)

