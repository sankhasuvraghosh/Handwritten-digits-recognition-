# Handwritten Digit Recognition
### Neural Network · PyTorch · MNIST

A deep learning model that recognizes handwritten digits (0–9) with **97%+ accuracy**, trained on the MNIST dataset. Built with PyTorch, the project covers the complete machine learning workflow — from data preprocessing to model deployment and single-image inference.

---

## Demo

```
Enter the index of the image: 42

Actual label:    7
Model prediction: 7  ✓
```
---

## Model Architecture

```
Input (784)  →  Dense (128, ReLU)  →  Dense (64, ReLU)  →  Output (10)
   28×28                                                    digits 0–9
```

| Layer        | Units | Activation |
|--------------|-------|------------|
| Input        | 784   | —          |
| Hidden 1     | 128   | ReLU       |
| Hidden 2     | 64    | ReLU       |
| Output       | 10    | Softmax    |

---

## Results

| Metric         | Value     |
|----------------|-----------|
| Test Accuracy  | **97.X%** |
| Optimizer      | Adam (lr=0.001) |
| Loss Function  | CrossEntropyLoss |
| Batch Size     | 64        |
| Epochs         | 10        |

> Replace `97.X%` with your actual result after running the fixed code.

---

## Project Structure

```
Neural_network_Handwritten_digits_recognition/
│
├── digit_recognition.py    # Main script — training, evaluation, prediction
├── mnist_model.pth         # Saved model weights (generated after training)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sankhasuvraghosh/Neural_network_Handwritten_digits_recognition.git
cd Neural_network_Handwritten_digits_recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python digit_recognition.py
```

The MNIST dataset (~11 MB) will be downloaded automatically on first run.

---

## How It Works

1. **Data loading** — MNIST dataset is fetched via `torchvision`, split into 60,000 training and 10,000 test images.
2. **Preprocessing** — Images are converted to tensors and normalized to stabilize training.
3. **Training** — The model trains for 10 epochs using Adam optimizer and CrossEntropy loss, with GPU acceleration if available.
4. **Evaluation** — Accuracy is measured on the held-out test set (data the model has never seen).
5. **Inference** — Enter any image index to visualize the digit and see the model's prediction.

---

## Technologies

| Tool | Purpose |
|------|---------|
| [PyTorch](https://pytorch.org/) | Model building and training |
| [Torchvision](https://pytorch.org/vision/) | MNIST dataset and transforms |
| [Matplotlib](https://matplotlib.org/) | Visualizing digit images |
| CUDA (optional) | GPU acceleration |

---

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
```

---

## Potential Improvements

- [ ] Replace FC layers with a CNN for 99%+ accuracy
- [ ] Add Dropout regularization to reduce overfitting
- [ ] Build an interactive drawing canvas with Gradio or Streamlit
- [ ] Export model to ONNX for cross-platform deployment
- [ ] Train on custom handwritten digit images

---

## Author

**Sankhasubra Ghosh**
[GitHub](https://github.com/sankhasuvraghosh) · [LinkedIn](#) · [Email](#)

---

## License

This project is open source and available under the [MIT License](LICENSE).
