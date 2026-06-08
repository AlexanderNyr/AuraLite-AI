# 🌟 AuraLite AI

**AuraLite AI** is a lightweight, educational Large Language Model (LLM) implemented using **PyTorch**. It is designed to demonstrate the inner workings of the Transformer architecture (the foundation of models like GPT-4) in a way that is accessible and runnable on consumer hardware.

## 🚀 Key Features
- **PyTorch Engine**: Transitioned from NumPy to PyTorch for professional-grade tensor operations and optimization.
- **Transformer Architecture**: A decoder-only transformer featuring a custom Self-Attention mechanism and Layer Normalization.
- **Hardware Acceleration**: Automatic detection and usage of **NVIDIA CUDA** (GPU) for training and generation, with a seamless fallback to **CPU**.
- **Advanced GUI**: A comprehensive control panel built with `tkinter` that allows real-time interaction with the model.
- **Hyperparameter Tuning**: Full control over the AI's "brain" directly from the interface:
  - **Learning Rate**: Controls how fast the model adapts to new data.
  - **Epochs**: Determines how many times the model studies the dataset.
  - **Model Dimension (D_Model)**: Sets the size of the internal vector representations.
  - **Feed-Forward Dimension (D_FF)**: Controls the capacity of the processing layers.
  - **Context Window (Seq Length)**: Defines how many previous characters the AI considers when predicting the next one.
- **Custom Training**: Upload any `.txt` file to teach the AI specific styles, languages, or fictional worlds.
- **Interruptible Training**: Ability to stop training at any point and preserve the learned weights for immediate testing.

## 🛠 Technical Specifications
- **Framework**: PyTorch (Tensors, Autograd, Adam Optimizer).
- **Attention**: Scaled Dot-Product Self-Attention with Causal Masking.
- **Optimizer**: Adam (Adaptive Moment Estimation) for faster and more stable convergence.
- **Input/Output**: Character-level tokenization.

## 📦 Installation & Setup

### Prerequisites
- **Python 3.8+** (Recommended)
- **NVIDIA GPU** (Optional, for CUDA acceleration. Requires CUDA Toolkit installed).

### Manual Installation
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install torch numpy
   ```

## 📖 How to Use
1. **Launch the App**:
   ```bash
   python gui_app.py
   ```
2. **Configure & Train**:
   - Adjust the **Hyperparameters** to suit your hardware and dataset.
   - Click **"Select .txt File"** and provide your training data.
   - Click **"Start Training"**. Monitor the **Loss** value; a decreasing loss indicates the AI is learning.
3. **Generate Text**:
   - Enter a **Seed phrase** to give the AI a starting point.
   - Set the desired **Length** of the output.
   - Click **"Generate Text"** and watch the AI create content based on its training.

## 🔨 Compiling to .exe (Windows)
To bundle the application into a single portable executable:
1. Run the provided `build_exe.bat` file.
2. The script will automatically install `PyInstaller` and bundle the PyTorch environment.
3. The final executable will be located in the `dist/` folder as `AuraLite_AI_CUDA.exe`.

## ⚠️ Hardware Compatibility Note
- **CUDA Acceleration**: Requires an NVIDIA GPU with Compute Capability 5.0 or higher.
- **CPU Fallback**: If a compatible GPU is not detected, AuraLite AI automatically switches to CPU mode. While slower, it remains fully functional.
- **Memory Tip**: For CPU-only users, keeping `D_Model` at 64 and `Seq Length` at 16 is recommended for optimal performance.
