# Biomimetic Visual-Cognitive Agent

This repository contains the official source code and experimental datasets for the paper:  
**"A Biomimetic Visual-Cognitive Architecture for Resilient and Stealthy Graphical User Interface Automation"**

## 📌 Overview
Traditional RPA (Robotic Process Automation) frameworks rely heavily on DOM structures and exhibit zero-variance, deterministic execution, making them highly vulnerable to GUI deformations and behavioral biometric heuristic detection. 

This project introduces a closed-loop Visual-Cognitive-Motor architecture that:
1. Uses pure-pixel **Morphological Image Processing** and localized ONNX-OCR for spatial grounding (DOM-independent).
2. Manages context and token overhead via a **Sliding-Window Deque Memory**.
3. Integrates **Fitts' Law** and the **Keystroke-Level Model (KLM)** with stochastic noise to mathematically synthesize human-like typing kinetics.

## 📂 Repository Structure
* `GM.py`: The core biomimetic visual-cognitive agent implementation.
* `paper_experiment_data_v2.csv`: The end-to-end execution telemetry data (N=228 interactions) used for the K-S test and latency evaluations in the paper.

## 🚀 Quick Start
### Prerequisites
* Python 3.10+
* `mss`, `opencv-python`, `rapidocr-onnxruntime`, `openai`

```bash
pip install -r requirements.txt
python GM.py
