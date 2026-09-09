# Plant Disease Detection System

A Streamlit application that uses a trained TensorFlow/Keras model to identify common tomato leaf conditions from uploaded images. The app also provides disease information, treatment suggestions, severity scoring, prediction history, Grad-CAM visualisation, and a LIME-style explanation.

## Features

- Upload JPG, JPEG, or PNG leaf images
- Classify images into:
  - Tomato Early Blight
  - Tomato Late Blight
  - Healthy
- Display prediction confidence and severity level
- Show disease descriptions and treatment steps
- Generate smart recommendations based on the prediction
- Visualise model attention with Grad-CAM
- Display a LIME-style highlighted explanation
- Support English and Hindi labels
- Keep prediction history during the current Streamlit session

## Demo

Run the application locally and open the URL shown in the terminal:

```text
http://localhost:8501
```

## Requirements

- Python 3.10 or later
- Git
- A working TensorFlow installation

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Jyothsnasri5607/plant_project.git
cd plant_project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit tensorflow numpy pillow opencv-python
```

## Run the Application

From the repository root, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser, upload a plant leaf image, and review the prediction and explanations.

To stop the application, press `Ctrl+C` in the terminal.

## Project Structure

```text
plant_project/
|-- app.py                     # Streamlit application entry point
|-- model/
|   |-- __init__.py             # Model package marker
|   |-- plant_model.h5          # Trained TensorFlow/Keras model
|   |-- disease_info.py         # Disease descriptions and treatments
|   |-- gradcam.py              # Grad-CAM visualisation
|   |-- lime_explainer.py       # LIME-style explanation
|   |-- predict.py              # Standalone prediction helper
|   |-- train.py                # Model training script
|   |-- test.JPG                # Sample test image
|   `-- gradcam.jpg             # Stored visualisation asset
|-- .gitignore                  # Ignored datasets, model files, and caches
`-- README.md                   # Project documentation
```

## Supported Classes

| Class | Meaning |
| --- | --- |
| `Tomato___Early_blight` | Tomato leaf affected by early blight |
| `Tomato___Late_blight` | Tomato leaf affected by late blight |
| `Healthy` | No disease detected by the model |

## How It Works

1. The uploaded image is converted to RGB and resized to `224 x 224` pixels.
2. Pixel values are normalised to the range `0.0` to `1.0`.
3. The TensorFlow model predicts the most likely class.
4. The highest prediction confidence is mapped to a severity level.
5. Disease information, treatment guidance, recommendations, and visual explanations are displayed.

## Troubleshooting

### `streamlit` is not recognised

Activate the virtual environment and run Streamlit through Python:

```bash
python -m streamlit run app.py
```

### TensorFlow installation fails

Use a supported Python version, preferably Python 3.10 or 3.11, create a fresh virtual environment, and install TensorFlow again:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow
```

### Model file not found

Run the application from the repository root and confirm that this file exists:

```text
model/plant_model.h5
```

### Grad-CAM fails

Grad-CAM expects the loaded model to contain a convolutional layer named `Conv_1`. The main prediction still appears, but the Grad-CAM section may show an error if a different model architecture is used.

## Development Notes

- The trained model is loaded from `model/plant_model.h5` when the app starts.
- The model is used for inference only by the Streamlit application.
- Prediction history is stored in Streamlit session state and is cleared when the session is restarted.
- The current LIME module provides a highlighted-region explanation rather than a full `lime` package implementation.

## License

This project does not currently include a license file. Add a license before distributing or reusing the project commercially.
