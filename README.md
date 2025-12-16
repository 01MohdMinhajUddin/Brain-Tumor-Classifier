# Brain Tumor MRI Classifier 🧠

This is a deep learning project I built to classify brain MRI images into four categories using TensorFlow and Gradio:

- **Pituitary**
- **No tumor**
- **Meningioma**
- **Glioma**

>  This is a learning project, **not** a medical tool. It should never be used for real diagnosis.

---

## Model Architecture

This project uses **transfer learning** with **VGG16** pretrained on ImageNet:

- Base model: `VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))`
- Trainable layers: last convolutional block fine-tuned
- Custom classification head: GlobalAveragePooling2D + Dense layers + softmax
- Framework: **TensorFlow / Keras**


## What’s in this repo?

- `BTumor.ipynb` – Jupyter notebook with the full training pipeline  
  (data loading, augmentation, model architecture, training, evaluation).
- `app.py` – Gradio web app that loads the trained model and predicts on uploaded MRI images.
- `requirements.txt` – Python packages needed to run `app.py`.

The trained model file is **not** stored in this repo because it’s larger than GitHub’s 100 MB limit.

---

## Download the trained model

Download the model file `BTumor.keras` from Google Drive:

 **[Download BTumor.keras](https://drive.google.com/file/d/1IebGykinUH273nRjvAhXA7qkQ5gd9fen/view?usp=drive_link)**

After downloading, put the file in the root of this project, next to `app.py`, like this:

```text
Brain-Tumor-Classifier
├── app.py
├── BTumor.keras   <- place the downloaded file here
├── BTumor.ipynb
└── requirements.txt
```

## How I trained the model (summary)

1. Loaded MRI images from four folders (one per class).
2. Resized images to 128×128 and scaled pixel values to [0, 1].
3. Applied simple augmentation (brightness and contrast changes).
4. Built a CNN in Keras with `sparse_categorical_crossentropy` loss.
5. Trained on the training set, validated on a hold-out set.
6. Saved the final model as `BTumor.keras` and used it in `app.py`.

## How to run this on your machine

1. **Clone the repo**

   ```bash
   git clone https://github.com/01MohdMinhajUddin/Brain-Tumor-Classifier.git
   cd Brain-Tumor-Classifier
   ```

2. **Create a virtual environment**

  ```bash
  python -m venv .venv
  .venv\Scripts\activate         # on Windows
  # source .venv/bin/activate    # on macOS / Linux
```

3. **Install dependencies**
  
  ```bash 
pip install -r requirements.txt
```

4. **Run the app**

  ```bash
python app.py
```




