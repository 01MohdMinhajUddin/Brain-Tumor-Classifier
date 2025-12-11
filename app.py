import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import gradio as gr

class_names=['pituitary', 'notumor', 'meningioma', 'glioma']

#==========================
# Load the model
#==========================
model=load_model("BTumor.keras")

# Predict Function

def predict(img):

    # image rgb,resize,to array and expanding dims
    img= img.convert('RGB')
    img=img.resize((128,128))
    img_array=image.img_to_array(img)/255
    img_array=np.expand_dims(img_array,axis=0)

    # predict

    pred=model.predict(img_array)[0]
    pred_idx=np.argmax(pred)
    pred_class= class_names[pred_idx]
    confidence=float(pred[pred_idx])*100

    result=(f"**Prediction:** {pred_class}\n\n**Confidence:** {confidence:.2f}% ")
    return result


demo= gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil",label="Upload The Brain MRI"),
    outputs=gr.Markdown(label="Result"),
    title="Brain Tumor MRI Classifier",
    description="Upload an MRI Image to Classify as : pituitary, notumor, meningioma, or glioma."
)

demo.launch(share=True)
