import streamlit as st
from PIL import Image
import io
import pandas as pd

from modules.inferencing_func import inference_fn
from modules.wrapper import time_taken
from modules.helper import extract_text_from_pdf, compare_dataframes
from modules.llm_model import llm_pipeline_get_df

from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load Model & Processor
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Models
models_info = {
    "donut_cord": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "donut-doc": ""
}

def get_processor_and_model(model_key: str):
    model_name = models_info.get(model_key)
    if not model_name:
        st.error("Invalid model selection.")
        return None, None
    
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

def for_pdf_data(file_data):
    text = extract_text_from_pdf(file_data)
    df = llm_pipeline_get_df(text)
    return df

def for_image_data(image):
    """Run inference on the uploaded image."""
    df = inference_fn(image, model, processor)
    return df

@time_taken
def main():
    st.title("Convert Your Invoice into an Interactive Tabular Format")
    solution_options = ["donut_cord", "prompting_soln"]

    uploaded_file = st.file_uploader("Upload an invoice image or PDF", type=["jpg", "png", "jpeg", "pdf"])

    if not uploaded_file:
        st.info("Please upload an image or PDF.")
        return

    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "pdf":
        solution_options = ["prompting_soln"]
    
    image = None
    if extension in ["jpg", "png", "jpeg"]:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    selected_solution = st.selectbox("Choose Solution", solution_options)
    submit = st.button("Submit")

    dfs = pd.DataFrame()

    if submit:
        with st.spinner("Processing..."):
            if type(image) == str():
                # pdf
                dfs = []
            else:
                dfs = for_image_data(image)

        print("dataframes : ", dfs)

        for df in dfs:
            name = df[0]
            dataframe = df[1]
            st.text(name)
            st.dataframe(dataframe)
    

if __name__ == "__main__":
    main()