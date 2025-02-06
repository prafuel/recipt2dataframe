
import fitz  # PyMuPDF

from PIL import Image
import io
import pytesseract
import cv2

import pandas as pd
from fuzzywuzzy import fuzz

from pandas import DataFrame

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def extract_text_from_img(image):
    text = (
        pytesseract.image_to_string(image)
    ).lower().strip()

    return text

def compare_dataframes(df1, df2):
    # Ensure both DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape.")

    # Apply fuzz.ratio element-wise
    similarity_df = df1.copy()
    for col in df1.columns:
        similarity_df[col] = [
            fuzz.ratio(str(a), str(b)) for a, b in zip(df1[col], df2[col])
        ]

    return similarity_df

if __name__ == "__main__":
    # img = "./flipkart_recipt.png"
    # image=cv2.imread(img, 0)

    # text = extract_text_from_img(image)
    # print(text)

    pass