
import fitz  # PyMuPDF

from PIL import Image
import io
import pytesseract
import cv2

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def extract_text_from_img(image):
    text = (
        pytesseract.image_to_string(image)
    ).lower().strip()

    return text

if __name__ == "__main__":
    img = "./flipkart_recipt.png"
    image=cv2.imread(img, 0)

    text = extract_text_from_img(image)
    print(text)
