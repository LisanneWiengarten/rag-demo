import base64
import glob
import os
from io import BytesIO

import pymupdf
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_with_llm_ocr(pdf_path: str) -> str:
    """
    Converts the given pdf to LLM-readable images page by page for OCR.
    Collects all text from all pages.
    :param pdf_path: Path to a pdf file whose text should be extracted via LLM OCR
    :return: The full text of this pdf
    """
    all_pages_content = ""
    # Each page becomes a single image that is sent to LLM
    images = convert_from_path(pdf_path)
    for image in tqdm(images, desc=f"Performing LLM OCR on file {pdf_path}"):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = llm_client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bitte extrahiere den vollständigen Text aus diesem Bild. Bitte "
                                                 "antworte ausschließlich mit dem extrahierten Text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                },
            ]
        )
        extracted_content = response.choices[0].message.content
        all_pages_content += "\n" + extracted_content
    return all_pages_content

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extract text from a single PDF file.
    If a file is image-based, it needs to be sent to an LLM for OCR instead of simple scraping
    :param pdf_file_path: Path to a pdf file whose text should be extracted
    :return: The full text of this pdf
    """
    all_pages_text = ""
    doc = pymupdf.open(pdf_file_path)
    for page in doc:
        all_pages_text += page.get_text()

    if len(all_pages_text) > 1:
        return all_pages_text
    # Use LLM OCR if simple extraction was not successful
    return extract_text_with_llm_ocr(pdf_file_path)

def extract_text_from_folder(input_folder: str) -> dict[int, dict]:
    """
    Walks through all pdfs in the given folder and extracts the text from them
    For simplicity, assuming that there are no subfolders
    :param input_folder: pdf folder to extract text from
    :return: A dictionary with ids as keys, and dictionaries with file_name and extracted texts as values
    """
    all_pdf_texts = {}
    for i, pdf_path in enumerate(glob.glob(os.path.join(input_folder, "*.pdf"))):
        file_text = extract_text_from_pdf(pdf_path)
        all_pdf_texts[i] = {
            "file_name": pdf_path,
            "full_text": file_text
        }
    return all_pdf_texts
