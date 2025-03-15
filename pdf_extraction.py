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
    all_pages_content = ""
    images = convert_from_path(pdf_path)
    for image in tqdm(images):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = llm_client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bitte extrahiere den vollständigen Text aus diesem Bild"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                },
            ]
        )
        extracted_content = response.choices[0].message.content
        all_pages_content += "\n" + extracted_content
    return all_pages_content

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extract text from a single PDF file."""
    all_pages_text = ""
    doc = pymupdf.open(pdf_file_path)
    for page in doc:
        all_pages_text += page.get_text()

    if len(all_pages_text) > 1:
        return all_pages_text
    return extract_text_with_llm_ocr(pdf_file_path)

def extract_text_from_folder(input_folder: str) -> list[str]:
    all_pdf_texts = []
    # Assuming that there are no subfolders
    for pdf_path in glob.glob(os.path.join(input_folder, "*.pdf")):
        file_text = extract_text_from_pdf(pdf_path)
        all_pdf_texts.append(file_text)
    return all_pdf_texts
