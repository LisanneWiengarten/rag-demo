import spacy
from tqdm import tqdm

nlp = spacy.load("de_core_news_lg")

def split_texts_into_chunks(text_data: dict):
    """
    Splits the full_text of every given file into chunks
    :param text_data: A dictionary with ids as keys, and dictionaries with file_name and extracted texts as values
    :return: The input dictionary, but now each value-dictionary has a new item "chunks", which is a list of strings
    """
    for extracted_file in tqdm(text_data.values(), desc="Splitting texts into chunks:"):
        chunks = []
        doc = nlp(extracted_file["full_text"])
        current_chunk = []

        for sent in doc.sents:
            current_chunk.append(sent.text)
            # Arbitrary chunk size of 10 sentences or paragraph
            if len(current_chunk) >= 10 or sent.text.endswith("\n\n"):
                section_text = " ".join(current_chunk).strip()
                if section_text:
                    chunks.append(section_text)
                current_chunk = []

        # Catch any leftovers
        if current_chunk:
            section_text = " ".join(current_chunk).strip()
            if section_text:
                chunks.append(section_text)
        extracted_file["chunks"] = chunks
    return text_data
