import spacy

nlp = spacy.load("de_core_news_lg")

def split_pdfs_into_chunks(text_data: dict):
    for extracted_file in text_data.values():
        chunks = []
        doc = nlp(extracted_file["full_text"])
        current_chunk = []
        for sent in doc.sents:
            current_chunk.append(sent.text)
            if len(current_chunk) >= 10 or sent.text.endswith("\n\n"):
                section_text = " ".join(current_chunk).strip()
                if section_text:
                    chunks.append(section_text)
                current_chunk = []
        # Reste auffangen
        if current_chunk:
            section_text = " ".join(current_chunk).strip()
            if section_text:
                chunks.append(section_text)
        extracted_file["chunks"] = chunks
    return text_data
