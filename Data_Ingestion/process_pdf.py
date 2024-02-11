import fitz

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text