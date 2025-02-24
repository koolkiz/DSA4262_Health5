import os
import sys
import logging
import re
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_docx(file_path):
    """Load a .docx file and extract text"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def clean_clinical_text(text):
    """Cleans clinical text for NLP processing."""
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    text = text.lower()
    text = text.split("\n", 1)[1]
    return text

def load_clinical_notes(file_path):
    logger.info(f"Loading all clinical data from {file_path}")
    notes = []
    for root, dirs, files in os.walk(file_path):
        print(root)
        for file in files:
            if file.endswith(".docx"):
                note = load_docx(os.path.join(root, file))
                note = clean_clinical_text(note)
                notes.append(note)
    return notes                

