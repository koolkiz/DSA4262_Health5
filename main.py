import os
import sys
import logging

from processing.nlp_preprocessor import load_clinical_notes
from models.nlp import get_clinical_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load clinical notes
    notes = load_clinical_notes("data/notes")
    logger.info(f"Loaded {len(notes)} clinical notes")
    
    # Get ClinicalBERT embeddings
    embeddings = [get_clinical_embeddings(note) for note in notes]
    logger.info(f"Generated embeddings for {len(embeddings)} clinical notes")

if __name__ == "__main__":
    main()