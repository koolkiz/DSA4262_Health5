import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
import os 
from docx import Document
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_clinical_text(text):
    """Cleans clinical text for NLP processing."""
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    text = text.lower()
    text = text.split("\n", 1)[1]
    return text

def load_docx(file_path):
    """Load a .docx file and extract text"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
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


documents = load_clinical_notes("data/notes")
print(documents)

# Preprocess each document and create a list of tokenized sentences
tokenized_documents = [clean_clinical_text(doc) for doc in documents]

# Train the Word2Vec model using Skip-gram (sg=1)
model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, sg=1)

# To represent each document as a vector, average the embeddings of the words in the document
def get_document_vector(document_tokens, model):
    # Get the word embeddings for each word in the document
    word_vectors = [model.wv[word] for word in document_tokens if word in model.wv]
    
    if word_vectors:
        # Average the word vectors to get the document vector
        document_vector = sum(word_vectors) / len(word_vectors)
        return document_vector
    else:
        # Return a zero vector if no words from the document are in the model's vocabulary
        return [0] * model.vector_size

# Represent each document as a vector
document_vectors = [get_document_vector(doc, model) for doc in tokenized_documents]

# Save the document vectors to a csv file
df = pd.DataFrame(document_vectors)

# open a new column for document id 
df['doc_id'] = range(1, len(df) + 1)

# save the document vectors to a csv file
df.to_csv('document_vectors.csv', index=False)
