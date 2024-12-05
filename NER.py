import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load a pre-trained BERT model for biomedical text
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Define the pipeline for Named Entity Recognition (NER)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

def extract_medical_terms(text):
    """
    Extracts key terms and useful data from a medical report.
    
    Args:
        text (str): The text of the medical report.

    Returns:
        List of key terms and their types.
    """
    # Run the NER pipeline on the input text
    ner_results = ner_pipeline(text)
    
    # Filter the entities by the types we want to extract (e.g., diagnoses, symptoms)
    key_terms = []
    for entity in ner_results:
        # Adjust based on specific entity types if necessary
        entity_info = {
            "word": entity['word'],
            "entity_type": entity['entity_group'],  # NER model assigns a general entity group
            "score": entity['score']  # Confidence score for the extraction
        }
        key_terms.append(entity_info)
    
    return key_terms

# Example medical report text
medical_report = """
Patient exhibits symptoms of severe headache and nausea. 
Diagnosis includes possible migraine. Prescribed medication includes acetaminophen and ibuprofen.
"""

# Extract terms
extracted_terms = extract_medical_terms(medical_report)
print("Extracted Medical Terms and Information:")
for term in extracted_terms:
    print(f"Word: {term['word']}, Type: {term['entity_type']}, Confidence: {term['score']:.2f}")
