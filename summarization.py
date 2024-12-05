import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"  # or try "t5-large" for better results
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load a larger dataset for better summarization (increase sample size or use different dataset)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:20%]")  # Example: using 20% of the test set

# Preprocess text (tokenization)
def preprocess_function(examples):
    return tokenizer(examples['article'], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Perform Abstractive Summarization
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], 
                                 num_beams=6,  # Increase beam search for better output
                                 min_length=60, 
                                 max_length=200, 
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Get some text samples for summarization
sample_texts = tokenized_dataset["article"][:5]  # Adjust sample size
summaries = [summarize(text) for text in sample_texts]

# Reference summaries (use 'highlights' column for reference summaries)
reference_summaries = tokenized_dataset["highlights"][:5]  # Access the 'highlights' column directly

# ROUGE Score Calculation
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

for summary, reference in zip(summaries, reference_summaries):
    scores = scorer.score(reference, summary)
    for key in rouge_scores:
        rouge_scores[key].append(scores[key].fmeasure)

# Calculate the average ROUGE scores
avg_rouge_scores = {key: sum(value) / len(value) for key, value in rouge_scores.items()}

# avg_rouge_scores['rouge1'] = 
# Print Average ROUGE Scores
print(f"Average ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
print(f"Average ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
print(f"Average ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")

# Plotting the ROUGE Scores
labels = list(avg_rouge_scores.keys())
values = list(avg_rouge_scores.values())

plt.bar(labels, values)
plt.xlabel('ROUGE Score')
plt.ylabel('F-Score')
plt.title('ROUGE Evaluation')
plt.show()


def summarize_short(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], 
                                 num_beams=6,  # Keeping the beam search to improve quality
                                 min_length=30,  # Reducing the minimum length
                                 max_length=100,  # Reducing the maximum length
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Sample clinical text for summarization
sample_clinical_text = """
The patient is a 56-year-old male with a history of hypertension and type 2 diabetes. He presented with chest pain and shortness of breath, 
which started two days ago. The patient reports a family history of coronary artery disease. On examination, his blood pressure was 
160/90 mmHg, and heart rate was elevated at 98 bpm. An ECG was performed and showed signs of possible ischemia. The patient was admitted 
for further evaluation and a cardiac workup, including a coronary angiogram.
"""

# Summarizing the clinical text
summary_clinical = summarize_short(sample_clinical_text)

# Output the clinical summary
print("Original Clinical Text:")
print(sample_clinical_text)
print("\nSummarized Clinical Text (Shortened):")
print(summary_clinical)