from concurrent.futures import ThreadPoolExecutor
import spacy, os, igraph as ig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load spacy model with disabled components for speed
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 3000000
nlp.add_pipe("sentencizer")
print("spaCy model loaded.")

# Load tokenizer and model
print("Loading T5 tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
print("T5 tokenizer and model loaded.")

def load_txt_files(directory):
    print(f"Loading text files from directory: {directory}...")
    text_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                print(f"Reading file: {filename}")
                text_data.append(file.read())
    print(f"Loaded {len(text_data)} text file(s).")
    return text_data

def preprocess_texts(text_chunks):
    print("Starting text preprocessing...")
    processed_texts = []
    with ThreadPoolExecutor() as executor:
        for processed in tqdm(executor.map(lambda x: [sent.text.strip() for sent in nlp(x).sents], text_chunks), total=len(text_chunks), desc="Processing Chunks"):
            processed_texts.append(processed)
    print("Text preprocessing completed.")
    return processed_texts

def summarize_text(text):
    print("Generating extractive summary...")
    sentences = preprocess_texts([text])[0]
    print(f"Extracted {len(sentences)} sentences.")
    unique_sentences = list(set(sentences))
    print(f"Reduced to {len(unique_sentences)} unique sentences.")

    vectorizer = TfidfVectorizer(stop_words="english")
    print("Creating TF-IDF matrix...")
    tfidf_matrix = vectorizer.fit_transform(unique_sentences)
    print("TF-IDF matrix created.")

    sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    print("Cosine similarity matrix computed.")

    # graph = ig.Graph.Adjacency((sim_matrix > 0).tolist(), mode="UNDIRECTED")
    graph = ig.Graph.Adjacency((sim_matrix > 0).astype(int).toarray().tolist(), mode="UNDIRECTED")

    print("Graph created for sentence ranking.")
    scores = graph.pagerank()
    print("PageRank scores computed.")

    ranked_sentences = sorted(zip(scores, unique_sentences), reverse=True)[:5]
    print("Top sentences selected for summary.")
    return " ".join([sentence for _, sentence in ranked_sentences])

def generate_abstractive_summary(text, max_length=150):
    print("Generating abstractive summary...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, early_stopping=True)
    abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Abstractive summary generated.")
    return abstractive_summary

# Main Execution
directory = "books/txt"
print("Starting main execution...")
raw_texts = load_txt_files(directory)

if raw_texts:
    print("Processing first text file...")
    sample_text = raw_texts[0]

    # Extractive summary
    print("Step 1: Extractive summarization...")
    extractive_summary = summarize_text(sample_text)
    print(f"Extractive Summary:\n{extractive_summary}")

    # Abstractive summary
    print("Step 2: Abstractive summarization...")
    abstractive_summary = generate_abstractive_summary(extractive_summary)
    print(f"Abstractive Summary:\n{abstractive_summary}")

    # Combine results
    final_summary = f"Extractive Summary: {extractive_summary}\n\nAbstractive Summary: {abstractive_summary}"
    print("Final Hybrid Summary:")
    print(final_summary)
else:
    print("No text files found in the directory.")