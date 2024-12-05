import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

warnings.filterwarnings('ignore')

class ClinicalDocumentationSystem:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.le_disease = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.model = RandomForestClassifier(max_depth=10, n_estimators=99, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def getDetailsFromColumn(self,symp_path, disease_name):
        df = pd.read_csv(symp_path)
        disease_name = str(disease_name).strip()
        if not disease_name:
            raise ValueError("Disease name is invalid (NaN or empty).")
        disease_column = df.iloc[:, 0].str.lower().str.strip()
        disease_row = df[disease_column == disease_name.lower()]
        if not disease_row.empty:
            symptoms = disease_row.iloc[:, 1:].values.flatten().tolist()
            return symptoms
        else:
            return None

    def load_data(self):
        self.symptom_description = pd.read_csv('dataset/Symptom-severity.csv')
        self.disease_description = pd.read_csv('dataset/symptom_Description.csv')
        self.symptom_precaution = pd.read_csv('dataset/symptom_precaution.csv')
        self.training_data = pd.read_csv('dataset/dataset.csv')
        print("Disease Description Data:")
        print(self.disease_description.head())
        print("Training Data Columns:", self.training_data.columns)
        print("Symptom Description Columns:", self.symptom_description.columns)
        print("Disease Description Columns:", self.disease_description.columns)
        print("Symptom Precaution Columns:", self.symptom_precaution.columns)
        self.training_data.columns = [col.strip() for col in self.training_data.columns]
        self.symptom_description.columns = [col.strip() for col in self.symptom_description.columns]
        self.disease_description.columns = [col.strip() for col in self.disease_description.columns]
        self.symptom_precaution.columns = [col.strip() for col in self.symptom_precaution.columns]

    def preprocess_symptoms(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        return ' '.join(tokens)
    
    def prepare_data(self):
        symptom_cols = self.training_data.columns[:-1]
        symptoms_list = self.training_data[symptom_cols].fillna('').values.tolist()
        symptoms_list = [[str(sym) for sym in symptoms if sym != '0'] 
                        for symptoms in symptoms_list]
        self.X = self.mlb.fit_transform(symptoms_list)
        self.y = self.le_disease.fit_transform(self.training_data['Disease'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(self.y_test, y_pred, average='weighted')
        }
        return metrics, y_pred
    
    def plot_metrics(self, metrics, file_name='metrics.png'):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Score')
        
        # Save the plot as a PNG file
        plt.savefig(file_name)
        plt.close()

    def predict_disease(self, symptoms: list):
        processed_symptoms = [self.preprocess_symptoms(sym) for sym in symptoms]
        symptoms_binary = self.mlb.transform([processed_symptoms])
        disease_idx = self.model.predict(symptoms_binary)[0]
        disease = self.le_disease.inverse_transform([disease_idx])[0]
        print(f"Predicted Disease: {disease}")
        description = self.disease_description[[ 
            'Disease', 'Description']].loc[self.disease_description['Disease'] == disease, 'Description']
        if description.empty:
            print("No description found for the predicted disease.")
            description = "No description available"
        description = description.iloc[0] if not description.empty else description
        precautions_columns = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        precautions = self.symptom_precaution.loc[self.symptom_precaution['Disease'] == disease, precautions_columns]
        precautions = precautions.values.flatten()
        precautions = [precaution for precaution in precautions if pd.notna(precaution)]
        if not precautions:
            print(f"No precautions found for disease: {disease}")
            precautions = ["No precautions data available"]
        return {
            'disease': disease,
            'description': description,
            'precautions': precautions,
            'confidence': float(max(self.model.predict_proba(symptoms_binary)[0]))
        }

    def getSeverityRate(self,disease):
        data_path = 'dataset/dataset.csv'
        symptoms = self.getDetailsFromColumn(data_path,disease)
        severity_rate = 0
        symp_path = 'dataset/Symptom-severity.csv'
        c=0
        for symptom in symptoms:
            srate = self.getDetailsFromColumn(symp_path,symptom)
            if srate is not None:
                severity_rate += srate[0]
                c+=1
        print("\nSeverity Rate: ","{:.2f}".format(severity_rate/c),"\n")
    
    def get_medicine_recommendations(self,disease):
        data_path = 'dataset/medications.csv'
        meds = self.getDetailsFromColumn(data_path,disease)
        c = 0
        print("Medicines Recommended:")
        meds_list = ast.literal_eval(meds[0])
        for med in meds_list:
            c+=1
            print(c," ",med)

def main():
    system = ClinicalDocumentationSystem()
    print("Loading data...")
    system.load_data()
    system.prepare_data()
    print("Training model...")
    system.train_model()
    print("\nModel Evaluation Metrics:")
    metrics, y_pred = system.evaluate_model()
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plotting evaluation metrics
    system.plot_metrics(metrics)
    
    # Plotting confusion matrix
    # system.plot_confusion_matrix(system.y_test, y_pred)

    print("\nExample Prediction:")
    sample_symptoms = ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever","redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell", "muscle_pain"]
    prediction = system.predict_disease(sample_symptoms)
    print(f"\nPredicted Disease: {prediction['disease']}")
    # print(f"Confidence: {prediction['confidence']:.2f}")
    print(f"\nDescription: {prediction['description']}")
    print("\nPrecautions:")
    for i, precaution in enumerate(prediction['precautions'], 1):
        print(f"{i}. {precaution}")
    system.getSeverityRate(prediction['disease'])
    system.get_medicine_recommendations(prediction['disease'])

if __name__ == "__main__":
    main()
