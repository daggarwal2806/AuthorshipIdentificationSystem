'''
To run the given Python code on your machine, you'll need to have the following prerequisites installed:
1. Python: The code is written in Python, so you'll need to have Python installed on your system. You can download the latest version of Python from the official website python.org.   
2. google.generativeai: This package provides access to the Gemini API. You can install it using pip install google-generativeai.
3. spaCy: This library is used for Natural Language Processing tasks like Named Entity Recognition. You can install it using pip install spacy.
4. spaCy Language Model: You'll need to download the English language model for spaCy. You can do this by running the command python -m spacy download en_core_web_sm.
5. pandas: This library is used for data manipulation and analysis. You can install it using pip install pandas.
6. requests: This library is used for making HTTP requests. You can install it using pip install requests.
7. numpy: This library is used for numerical operations. You can install it using pip install numpy.
8. BeautifulSoup: This library is used for web scraping. You can install it using pip install beautifulsoup4.
9. scikit-learn: This library is used for machine learning tasks. You can install it using pip install scikit-learn.
10. torch: This library is used for deep learning tasks. You can install it using pip install torch.
11. transformers: This library is used for natural language processing tasks. You can install it using pip install transformers.
12. tqdm: This library is used for progress bars. You can install it using pip install tqdm.
13. shutil: This library is used for file operations. It is included in the Python standard library.
14. google.colab: This library is used for Google Colab operations. You can install it using pip install google-colab.
15. concurrent.futures: This library is used for parallel processing. It is included in the Python standard library.
'''
import logging
import spacy
import string
import os
import google.generativeai as genai
import os
import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import shutil
from google.colab import files
import concurrent.futures

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
CHUNK_SIZE = 256
OVERLAP = 50
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# Configure API key
'''
The API key provided here is owned by Sangram Gaikwad and is provided here for temporary use. 
The key will be deleted after grading of this assignement. 
Users are encouarged to use their own key if this key does not work. You can get your API key for Gemini from - https://aistudio.google.com/apikey
'''
genai.configure(api_key="AIzaSyAj7fmnT9sGYRQojR6grU_48AJ95cc6_Zc") 

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Maximum limit set to 2GB after testing with mysterybook3

def extract_named_entities(text_file):
    """
    Extracts named entities from the given text file using spaCy,
    categorizes them by label, and returns unique PERSON entities.

    Args:
        text_file: The path to the text file.

    Returns:
        A set containing PERSON entities.
    """

    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    doc = nlp(text)
    entities_by_label = {}
    unique_persons = set()
    for ent in doc.ents:
        label = ent.label_
        if label not in entities_by_label:
            entities_by_label[label] = []
        if label == "PERSON":
            if ent.text not in unique_persons:
                entities_by_label[label].append(ent.text)
                unique_persons.add(ent.text)
        else:
            entities_by_label[label].append(ent.text)
    return unique_persons

def guess_author_and_book(unique_persons):
    """
    Sends the set of unique character/persons to Gemini with a prompt
    to guess the author and book name.

    Args:
        unique_persons: A set of unique character/person names.

    Returns:
        The generated text response from the model.
    """

    prompt = f"I have a set of character/persons from an unknown book as mentioned below, " \
             f"kindly provide a possible author name and book name.\n\n{unique_persons}"

    # Generate text
    response = model.generate_content(prompt)
    return response.text

def get_author_description(author_name):
    # Generate text
    response = model.generate_content(f"Tell me about the author and book {author_name}")
    return response.text

def make_ai_prediction():
    """
    Prompts the user for a filename, extracts named entities from the file,
    sends the entities to the AI model to guess the author and book name,
    and prints the response. Additionally, it sends the response to the AI
    model to get more information about the author and book.

    This function performs the following steps:
    1. Prompts the user to enter a filename.
    2. Extracts named entities from the specified file.
    3. Sends the extracted entities to the AI model to guess the author and book name.
    4. Prints the AI model's response.
    5. Sends the AI model's response to get more information about the author and book.
    6. Prints the additional information about the author and book.
    """
    # Prompt the user to enter a filename
    filename = input('Enter filename: ')

    # Extract named entities and get the AI model's response
    response = guess_author_and_book(extract_named_entities(filename))

    # Print the AI model's response
    print ("--------------------------------------------")
    print("Answer from AI ->")
    print(response)
    print ("--------------------------------------------")
    
    # Send the AI model's response to get more information about the author and book
    description = get_author_description(response)
    print(f"Tell me about the author and book: {description}")
    print ("--------------------------------------------")

class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts.tolist()
        self.labels = torch.tensor(labels.values)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }

# only removes funny tokens for English texts
def remove_funny_tokens(text):
    tokens = text.split()
    sample = ' '.join(' '.join(tokens).replace('xe2x80x9c', ' ').replace('xe2x80x9d', ' ')\
                                      .replace('xe2x80x94', ' ').replace('xe2x80x99', "'")\
                                      .replace('xe2x80x98', "'").split())
    return sample

# clean newlines, carriage returns and tabs
def clean_text(text):
    cleaned_listed_text = []
    listed_text = list(text)

    for iter in range(len(listed_text) - 1):
        if (listed_text[iter] == '\\' and listed_text[iter + 1] == 'n') or \
            (listed_text[iter] == 'n' and listed_text[iter - 1] == '\\'):
            continue
        elif listed_text[iter] == '\\' and listed_text[iter + 1] == 'r' or \
            (listed_text[iter] == 'r' and listed_text[iter - 1] == '\\'):
            continue
        elif listed_text[iter] == '\\' and listed_text[iter + 1] == 't' or \
            (listed_text[iter] == 't' and listed_text[iter - 1] == '\\'):
            continue
        elif listed_text[iter] == '\\':
            continue
        else:
            cleaned_listed_text.append(listed_text[iter])

    cleaned_text = ''.join([str(char) for char in cleaned_listed_text])
    cleaned_text = remove_funny_tokens(cleaned_text)

    return ''.join(cleaned_text)

# function to strip headers from the text
def strip_headers(text):
    lines = text.split('\n')
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if '*** START OF THIS PROJECT GUTENBERG EBOOK' in line:
            start = i + 1
        elif '*** END OF THIS PROJECT GUTENBERG EBOOK' in line:
            end = i
            break
    return '\n'.join(lines[start:end])

# Uses this script from Kaggle (edited further) to fetch text form links and clean it
def process_links(row):
  combined_text = ""
  for link in row['Link']:
      try:
          page = requests.get(link)
          soup = BeautifulSoup(page.content, 'html.parser')
          text_link_element = soup.find_all("a", string="Plain Text UTF-8")
          if text_link_element:
              text_link = 'http://www.gutenberg.org' + text_link_element[0]['href']
              response = requests.get(text_link)
              response.raise_for_status()
              text = strip_headers(response.text)
              text = ' '.join(' '.join(' '.join(text.split('\n')).split('\t')).split('\r'))
              text = ' '.join(text.split())
              text = clean_text(str(text))
          else:
              print(f"Couldn't find 'Plain Text UTF-8' link for {row['Author']}. Link: {row['Link']}")
              text = None
      except Exception as e:
          print(f"Couldn't acquire text for {row['Author']}. Link: {row['Link']}. Error: {e}")
          text = None
      if text == None:
          continue
      combined_text += text
  if combined_text == "":
    return None, None # return to skip adding if no text
  return row['Author'], combined_text # return to add to data if there is text

# Process the author text data in parallel using threads
# Creates a csv file with Author and Text columns
def parallel_process_data(df_author_link):
  # Use a thread pool for parallel processing
  with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
      futures = []
      for key, row in df_author_link.iterrows():
          futures.append(executor.submit(process_author, row))

      data = {'Author': [], 'Text': []}
      for future in concurrent.futures.as_completed(futures):
          author, combined_text = future.result()  # Get the result from each thread
          if author and combined_text:
              data['Author'].append(author)
              print(author)
              data['Text'].append(combined_text)


  df_data = pd.DataFrame(data, columns=['Author', 'Text'])
  df_data.to_csv(f'/content/author_text.csv', index=False)
  return df_data

# Loads metadata from gutenberg-corpus, takes only authors with more than 5 books and processes the data
# Returns a dataframe with Author and Text columns containing the processed data
def load_and_preprocess_data(metadata_file):
    df_metadata = pd.read_csv(metadata_file)
    df_author_link = df_metadata[['Author', 'Link']]

    df_author_link = df_author_link.groupby('Author')['Link'].agg(list).reset_index()
    df_author_link['Link'] = df_author_link['Link'].apply(lambda x: list(set(x)))
    df_author_link = df_author_link[df_author_link['Link'].apply(len) >= 5]

    df_data = parallel_process_data(df_author_link)

# Split the text into chunks of fixed size
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Create a dataset class for authorship identification
def create_dataset(texts, labels, tokenizer):
    return AuthorshipDataset(texts, labels, tokenizer)

# Train the model using the given dataset
def train_model(train_dataset, test_dataset, num_classes, device):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    return model

# Evaluate the model on the test dataset
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Accuracy
    return accuracy_score(all_labels, all_preds)

# Save the model and tokenizer
def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    shutil.make_archive("bert_authorship_model", 'zip', "bert_authorship_model")
    files.download("bert_authorship_model.zip")

# Load the model and tokenizer and predict the author of a new text
def predict_author_with_downloaded_model(model_path, author_mapping_reversed, device):
    # model_path = "bert_authorship_model"  # Update this if needed
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    print("Model loaded successfully!")

    new_text = """
    It was a bright cold day in April, and the clocks were striking thirteen.
    Winston Smith, his chin nuzzled into his breast in an effort to escape
    the vile wind, slipped quickly through the glass doors of Victory Mansions.
    """

    # Tokenize the input text
    inputs = tokenizer(new_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_author = author_mapping_reversed[predicted_class]
    return predicted_author

# Load the model and tokenizer and predict the author of a new text
def predict_author_with_loaded_model(model, tokenizer, new_text, author_mapping_reversed, device):
    # Tokenize the input text
    inputs = tokenizer(new_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_author = author_mapping_reversed[predicted_class]
    return predicted_author

# Data manipulation
def prepare_data(df):
    df["text_chunks"] = df["Text"].apply(lambda x: split_text_into_chunks(str(x)))
    df = df.explode("text_chunks").reset_index(drop=True)
    df = df.drop(columns=['Text'])
    return df

# Encode labels
def encode_labels(df):
    label_encoder = LabelEncoder()
    df["author_id"] = label_encoder.fit_transform(df["Author"])
    author_mapping = dict(zip(df["Author"], df["author_id"]))
    df = df.drop(columns=['Author'])
    return df, author_mapping

# Train and evaluate the model
def train_and_evaluate(df, tokenizer):
    train_texts, test_texts, train_labels, test_labels = train_test_split(df["text_chunks"], df["author_id"], test_size=0.2, random_state=42)
    train_dataset = create_dataset(train_texts, train_labels, tokenizer)
    test_dataset = create_dataset(test_texts, test_labels, tokenizer)
    num_classes = len(set(train_labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(train_dataset, test_dataset, num_classes, device)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    accuracy = evaluate_model(model, test_dataloader, device)
    return model, accuracy, device

# Make a prediction using the trained model
def make_prediction(model, tokenizer, author_mapping, device, new_text):
    author_mappings_reversed = {value: key for key, value in author_mapping.items()}
    predicted_author = predict_author_with_loaded_model(model, tokenizer, new_text, author_mappings_reversed, device)
    return predicted_author

def make_ml_prediction_after_training_model():
    # Load and preprocess data
    df_data = load_and_preprocess_data("gutenberg_metadata.csv")

    # Load the processed data
    df = pd.read_csv("author_text.csv")

    # Prepare data for tokenization
    df = prepare_data(df)

    # Encode labels
    df, author_mapping = encode_labels(df)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Training and evaluation
    model, accuracy, device = train_and_evaluate(df, tokenizer)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model
    save_path = "bert_authorship_model"
    save_model(model, tokenizer, save_path)

    # Prediction
    new_text = "Your new text here"
    predicted_author = make_prediction(model, tokenizer, author_mapping, device, new_text)
    print(f"Predicted author: {predicted_author}")

# Function to make a prediction using the ml model
def make_ml_prediction():

    option = input("\n\nDo you have the pre-trained model?\n\nChoose an option:\nYES: Have the model_path ready (NOT AVAILABLE)\nNO: Wait for the model to train\n\nEnter YES or NO: ").strip().upper()
    while True:
        if option == 'YES':
            model_path = input("Enter the path to the pre-trained model: ")
            # Need to have author_mapping stored based on the trained model for this to work
            # Get author_mapping from the model_path folder if it is stored there
            # author_mapping.txt contains lines containg {id, author_name} in each line
            author_mapping = {}
            with open("author_mapping.txt", "r") as file:
                for line in file:
                    key, value = line.strip().split(", ")
                    author_mapping[int(key)] = value
            author_mappings_reversed = {value: key for key, value in author_mapping.items()}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predicted_author = predict_author_with_downloaded_model(model_path, author_mappings_reversed, device)
            print(f"Predicted author: {predicted_author}")
            break
        elif option == 'NO':
            make_ml_prediction_after_training_model()
            break
    

def main():
    while True:
        option = input("\n\nAUTHORSHIP IDENTIFICATION SYSTEM\n\nChoose an option:\nA: Identify Author using Gemini AI\nB: Identify Author using pre-trained ML model on Famous Authors\n\nEnter A or B: ").strip().upper()
        if option == 'A':
            make_ai_prediction()
            break
        elif option == 'B':
            make_ml_prediction()
            break
        else:
            print("Invalid option. Please enter A or B.")

if __name__ == "__main__":
    main()

# Word2Vec implementation
'''
This following code was used for Random Forest, SVM and Logistic Regression models implementation after getting the processed data
This processed data had Word2Vec embeddings for the text data

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def get_word2vec_features(text):
    # 1. Clean the text
    # Cleaned already

    # 2. Tokenize the text
    tokens = word_tokenize(text.lower())

    # 3. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]

    # 4. Train a Word2Vec model (or load a pre-trained one)
    # For demonstration, we train a simple model here
    if len(tokens) < 2:  # Need at least 2 words to train Word2Vec
        return np.zeros(100) # Return a zero vector if not enough words
    model = Word2Vec([tokens], min_count=1, vector_size=100, window=5)  # Adjust parameters as needed

    # 5. Generate word vectors
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    # 6. Aggregate word vectors (e.g., average)
    if word_vectors:
      feature_vector = np.mean(word_vectors, axis=0)
    else:
      feature_vector = np.zeros(100)

    return feature_vector

# Load the dataframe
df_data = pd.read_csv('/content/data_features.csv')

# Convert the string representation of the list to a list of floats
df_data['Word2Vec_Features'] = df_data['Word2Vec_Features'].apply(lambda x: [float(val) for val in x[1:-1].split() if val])

# Handle rows with empty Word2Vec_Features lists
df_data = df_data[df_data['Word2Vec_Features'].apply(lambda x: len(x) > 0)]

# Fill NaN values with 0
df_data['Word2Vec_Features'] = df_data['Word2Vec_Features'].apply(lambda x: [0.0 if pd.isna(val) else val for val in x])

# Pad or truncate the lists to a uniform length
max_len = max(len(x) for x in df_data['Word2Vec_Features'])
df_data['Word2Vec_Features'] = df_data['Word2Vec_Features'].apply(lambda x: x + [0.0] * (max_len - len(x)) if len(x) < max_len else x[:max_len])

#Prepare the data for training
X = pd.DataFrame(df_data['Word2Vec_Features'].values.tolist())
y = df_data['Author']

# Split data into training and testing sets
X_train, y_train = X, y

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter to ensure convergence
lr_model.fit(X_train, y_train)

# Save the models
with open('/content/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('/content/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('/content/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Load the saved models
with open('/content/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('/content/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('/content/logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load the test data
df_test = pd.read_csv('/content/author_word2vec_features.csv')

# Preprocess the test data (similar to the training data preprocessing)
df_test['Word2Vec_Features'] = df_test['Word2Vec_Features'].apply(lambda x: [float(val) for val in x[1:-1].split() if val])
df_test = df_test[df_test['Word2Vec_Features'].apply(lambda x: len(x) > 0)]
df_test['Word2Vec_Features'] = df_test['Word2Vec_Features'].apply(lambda x: [0.0 if pd.isna(val) else val for val in x])

# max_len = 100 # Use the same max_len as used for training
df_test['Word2Vec_Features'] = df_test['Word2Vec_Features'].apply(lambda x: x + [0.0] * (max_len - len(x)) if len(x) < max_len else x[:max_len])

X_test = pd.DataFrame(df_test['Word2Vec_Features'].values.tolist())
y_test = df_test['Author']

# Ensure the shapes are compatible
if X_test.shape[1] != max_len:
    X_test = X_test.iloc[:,:max_len] # Or pad if necessary

# Make predictions
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"SVM Accuracy: {svm_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}")

'''

# LSTM model implementation
'''
# This following code was used for LSTM model implementation after getting the processed data

# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from nltk.tokenize import sent_tokenize, word_tokenize
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# import gensim.downloader as api
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import accuracy_score

# # Load the dataset (assuming CSV format with 'author' and 'text' columns)
# df = pd.read_csv("author_text.csv")

# # Convert text to lowercase and tokenize
# df['Text'] = df['Text'].apply(lambda x: word_tokenize(x.lower()))

# # Encode authors into numeric labels
# label_encoder = LabelEncoder()
# df['Author'] = label_encoder.fit_transform(df['Author'])

# # Split text into smaller chunks (e.g., 200-word sequences)
# def split_text(text, chunk_size=200):
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# df['text_chunks'] = df['Text'].apply(lambda x: split_text(x))

# # Explode the dataframe so each chunk is a separate row
# df = df.explode('text_chunks').dropna().reset_index(drop=True)

# # Train-test split (80-20)
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df['text_chunks'], df['Author'], test_size=0.2, random_state=42, stratify=df['Author'])

# # Convert tokenized words back to space-separated text
# train_texts = [" ".join(words) for words in train_texts]
# test_texts = [" ".join(words) for words in test_texts]

# # Load GloVe embeddings (100-dimensional)
# glove_model = api.load("glove-wiki-gigaword-100")

# # Convert words to indices
# vocab = glove_model.key_to_index
# embedding_matrix = np.zeros((len(vocab) + 1, 100))  # Extra row for unknown words

# for word, idx in vocab.items():
#     embedding_matrix[idx] = glove_model[word]

# # Function to convert text into sequences of word indices
# def text_to_sequence(text, max_length=200):
#     tokens = word_tokenize(text.lower())
#     sequence = [vocab[word] if word in vocab else len(vocab) for word in tokens[:max_length]]
#     sequence += [0] * (max_length - len(sequence))  # Padding
#     return torch.tensor(sequence, dtype=torch.long)

# # Convert dataset into tensors
# train_sequences = torch.stack([text_to_sequence(text) for text in train_texts])
# test_sequences = torch.stack([text_to_sequence(text) for text in test_texts])

# # Convert labels to tensors
# train_labels = torch.tensor(train_labels.values, dtype=torch.long)
# test_labels = torch.tensor(test_labels.values, dtype=torch.long)

# class AuthorLSTM(nn.Module):
#     def __init__(self, embedding_dim=100, hidden_dim=256, num_layers=2, num_classes=1000, vocab_size=len(vocab)+1):
#         super(AuthorLSTM, self).__init__()
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
#                             num_layers=num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional LSTM
#         self.dropout = nn.Dropout(0.5)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         lstm_out = self.dropout(lstm_out[:, -1, :])  # Take the last LSTM output
#         output = self.fc(lstm_out)
#         return self.softmax(output)

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AuthorLSTM(num_classes=len(label_encoder.classes_)).to(device)

# # Create Datasets and DataLoaders
# train_dataset = TensorDataset(train_sequences.to(device), train_labels.to(device))
# test_dataset = TensorDataset(test_sequences.to(device), test_labels.to(device))

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for texts, labels in train_loader:
#         texts, labels = texts.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(texts)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# # Evaluation
# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for texts, labels in test_loader:
#         texts, labels = texts.to(device), labels.to(device)
#         outputs = model(texts)
#         _, predicted = torch.max(outputs, 1)
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# accuracy = accuracy_score(all_labels, all_preds)
# print(f"Test Accuracy: {accuracy:.4f}")
# '''
