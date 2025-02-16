'''
To run the given Python code on your machine, you'll need to have the following prerequisites installed:
1. Python: The code is written in Python, so you'll need to have Python installed on your system. You can download the latest version of Python from the official website python.org.   
2. google.generativeai: This package provides access to the Gemini API. You can install it using pip install google-generativeai.
3. spaCy: This library is used for Natural Language Processing tasks like Named Entity Recognition. You can install it using pip install spacy.
4. spaCy Language Model: You'll need to download the English language model for spaCy. You can do this by running the command python -m spacy download en_core_web_sm.
'''
import logging
import spacy
import string
import os
import google.generativeai as genai

# Initialize logging
logging.basicConfig(level=logging.INFO)

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

def make_guess():
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

# Main program

make_guess()
