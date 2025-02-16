'''
To run the given Python code on your machine, you'll need to have the following prerequisites installed:
1. Python: The code is written in Python, so you'll need to have Python installed on your system. You can download the latest version of Python from the official website python.org.   
2. google.generativeai: This package provides access to the Gemini API. You can install it using pip install google-generativeai.
3. spaCy: This library is used for Natural Language Processing tasks like Named Entity Recognition. You can install it using pip install spacy.
4. spaCy Language Model: You'll need to download the English language model for spaCy. You can do this by running the command python -m spacy download en_core_web_sm.
'''
import spacy
import string
import os
import google.generativeai as genai

# Configure API key
genai.configure(api_key="xxxxxxxxxxxxx")  # Replace with your actual API key

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Maximum limit set to 2GB after testing with mysterybook3

# Helper functions

def clean_word(word):
    '''
    word is a string.
    Return a version of word in which all letters have been
    converted to lowercase, and punctuation characters have
    been stripped from both ends. Inner punctuation is left
    untouched.

    >>> clean_word('Pearl!')
    'pearl'
    >>> clean_word('card-board')
    'card-board'
    '''
    word = word.lower()
    word = word.strip(string.punctuation)
    return word

def average_word_length(text):
    '''
    text is a string of text.
    Return the average word length of the words in text.
    Do not count empty words as words.
    Do not include surrounding punctuation.
    '''
    words = text.split()
    total = 0
    count = 0
    for word in words:
        word = clean_word(word)
        if word!= '':
            total += len(word)
            count += 1
    return total / count

def different_to_total(text):
    '''
    text is a string of text.
    Return the number of unique words in text
    divided by the total number of words in text.
    Do not count empty words as words.
    Do not include surrounding punctuation.
    '''
    words = text.split()
    total = 0
    unique = set()
    for word in words:
        word = clean_word(word)
        if word!= '':
            total += 1
            unique.add(word)
    return len(unique) / total

def exactly_once_to_total(text):
    '''
    text is a string of text.
    Return the number of words that show up exactly once in text
    divided by the total number of words in text.
    Do not count empty words as words.
    Do not include surrounding punctuation.
    '''
    words = text.split()
    total = 0
    unique = set()
    once = set()
    for word in words:
        word = clean_word(word)
        if word!= '':
            total += 1
            if word in unique:
                once.discard(word)
            else:
                unique.add(word)
                once.add(word)
    return len(once) / total

def split_string(text, separators):
    '''
    text is a string of text.
    separators is a string of separator characters.

    Split the text into a list using any of the one-character
    separators and return the result. Remove spaces from
    beginning and end of a string before adding it to the list.
    Do not include empty strings in the list.
    '''
    words = []
    word = ''
    for char in text:
        if char in separators:
            word = word.strip()
            if word!= '':
                words.append(word)
            word = ''
        else:
            word += char
    word = word.strip()
    if word!= '':
        words.append(word)
    return words

def get_sentences(text):
    '''
    text is a string of text.
    Return a list of the sentences from text.
    Sentences are separated by a '.', '?' or '!'.
    '''
    return split_string(text, '.?!')

def average_sentence_length(text):
    '''
    text is a string of text.
    Return the average number of words per sentence in text.
    Do not count empty words as words.
    '''
    sentences = get_sentences(text)
    total = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word!= '':
                total += 1
    return total / len(sentences)

def get_phrases(sentence):
    '''
    sentence is a sentence string.
    Return a list of the phrases from sentence.
    Phrases are separated by a ',', ';' or ':'.
    '''
    return split_string(sentence, ',;:')

def average_sentence_complexity(text):
    '''
    text is a string of text.
    Return the average number of phrases per sentence in text.
    '''
    sentences = get_sentences(text)
    total = 0
    for sentence in sentences:
        phrases = get_phrases(sentence)
        total += len(phrases)
    return total / len(sentences)

def make_signature(text):
    '''
    The signature for text is a list of five elements:
    average word length, different words divided by total words,
    words used exactly once divided by total words,
    average sentence length, and average sentence complexity.
    Return the signature for text.   
    '''
    return [average_word_length(text), different_to_total(text),
            exactly_once_to_total(text),
            average_sentence_length(text),
            average_sentence_complexity(text)]

def get_all_signatures(known_dir):
    '''
    known_dir is the name of a directory of.stat files, where each file
    contains the author's name on the first line and the signature on
    the subsequent lines.

    Return a dictionary mapping author names to their signatures.
    '''
    signatures = {}
    for filename in os.listdir(known_dir):
        with open(os.path.join(known_dir, filename), encoding='utf-8') as f:
            lines = [line.strip() for line in f]  # Read all lines and strip whitespace
        author_name = lines  # First line is the author's name
        signature = [float(value) for value in lines[1:]]  # Convert signature values to floats
        signatures[filename] = signature  # Use filename as key, not author_name
    return signatures

def get_score(signature1, signature2, weights):
    '''
    signature1 and signature2 are signatures.
    weights is a list of five weights.

    Return the score for signature1 and signature2.
    '''
    score = 0
    # print("Unknwn_Signature", signature2)
    for i in range(len(signature1)):
        score += abs(signature1[i] - signature2[i]) * weights[i]
    # print("Score:",score)
    return score

def lowest_score(signatures_dict, unknown_signature, weights):
    '''
    signatures_dict is a dictionary mapping keys to signatures.
    unknown_signature is a signature.
    weights is a list of five weights.

    Return the key whose signature value has the lowest
    score with unknown_signature.
    '''
    lowest = None
    for key in signatures_dict:
        score = get_score(signatures_dict[key], unknown_signature, weights)
        if lowest is None or score < lowest[1]:
            lowest = (key, score)
    return lowest[0]

def process_data(mystery_filename, known_dir):
    '''
    mystery_filename is the filename of a mystery book whose
    author we want to know.
    known_dir is the name of a directory of books.

    Return the name of the author closest to the signature of the
    mystery filename.
    '''
    signatures = get_all_signatures(known_dir)
    with open(mystery_filename, encoding='utf-8') as f:
        text = f.read()
    unknown_signature = make_signature(text)
    author_file = lowest_score(signatures, unknown_signature, [11, 33, 50, 0.4, 4])
    author_name = author_file.replace(".stats", "")  # Remove ".stats" from the filename
    author_name = author_name.replace(".", " ").title()  # Replace dots with spaces and capitalize each word
    return author_name  # Return the formatted author name

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
    """

    prompt = f"I have a set of character/persons from an unknown book as mentioned below, " \
             f"kindly provide a possible author name and book name.\n\n{unique_persons}"

    # Generate text
    response = model.generate_content(prompt)
    print(response.text)

def get_author_description(author_name):
    # Generate text
    response = model.generate_content(f"Tell me about {author_name}")
    return response.text

def make_guess(known_dir):
    '''
    Ask user for a filename.
    Get all known signatures from known_dir,
    and print the name of the one that has the lowest score
    with the user's filename.
    '''
    filename = input('Enter filename: ')
    author_name = process_data(filename, known_dir)
    print ("--------------------------------------------")
    print(f"The author of the unknown book is likely: {author_name}")

    # Send the author name to the AI chatbot and get a description
    description = get_author_description(author_name)
    print ("--------------------------------------------")
    print("Answer from AI ->")
    print(guess_author_and_book(extract_named_entities(filename)))
    print ("--------------------------------------------")
    print(f"Tell me about the author: {description}")

# Main program

make_guess(r"C:/Users/sangr/OneDrive/Desktop/known_authors")

