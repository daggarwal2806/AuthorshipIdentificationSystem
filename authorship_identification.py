import string
import os
from typing import List, Dict

def make_guess(known_dir: str) -> None:
    """
    Top-level function to make a guess about the authorship of a mystery text file.
    
    Args:
        known_dir (str): Directory containing known author books.
    """
    filename = input('Enter filename: ')
    print(process_data(filename, known_dir))

def process_data(mystery_filename: str, known_dir: str) -> str:
    """
    Takes in mystery book filename and name of a directory of known-author books as parameters.
    Returns the name of the closest known signature.
    
    Args:
        mystery_filename (str): Filename of the mystery book.
        known_dir (str): Directory containing known author books.
    
    Returns:
        str: Name of the closest known signature.
    """
    signatures = get_all_signatures(known_dir)
    with open(mystery_filename, encoding='utf-8') as file:
        text = file.read()
        unknown_signature = make_signature(text)
    return lowest_score(signatures, unknown_signature, [11, 33, 50, 0.4, 4])

def make_signature(text: str) -> List[float]:
    """
    Figure out the signature of a text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        List[float]: Signature of the text.
    """
    return [
        average_word_length(text),
        different_to_total(text),
        exactly_once_to_total(text),
        average_sentence_length(text),
        average_sentence_complexity(text)
    ]

def get_all_signatures(known_dir: str) -> Dict[str, List[float]]:
    """
    Figure out signatures for all the known author books.
    
    Args:
        known_dir (str): Directory containing known author books.
    
    Returns:
        Dict[str, List[float]]: Dictionary of filenames and their signatures.
    """
    signatures = {}

    for filename in os.listdir(known_dir):
        with open(os.path.join(known_dir, filename), encoding='utf-8', errors='ignore') as file:
            text = file.read()
            signatures[filename] = make_signature(text)
    
    # Note: Consider using multithreading for performance improvement in the future.
    return signatures   
            
    # The following code is commented out because it is not needed for the current implementation.
    # It is kept here for reference. It reads the signature from a file instead of calculating it.
    # This is useful for testing purposes. It also deals with errors that may occur while reading files.
    """
    for filename in os.listdir(known_dir):
    # Skip hidden files
    if filename.startswith('.'):
        continue
    with open(os.path.join(known_dir, filename), encoding='utf-8', errors='ignore') as file:
        try:
            with open(os.path.join(known_dir, filename), encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            print(lines)
            if len(lines) < 6:
                print(f"Skipping {filename}: not enough lines")
                continue
            author_name = lines[0].strip()
            try:
                signature = [float(lines[i].strip()) for i in range(1, 6)]
                signatures[author_name] = signature
            except ValueError as e:
                print(f"Skipping invalid line in {filename}: {e}")
        except (UnicodeDecodeError, ValueError) as e:
            print(f"Error reading {filename}: {e}")
    """

    # This code was used for testing purposes. It reads the signature from a file instead of calculating it.
    """
    lines = file.readlines()
    author_name = lines[0].strip()
    signature = [float(lines[i].strip()) for i in range(1, 6)]
    signatures[author_name] = signature
    """

def lowest_score(signatures_dict: Dict[str, List[float]], unknown_signature: List[float], weights: List[float]) -> str:
    """
    Compare the unknown signature to all the known signatures and return the closest match.
    
    Args:
        signatures_dict (Dict[str, List[float]]): Dictionary of known signatures.
        unknown_signature (List[float]): Signature of the unknown text.
        weights (List[float]): Weights for each feature in the signature.
    
    Returns:
        str: Filename of the closest known signature.
    """
    lowest = None
    for key in signatures_dict:
        score = get_score(signatures_dict[key], unknown_signature, weights)
        if lowest is None or score < lowest[1]:
            lowest = (key, score)
    return lowest[0]

def average_word_length(text: str) -> float:
    """
    Calculate the average word length in the text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        float: Average word length.
    
    >>> average_word_length("Hello world!")
    5.0
    """
    words = text.split()
    total = 0
    count = 0
    for word in words:
        word = clean_word(word)
        if word != '':
            total += len(word)
            count += 1
    return total / count if count != 0 else 0

def different_to_total(text: str) -> float:
    """
    Calculate the ratio of different words to total words in the text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        float: Ratio of different words to total words.
    
    >>> different_to_total("Hello world! Hello everyone.")
    0.75
    """
    words = text.split()
    total = 0
    unique = set()
    for word in words:
        word = clean_word(word)
        if word != '':
            total += 1
            unique.add(word)
    return len(unique) / total if total != 0 else 0

def exactly_once_to_total(text: str) -> float:
    """
    Calculate the ratio of words used exactly once to total words in the text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        float: Ratio of words used exactly once to total words.
    
    >>> exactly_once_to_total("Hello world! Hello everyone.")
    0.5
    """
    words = text.split()
    unique = set()
    once = set()
    total = 0
    for word in words:
        word = clean_word(word)
        if word != '':
            total += 1
            if word in unique:
                once.discard(word)
            else:
                unique.add(word)
                once.add(word)
    return len(once) / total if total != 0 else 0

def average_sentence_length(text: str) -> float:
    """
    Calculate the average number of words per sentence in the text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        float: Average number of words per sentence.
    
    >>> average_sentence_length("Hello world! Hello everyone.")
    2.0
    """
    sentences = get_sentences(text)
    total = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word != '':
                total += 1
    return total / len(sentences) if len(sentences) != 0 else 0

def average_sentence_complexity(text: str) -> float:
    """
    Calculate the average sentence complexity in the text.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        float: Average sentence complexity.
    
    >>> average_sentence_complexity("Hello world! Hello, everyone.")
    1.5
    """
    sentences = get_sentences(text)
    total = 0
    for sentence in sentences:
        phrases = get_phrases(sentence)
        total += len(phrases)
    return total / len(sentences) if len(sentences) != 0 else 0

def clean_word(word: str) -> str:
    """
    Clean a word by removing punctuation and converting to lowercase.
    
    Args:
        word (str): Word to clean.
    
    Returns:
        str: Cleaned word.
    
    >>> clean_word("Hello!")
    'hello'
    """
    word = word.lower()
    word = word.strip(string.punctuation)
    return word

def get_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        List[str]: List of sentences.
    
    >>> get_sentences("Hello world! Hello everyone.")
    ['Hello world', 'Hello everyone']
    """
    return split_string(text, '.?!')

def get_phrases(text: str) -> List[str]:
    """
    Split text into phrases.
    
    Args:
        text (str): Text of the book.
    
    Returns:
        List[str]: List of phrases.
    
    >>> get_phrases("Hello, world! Hello; everyone.")
    ['Hello', ' world', ' Hello', ' everyone']
    """
    return split_string(text, ',;:')

def split_string(text: str, separators: str) -> List[str]:
    """
    Split text by given separators.
    
    Args:
        text (str): Text to split.
        separators (str): String of separator characters.
    
    Returns:
        List[str]: List of split parts.
    
    >>> split_string("Hello, world! Hello; everyone.", ',;:')
    ['Hello', ' world', ' Hello', ' everyone']
    """
    words = []
    word = ''
    for char in text:
        if char in separators:
            word = word.strip()
            if word != '':
                words.append(word)
            word = ''
        else:
            word += char
    word = word.strip()
    if word != '':
        words.append(word)
    return words

def get_score(signature1: List[float], signature2: List[float], weights: List[float]) -> float:
    """
    Calculate the score between two signatures.
    
    Args:
        signature1 (List[float]): First signature.
        signature2 (List[float]): Second signature.
        weights (List[float]): Weights for each feature in the signature.
    
    Returns:
        float: Score between the two signatures.
    
    >>> get_score([1, 2, 3], [1, 2, 3], [1, 1, 1])
    0.0
    """
    score = 0
    for i in range(len(signature1)):
        score += abs(signature1[i] - signature2[i]) * weights[i]
    return score
    
def process_data(mystery_filename: str, known_dir: str) -> str:
    """
    Takes in mystery book filename and name of a directory of known-author books as parameters.
    Returns the name of the closest known signature.
    
    Args:
        mystery_filename (str): Filename of the mystery book.
        known_dir (str): Directory containing known author books.
    
    Returns:
        str: Name of the closest known signature.
    """
    signatures = get_all_signatures(known_dir)
    with open(mystery_filename, encoding='utf-8') as file:
        text = file.read()
        unknown_signature = make_signature(text)
    return lowest_score(signatures, unknown_signature, [11, 33, 50, 0.4, 4])

def make_guess(known_dir: str) -> None:
    """
    Top-level function to make a guess about the authorship of a mystery text file.
    
    Args:
        known_dir (str): Directory containing known author books.
    """
    filename = input('Enter filename: ')
    print(process_data(filename, known_dir))

if __name__ == "__main__":
    # Call the main function with the directory of known authors
    make_guess('known_authors')

    # Vishal Testing
