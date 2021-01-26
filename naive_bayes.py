import sys
import math
import string
import json

def json_reader(path: str):
    f = open(path, "r")
    text = json.load(f)
    f.close()
    return text

def get_document_words(filepath: str):
    f = open(filepath, 'r')
    text = f.read()

    # Design decision: "Subject" is also added into tokens while it appears in each spam and legitimate e-mail."

    # Design decision: I'll change punctuations into " ".
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    
    tokens = text.split()
    
    return tokens

def get_class_probability(email_type: str):
    return 0.5 # since our corpus contains 240 spam, 240 legitimate emails.

def get_word_probabilities(document_words, email_type):
    if email_type == "spam":
        bag_of_words = json_reader("spam_emails_bag_of_words_model.json")
    else:
        bag_of_words = json_reader("legitimate_emails_bag_of_words_model.json")
    
    # this can be improved later. no need to calculate this for each document.
    total_number_of_words = sum(list(bag_of_words.values()))

    word_probability_list = []
    for word in document_words:
        # laplace(add-1) smoothing performed
        word_probability_list.append((int(bag_of_words.get(word, 0))+1) / ((total_number_of_words)+len(document_words)))
    
    return word_probability_list

def get_score(word_probabilities, class_probability):
    scores = []
    log_of_word_probabilities = [math.log(prob,2) for prob in word_probabilities]
    return (math.log(class_probability, 2) + sum(log_of_word_probabilities))

if __name__ == "__main__":
    document_path = sys.argv[1]
    document_words = get_document_words(document_path)

    spam_class_probability = get_class_probability("spam")
    legitimate_class_probability = get_class_probability("legitimate")

    spam_word_probabilities = get_word_probabilities(document_words, "spam")
    legitimate_word_probabilities = get_word_probabilities(document_words, "legitimate")

    spam_score = get_score(spam_word_probabilities, spam_class_probability)
    legitimate_score = get_score(legitimate_word_probabilities, legitimate_class_probability)

    if spam_score < legitimate_score:
        print("document is legitimate")
    else:
        print("document is spam")
