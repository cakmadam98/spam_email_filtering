import sys
import math
import string
import json

# Reads the JSON file
def json_reader(path: str):
    f = open(path, "r")
    text = json.load(f)
    f.close()
    return text

# Given a file path, reads the file and returns the tokenized version of it.
# Design decision: "Subject" is also added into tokens while it appears in each spam and legitimate e-mail."
def get_document_words(filepath: str):

    f = open(filepath, 'r')
    text = f.read()

    # Design decision: I'll change punctuations into " ".
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    
    tokens = text.split()
    return tokens

def get_class_probability(email_type: str):
    return 0.5 # since our corpus contains 240 spam, 240 legitimate emails.

# For each word in the given document, calculates the probability of choosing that word from all of the words.
def get_word_probabilities(document_words, email_type):

    # corpus is read.
    if email_type == "spam":
        bag_of_words = json_reader("spam_emails_bag_of_words_model.json")
    else:
        bag_of_words = json_reader("legitimate_emails_bag_of_words_model.json")
    
    # total number of words in the corpus is calculated.
    # this can be improved later. no need to calculate this for each document.
    total_number_of_words = sum(list(bag_of_words.values()))

    word_probability_list = []
    for word in document_words:
        # laplace(add-1) smoothing performed
        # (# of occurrance of word W + 1) / (total # of words in corpus + total # of words in the document)
        word_probability_list.append((int(bag_of_words.get(word, 0))+1) / ((total_number_of_words)+len(document_words)))
    
    return word_probability_list

# Score of a document is calculated here by following Naive Bayes Pseudocode.
# Log is used for preventing floating point related errors since very low numbers are used.
def get_score(word_probabilities, class_probability):
    scores = []
    log_of_word_probabilities = [math.log(prob,2) for prob in word_probabilities]
    return (math.log(class_probability, 2) + sum(log_of_word_probabilities))

def main(given_document_path):

    # Reads the document words.
    document_path = given_document_path
    document_words = get_document_words(document_path)

    # Class probabilities are calculated.
    spam_class_probability = get_class_probability("spam")
    legitimate_class_probability = get_class_probability("legitimate")

    # Word probabilities are calculated.
    spam_word_probabilities = get_word_probabilities(document_words, "spam")
    legitimate_word_probabilities = get_word_probabilities(document_words, "legitimate")

    # Scores are calculated by using class and word probabilities.
    spam_score = get_score(spam_word_probabilities, spam_class_probability)
    legitimate_score = get_score(legitimate_word_probabilities, legitimate_class_probability)

    # spam_score and legitimate_score is compared to reach to the conclusion.
    # their scores does not matter. however their comparison matters.
    if spam_score < legitimate_score:
        return "legitimate"
    else:
        return "spam"