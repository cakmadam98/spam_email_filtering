'''
Notes: 
- P(ew, ec) == 0 olduğu durumlarda 10^-5 gibi düşük bir değer koyuyorum. Sıkıntı yaratır mı acep
'''

import glob
import json
import string
import math
import sys

# Saves the JSON file with given file name.
def json_saver(filename: str, file: str):
    # Save df in JSON format.
    f = open(filename, "w")
    json.dump(file, f, indent=2)
    f.close()

# Given a file path, reads the file and returns the tokenized version of it.
# Design decision: "Subject" is also added into tokens while it appears in each spam and legitimate e-mail."
def get_tokens(filepath: str):
    f = open(filepath, 'r')
    text = f.read()

    # Design decision: I'll change punctuations into " ".
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    
    tokens = text.split()
    return tokens

# Given list of emails and their type(spam/legitimate), return a document frequencies of words.
def create_document_frequency_dictionary(email_paths, email_type):
    df = dict()
    for filepath in email_paths:
        tokens = get_tokens(filepath)
        unique_tokens = list(set(tokens))
        for token in unique_tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1
    return df

# Given list of emails and their type(spam/legitimate), creates bag of words model.
def create_bag_of_words_model(email_paths, email_type):
    bag_of_words = dict()
    for filepath in email_paths:
        tokens = get_tokens(filepath)
        for token in tokens:
            if token in bag_of_words:
                bag_of_words[token] += 1
            else:
                bag_of_words[token] = 1
    return bag_of_words

# Given email type(spam/legitimate), returns all file paths in the training dataset.
def get_data_paths(email_type: str):
    if email_type == "spam":
        spam_files = glob.glob("./dataset/training/spam/*.txt")
        return spam_files
    else:
        legitimate_files = glob.glob("./dataset/training/legitimate/*.txt")
        return legitimate_files

# Given email type(spam/legitimate), returns all file paths in the test dataset.
def get_data_paths_for_testing(email_type: str):
    if email_type == "spam":
        spam_files = glob.glob("./dataset/test/spam/*.txt")
        return spam_files
    else:
        legitimate_files = glob.glob("./dataset/test/legitimate/*.txt")
        return legitimate_files

# Calculates the mutual information value.
def get_mutual_information_value(n_documents_in_spam_emails, n_documents_in_legit_emails, n_missing_documents_in_spam_emails, n_missing_documents_in_legit_emails, total_number_of_documents):
    if n_documents_in_spam_emails == 0:
        first_box = 0
    else:
        first_box = (n_documents_in_spam_emails / total_number_of_documents) * math.log((n_documents_in_spam_emails / total_number_of_documents) / (((n_documents_in_spam_emails + n_documents_in_legit_emails) / total_number_of_documents)*((n_documents_in_spam_emails + n_missing_documents_in_spam_emails) / total_number_of_documents)), 2)
    
    if n_documents_in_legit_emails == 0:
        second_box = 0
    else:
        second_box = (n_documents_in_legit_emails / total_number_of_documents) * math.log((n_documents_in_legit_emails / total_number_of_documents) / ((n_documents_in_spam_emails + n_documents_in_legit_emails) / total_number_of_documents * (n_documents_in_legit_emails + n_missing_documents_in_legit_emails) / total_number_of_documents), 2)
    
    if n_missing_documents_in_spam_emails == 0:
        third_box = 0
    else:
        third_box = (n_missing_documents_in_spam_emails / total_number_of_documents) * math.log((n_missing_documents_in_spam_emails / total_number_of_documents) / ( (n_missing_documents_in_spam_emails + n_missing_documents_in_legit_emails) / total_number_of_documents * (n_documents_in_spam_emails + n_missing_documents_in_spam_emails) / total_number_of_documents ), 2)
    
    if n_missing_documents_in_legit_emails == 0:
        fourth_box = 0
    else:
        fourth_box = (n_missing_documents_in_legit_emails / total_number_of_documents) * math.log((n_missing_documents_in_legit_emails / total_number_of_documents) / ( (n_missing_documents_in_spam_emails + n_missing_documents_in_legit_emails) / total_number_of_documents * (n_documents_in_legit_emails + n_missing_documents_in_legit_emails) / total_number_of_documents ), 2)
    
    return first_box + second_box + third_box + fourth_box

# Given K, class_type and document frequencies of words, returns top K distinctive words
# K: Number of distinctive words requested.
# class_type: Spam or Legitimate.
# spam_bag_of_words: document frequencies of all words in spam emails.
# legit_bag_of_words: document frequencies of all words in legitimate emails.
def get_distinctive_words(K, class_type, spam_bag_of_words, legit_bag_of_words):

    # Calculates total number of words which will be used later.

    # sanki number of words değil de, number of documents ile ilgilenmemiz gerekiyor?
    # variable isimler çok karışık onlara bi bak lütfen.
    
    number_of_documents_in_spam_class = 240
    number_of_documents_in_legitimate_class = 240
    total_number_of_documents = number_of_documents_in_spam_class + number_of_documents_in_legitimate_class

    if class_type == "spam":

        word_scores_in_spam_emails = dict()
        for word, document_frequency_of_that_word in spam_bag_of_words.items():

            # below variables are used for calculating mutual information value.
            n_documents_in_spam_emails = document_frequency_of_that_word
            n_documents_in_legit_emails = legit_bag_of_words.get(word, 0) # if word does not exist, it will return 0.
            n_missing_documents_in_spam_emails = number_of_documents_in_spam_class - n_documents_in_spam_emails
            n_missing_documents_in_legit_emails = number_of_documents_in_legitimate_class - n_documents_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(n_documents_in_spam_emails, n_documents_in_legit_emails, n_missing_documents_in_spam_emails, n_missing_documents_in_legit_emails, total_number_of_documents)
            
            # Add to the dictionary
            word_scores_in_spam_emails[word] = mutual_information
        
        # Sort dictionary by value
        sorted_scores = sorted(word_scores_in_spam_emails.items(), key = lambda x : x[1], reverse=True)
        
        # Returns words of top K results.
        return [word_score[0] for word_score in sorted_scores[:K]]
    
    else:

        word_scores_in_legit_emails = dict()
        for word, document_frequency_of_that_word in legit_bag_of_words.items():

            # below variables are used for calculating mutual informatin value.
            n_documents_in_spam_emails = spam_bag_of_words.get(word, 0) # if word does not exist, it will return 0.
            n_documents_in_legit_emails = document_frequency_of_that_word
            n_missing_documents_in_spam_emails = number_of_documents_in_spam_class - n_documents_in_spam_emails
            n_missing_documents_in_legit_emails = number_of_documents_in_legitimate_class - n_documents_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(n_documents_in_spam_emails, n_documents_in_legit_emails, n_missing_documents_in_spam_emails, n_missing_documents_in_legit_emails, total_number_of_documents)
            
            # Add to the dictionary
            word_scores_in_legit_emails[word] = mutual_information
        
        # Sort dictionary by value
        sorted_scores = sorted(word_scores_in_legit_emails.items(), key = lambda x : x[1], reverse=True)

        # Returns words of top K results.
        return [word_score[0] for word_score in sorted_scores[:K]]

# Given bag of words model and distinctive words, reduces the vocabulary size of bag of words model.
# Returns the new model at the end.
def get_subset(bag_of_words, distinctive_words):
    subset = dict()
    for distinctive_word in distinctive_words:
        assert bag_of_words.get(distinctive_word) != None # distinctive word should be in the training set.
        subset[distinctive_word] = bag_of_words[distinctive_word]
    return subset

# Main method of Preprocessor
# preprocess_type: "all" or "K".
# If preprocess_type is "all", vocabulary will contain all words.
# If preprocess_type is "K", vocabulary will contain 2*K words.
def preprocess(preprocess_type):

    # Gets paths of the files.
    spam_email_paths = get_data_paths("spam")
    legitimate_email_paths = get_data_paths("legitimate")

    # Creates bag of words models.
    spam_bag_of_words = create_bag_of_words_model(spam_email_paths, "spam")
    legit_bag_of_words = create_bag_of_words_model(legitimate_email_paths, "legitimate")

    # Create a dictionary containing words as key and "number of documents it appears" as values
    # this is needed for calculating mutual information.
    document_frequencies_of_words_in_spam_emails = create_document_frequency_dictionary(spam_email_paths, "spam")
    document_frequencies_of_words_in_legit_emails = create_document_frequency_dictionary(legitimate_email_paths, "legitimate")


    # If top K distinctive words is requested, reduces the vocabulary size of the model.
    if preprocess_type == "K":
        # Choose K distinctive words.
        K = 100
        spam_distinctive_words = get_distinctive_words(K, "spam", document_frequencies_of_words_in_spam_emails, document_frequencies_of_words_in_legit_emails)
        legit_distinctive_words = get_distinctive_words(K, "legitimate", document_frequencies_of_words_in_spam_emails, document_frequencies_of_words_in_legit_emails)

        # Update bag of words models.
        spam_bag_of_words = get_subset(spam_bag_of_words, spam_distinctive_words)
        legit_bag_of_words = get_subset(legit_bag_of_words, legit_distinctive_words)

    # Saves the bag of words models as JSON files.
    json_saver("spam_emails_bag_of_words_model.json", spam_bag_of_words)
    json_saver("legitimate_emails_bag_of_words_model.json", legit_bag_of_words)

if __name__ == "__main__":
    preprocess(sys.argv[1])