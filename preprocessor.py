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

# Given list of emails and their type(spam/legitimate), creates bag of words model.
# Also creates a JSON file under the current directory.
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
def get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words):
    if word_frequency_in_spam_emails == 0:
        first_box = 0
    else:
        first_box = (word_frequency_in_spam_emails / total_number_of_words) * math.log((word_frequency_in_spam_emails / total_number_of_words) / (((word_frequency_in_spam_emails + word_frequency_in_legit_emails) / total_number_of_words)*((word_frequency_in_spam_emails + missing_word_frequency_in_spam_emails) / total_number_of_words)))
    
    if word_frequency_in_legit_emails == 0:
        second_box = 0
    else:
        second_box = (word_frequency_in_legit_emails / total_number_of_words) * math.log((word_frequency_in_legit_emails / total_number_of_words) / ((word_frequency_in_spam_emails + word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_legit_emails + missing_word_frequency_in_legit_emails) / total_number_of_words))
    
    if missing_word_frequency_in_spam_emails == 0:
        third_box = 0
    else:
        third_box = (missing_word_frequency_in_spam_emails / total_number_of_words) * math.log((missing_word_frequency_in_spam_emails / total_number_of_words) / ( (missing_word_frequency_in_spam_emails + missing_word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_spam_emails + missing_word_frequency_in_spam_emails) / total_number_of_words ))
    
    if missing_word_frequency_in_legit_emails == 0:
        fourth_box = 0
    else:
        fourth_box = (missing_word_frequency_in_legit_emails / total_number_of_words) * math.log((missing_word_frequency_in_legit_emails / total_number_of_words) / ( (missing_word_frequency_in_spam_emails + missing_word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_legit_emails + missing_word_frequency_in_legit_emails) / total_number_of_words ))
    
    return first_box + second_box + third_box + fourth_box

# Given K, class_type and bag of words models, returns top K distinctive words
# K: Number of distinctive words requested.
# class_type: Spam or Legitimate.
def get_distinctive_words(K, class_type, spam_bag_of_words, legit_bag_of_words):

    # Calculates total number of words which will be used later.
    number_of_words_in_spam_class = sum(list(spam_bag_of_words.values()))
    number_of_words_in_legitimate_class = sum(list(legit_bag_of_words.values()))
    total_number_of_words = number_of_words_in_spam_class + number_of_words_in_legitimate_class

    if class_type == "spam":

        word_scores_in_spam_emails = dict()
        for word, word_frequency in spam_bag_of_words.items():

            # below variables are used for calculating mutual informatin value.
            word_frequency_in_spam_emails = word_frequency
            word_frequency_in_legit_emails = legit_bag_of_words.get(word, 0) # if word does not exist, it will return very low value.
            missing_word_frequency_in_spam_emails = number_of_words_in_spam_class - word_frequency_in_spam_emails
            missing_word_frequency_in_legit_emails = number_of_words_in_legitimate_class - word_frequency_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words)
            
            # Add to the dictionary
            word_scores_in_spam_emails[word] = mutual_information
        
        # Sort dictionary by value
        sorted_scores = sorted(word_scores_in_spam_emails.items(), key = lambda x : x[1], reverse=True)
        
        # Returns words of top K results.
        return [word_score[0] for word_score in sorted_scores[:K]]
    
    else:

        word_scores_in_legit_emails = dict()
        for word, word_frequency in legit_bag_of_words.items():

            # below variables are used for calculating mutual informatin value.
            word_frequency_in_spam_emails = spam_bag_of_words.get(word, 0) # if word does not exist, it will return very low value.
            word_frequency_in_legit_emails = word_frequency
            missing_word_frequency_in_spam_emails = number_of_words_in_spam_class - word_frequency_in_spam_emails
            missing_word_frequency_in_legit_emails = number_of_words_in_legitimate_class - word_frequency_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words)
            
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

    # If top K distinctive words is requested, reduces the vocabulary size of the model.
    if preprocess_type == "K":
        # Choose K distinctive words.
        K = 100
        spam_distinctive_words = get_distinctive_words(K, "spam", spam_bag_of_words, legit_bag_of_words)
        legit_distinctive_words = get_distinctive_words(K, "legitimate", spam_bag_of_words, legit_bag_of_words)

        # Update bag of words models.
        spam_bag_of_words = get_subset(spam_bag_of_words, spam_distinctive_words)
        legit_bag_of_words = get_subset(legit_bag_of_words, legit_distinctive_words)

    # Saves the bag of words models as JSON files.
    json_saver("spam_emails_bag_of_words_model.json", spam_bag_of_words)
    json_saver("legitimate_emails_bag_of_words_model.json", legit_bag_of_words)

if __name__ == "__main__":
    preprocess(sys.argv[1])