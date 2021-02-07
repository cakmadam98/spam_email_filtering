'''
Notes: 
- P(ew, ec) == 0 olduğu durumlarda 10^-5 gibi düşük bir değer koyuyorum. Sıkıntı yaratır mı acep
'''

import glob
import json
import string
import math
import sys

def json_saver(filename: str, file: str):
    # Save df in JSON format.
    f = open(filename, "w")
    json.dump(file, f, indent=2)
    f.close()

def get_tokens(filepath: str):
    f = open(filepath, 'r')
    text = f.read()

    # Design decision: "Subject" is also added into tokens while it appears in each spam and legitimate e-mail."

    # Design decision: I'll change punctuations into " ".
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    
    tokens = text.split()
    
    return tokens

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
    json_saver("{}_emails_bag_of_words_model.json".format(email_type), bag_of_words)


def get_data_paths(email_type: str):
    if email_type == "spam":
        spam_files = glob.glob("./dataset/training/spam/*.txt")
        return spam_files
    else:
        legitimate_files = glob.glob("./dataset/training/legitimate/*.txt")
        return legitimate_files

def get_data_paths_for_testing(email_type: str):
    if email_type == "spam":
        spam_files = glob.glob("./dataset/test/spam/*.txt")
        return spam_files
    else:
        legitimate_files = glob.glob("./dataset/test/legitimate/*.txt")
        return legitimate_files

def get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words):
    first_box = (word_frequency_in_spam_emails / total_number_of_words) * math.log((word_frequency_in_spam_emails / total_number_of_words) / (((word_frequency_in_spam_emails + word_frequency_in_legit_emails) / total_number_of_words)*((word_frequency_in_spam_emails + missing_word_frequency_in_spam_emails) / total_number_of_words)))
    second_box = (word_frequency_in_legit_emails / total_number_of_words) * math.log((word_frequency_in_legit_emails / total_number_of_words) / ((word_frequency_in_spam_emails + word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_legit_emails + missing_word_frequency_in_legit_emails) / total_number_of_words))
    third_box = (missing_word_frequency_in_spam_emails / total_number_of_words) * math.log((missing_word_frequency_in_spam_emails / total_number_of_words) / ( (missing_word_frequency_in_spam_emails + missing_word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_spam_emails + missing_word_frequency_in_spam_emails) / total_number_of_words ))
    fourth_box = (missing_word_frequency_in_legit_emails / total_number_of_words) * math.log((missing_word_frequency_in_legit_emails / total_number_of_words) / ( (missing_word_frequency_in_spam_emails + missing_word_frequency_in_legit_emails) / total_number_of_words * (word_frequency_in_legit_emails + missing_word_frequency_in_legit_emails) / total_number_of_words ))
    return first_box + second_box + third_box + fourth_box

def get_distinctive_words(K, class_type, spam_bag_of_words, legit_bag_of_words):
    number_of_words_in_spam_class = sum(list(spam_bag_of_words.values()))
    number_of_words_in_legitimate_class = sum(list(legit_bag_of_words.values()))
    total_number_of_words = number_of_words_in_spam_class + number_of_words_in_legitimate_class

    if class_type == "spam":

        word_scores_in_spam_emails = dict()
        for word, word_frequency in spam_bag_of_words.items():
            word_frequency_in_spam_emails = word_frequency
            word_frequency_in_legit_emails = legit_bag_of_words.get(word, 0.000001) # if word does not exist, it will return very low value.
            missing_word_frequency_in_spam_emails = number_of_words_in_spam_class - word_frequency_in_spam_emails
            missing_word_frequency_in_legit_emails = number_of_words_in_legitimate_class - word_frequency_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words)
            
            # Add to the dictionary
            word_scores_in_spam_emails[word] = mutual_information
        
        # Sort dictionary by value
        sorted_scores = sorted(word_scores_in_spam_emails.items(), key = lambda x : x[1], reverse=True)
        
        return [word_score[0] for word_score in sorted_scores[:K]]
    
    else:

        word_scores_in_legit_emails = dict()
        for word, word_frequency in legit_bag_of_words.items():
            word_frequency_in_spam_emails = spam_bag_of_words.get(word, 0.000001) # if word does not exist, it will return very low value.
            word_frequency_in_legit_emails = word_frequency
            missing_word_frequency_in_spam_emails = number_of_words_in_spam_class - word_frequency_in_spam_emails
            missing_word_frequency_in_legit_emails = number_of_words_in_legitimate_class - word_frequency_in_legit_emails
            
            # Get mutual information
            mutual_information = get_mutual_information_value(word_frequency_in_spam_emails, word_frequency_in_legit_emails, missing_word_frequency_in_spam_emails, missing_word_frequency_in_legit_emails, total_number_of_words)
            
            # Add to the dictionary
            word_scores_in_legit_emails[word] = mutual_information
        
        # Sort dictionary by value
        sorted_scores = sorted(word_scores_in_legit_emails.items(), key = lambda x : x[1], reverse=True)
        
        return [word_score[0] for word_score in sorted_scores[:K]]

def get_subset(bag_of_words, distinctive_words):
    subset = dict()
    for distinctive_word in distinctive_words:
        assert bag_of_words.get(distinctive_word) != None # distinctive word should be in the training set.
        subset[distinctive_word] = bag_of_words[distinctive_word]
    return subset

def preprocess(preprocess_type):
    print("preprocess is started")

    spam_email_paths = get_data_paths("spam")
    legitimate_email_paths = get_data_paths("legitimate")

    spam_bag_of_words = create_bag_of_words_model(spam_email_paths, "spam")
    legit_bag_of_words = create_bag_of_words_model(legitimate_email_paths, "legitimate")

    if preprocess_type == "K":
        # Choose K distinctive words.
        K = 100
        spam_distinctive_words = get_distinctive_words(K, "spam", spam_bag_of_words, legit_bag_of_words)
        legit_distinctive_words = get_distinctive_words(K, "legitimate", spam_bag_of_words, legit_bag_of_words)

        # Update bag of words models.
        spam_bag_of_words = get_subset(spam_bag_of_words, spam_distinctive_words)
        legit_bag_of_words = get_subset(legit_bag_of_words, legit_distinctive_words)

    json_saver("spam_emails_bag_of_words_model.json", spam_bag_of_words)
    json_saver("legitimate_emails_bag_of_words_model.json", legit_bag_of_words)

    print("preprocess is ended")

if __name__ == "__main__":
    preprocess(sys.argv[1])