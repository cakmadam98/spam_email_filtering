import glob
import json
import string

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
    json_saver("{}_emails_bag_of_words_model.json".format(email_type), bag_of_words)


def get_data_paths(email_type: str):
    if email_type == "spam":
        spam_files = glob.glob("./dataset/training/spam/*.txt")
        return spam_files
    else:
        legitimate_files = glob.glob("./dataset/training/legitimate/*.txt")
        return legitimate_files

def preprocess():
    print("preprocess is started")

    spam_email_paths = get_data_paths("spam")
    legitimate_email_paths = get_data_paths("legitimate")

    create_bag_of_words_model(spam_email_paths, "spam")
    create_bag_of_words_model(legitimate_email_paths, "legitimate")

    print("preprocess is ended")

if __name__ == "__main__":
    preprocess()