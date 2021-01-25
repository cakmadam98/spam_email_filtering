
def create_bag_of_words_model(email_paths, email_type):
    pass

def get_data_paths(email_type: str):
    if email_type == "spam":
        return []
    else:
        return []

def preprocess():
    print("preprocess is started")

    spam_email_paths = get_data_paths("spam")
    legitimate_email_paths = get_data_paths("legitimate")

    create_bag_of_words_model(spam_email_paths, "spam")
    create_bag_of_words_model(legitimate_email_paths, "legitimate")

    print("preprocess is ended")

if __name__ == "__main__":
    preprocess()