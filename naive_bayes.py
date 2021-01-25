import sys

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
        print("document is spam")
    else:
        print("document is legitimate")
