import preprocessor
import naive_bayes

# Add the feature selection part later.

def get_f_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)

# Given spam and legitimate file paths, calculates the precision and recall for "spam" class.
def get_precision_and_recall_for_spam_class(spam_email_paths, legitimate_email_paths):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for path in spam_email_paths:

        # predicted class of the email is stored in "predicted_result"
        predicted_result = naive_bayes.main(path)

        if predicted_result == "spam":
            true_positives += 1
        else:
            false_negatives += 1
    
    for path in legitimate_email_paths:

        # predicted class of the email is stored in "predicted_result"
        predicted_result = naive_bayes.main(path)
        
        if predicted_result == "spam":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

# Given spam and legitimate file paths, calculates the precision and recall for "legitimate" class.
def get_precision_and_recall_for_legitimate_class(spam_email_paths, legitimate_email_paths):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for path in legitimate_email_paths:

        # predicted class of the email is stored in "predicted_result"
        predicted_result = naive_bayes.main(path)

        if predicted_result == "legitimate":
            true_positives += 1
        else:
            false_negatives += 1
    
    for path in spam_email_paths:

        # predicted class of the email is stored in "predicted_result"
        predicted_result = naive_bayes.main(path)

        if predicted_result == "legitimate":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

# Main method of the evaluation part.
def get_precision_recall_F_measure():

    # Gets file paths
    spam_email_paths = preprocessor.get_data_paths_for_testing("spam")
    legitimate_email_paths = preprocessor.get_data_paths_for_testing("legitimate")

    # Calculates precision and recall for each class.
    precision1, recall1 = get_precision_and_recall_for_spam_class(spam_email_paths, legitimate_email_paths)
    precision2, recall2 = get_precision_and_recall_for_legitimate_class(spam_email_paths, legitimate_email_paths)

    # calculate F-measure for each class
    f1 = get_f_measure(precision1, recall1)
    f2 = get_f_measure(precision2, recall2)

    # Calculates macro-averaged precision and recall values
    precision = (precision1 + precision2) / 2
    recall = (recall1 + recall2) / 2

    # Calculates F-measure. 
    f_measure = get_f_measure(precision, recall)

    return precision, recall, f_measure

# Runs the code below if the file is called directly from terminal.
if __name__ == "__main__":
    precision, recall, f_measure = get_precision_recall_F_measure()
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f_measure: {}".format(f_measure))