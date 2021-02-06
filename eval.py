import preprocessor
import naive_bayes

# Add the feature selection part later.

def get_f_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)

def get_precision_and_recall_for_spam_class():
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    spam_email_paths = preprocessor.get_data_paths("spam")
    legitimate_email_paths = preprocessor.get_data_paths("legitimate")

    for path in spam_email_paths:
        predicted_result = naive_bayes.main(path)
        if predicted_result == "spam":
            true_positives += 1
        else:
            false_negatives += 1
    
    for path in legitimate_email_paths:
        predicted_result = naive_bayes.main(path)
        if predicted_result == "spam":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

def get_precision_and_recall_for_legitimate_class():
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    spam_email_paths = preprocessor.get_data_paths("spam")
    legitimate_email_paths = preprocessor.get_data_paths("legitimate")

    for path in legitimate_email_paths:
        predicted_result = naive_bayes.main(path)
        if predicted_result == "legitimate":
            true_positives += 1
        else:
            false_negatives += 1
    
    for path in spam_email_paths:
        predicted_result = naive_bayes.main(path)
        if predicted_result == "legitimate":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

    

precision1, recall1 = get_precision_and_recall_for_spam_class()
precision2, recall2 = get_precision_and_recall_for_legitimate_class()

# macro-averaged precision and recall values
precision = (precision1 + precision2) / 2
recall = (recall1 + recall2) / 2

f_measure = get_f_measure(precision, recall)

print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("f_measure: {}".format(f_measure))