import subprocess
import eval
import preprocessor
import naive_bayes
import random

# Given file paths of spam and legitimate emails, predicts the class of each file and returns it.
def get_test_predictions(spam_email_paths, legitimate_email_paths):

    spam_predictions = []
    for path in spam_email_paths:
        spam_predictions.append(naive_bayes.main(path))
    
    legitimate_predictions = []
    for path in legitimate_email_paths:
        legitimate_predictions.append(naive_bayes.main(path))

    return spam_predictions, legitimate_predictions

# Given predictions of spam emails and legit emails, calculates the precision and recall for "spam" class.
def get_precision_and_recall_for_spam_class(spam, legitimate):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for predicted_result in spam:
        if predicted_result == "spam":
            true_positives += 1
        else:
            false_negatives += 1
    
    for predicted_result in legitimate:
        if predicted_result == "spam":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

# Given predictions of spam emails and legit emails, calculates the precision and recall for "legitimate" class.
def get_precision_and_recall_for_legitimate_class(spam, legitimate):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for predicted_result in legitimate:
        if predicted_result == "legitimate":
            true_positives += 1
        else:
            false_negatives += 1
    
    for predicted_result in spam:
        if predicted_result == "legitimate":
            false_positives += 1
        else:
            true_negatives += 1
    
    recall = true_positives / ( true_positives + false_negatives )
    precision = true_positives / (true_positives + false_positives)

    return recall, precision

# Calculates F-measure given predictions of spam emails and legit emails.
def calculate_f_measure(spam, legit):

    # Calculate recall and precision values of each class.
    recall1, precision1 = get_precision_and_recall_for_spam_class(spam, legit)
    recall2, precision2 = get_precision_and_recall_for_legitimate_class(spam, legit)

    # Calculate macro-averaged precision and recall values
    precision = (precision1 + precision2) / 2
    recall = (recall1 + recall2) / 2

    # Calculate F-measure
    f_measure = 2 * precision * recall / (precision + recall)

    return f_measure

# Performs approximate randomization test and returns the p-value.
# number of iterations is a design choice(I've used 1000)
def calculate_p_value(f_measure_all, f_measure_K, spam_predictions_all, spam_predictions_K, legitimate_predictions_all, legitimate_predictions_K):
    reference_score = abs(f_measure_all - f_measure_K)
    counter = 0
    number_of_iterations = 1000

    for _ in range(number_of_iterations):

        # initialization
        spam1 = spam_predictions_all
        spam2 = spam_predictions_K
        legit1 = legitimate_predictions_all
        legit2 = legitimate_predictions_K

        # swaps with 50% probability.
        for i in range(len(spam1)):
            swap = random.randint(0, 1)
            if swap == 1:
                temp = spam1[i]
                spam1[i] = spam2[i]
                spam2[i] = temp
        
        # swaps with 50% probability.
        for i in range(len(legit1)):
            swap = random.randint(0, 1)
            if swap == 1:
                temp = legit1[i]
                legit1[i] = legit2[i]
                legit2[i] = temp
        
        # calculate F-measures of each model(with feature selection / without feature selection)
        f1 = calculate_f_measure(spam1, legit1)
        f2 = calculate_f_measure(spam2, legit2)
        
        score = abs(f1 - f2)
        # print(score)
        if score >= reference_score:
            counter += 1
    
    p_value = (counter + 1) / (number_of_iterations + 1)
    print("p_value: {}".format(p_value))
    return p_value


# **** Without Feature Selection ******
# preprocess with all words
preprocessor.preprocess("all")

# evaluates that model
precision_all, recall_all, f_measure_all = eval.get_precision_recall_F_measure()

# gets file paths of each class.
spam_email_paths = preprocessor.get_data_paths_for_testing("spam")
legitimate_email_paths = preprocessor.get_data_paths_for_testing("legitimate")

# stores the predictions
spam_predictions_all, legitimate_predictions_all = get_test_predictions(spam_email_paths, legitimate_email_paths)



# **** With Feature Selection ******
# preprocess with K words
preprocessor.preprocess("K")

# evaluates that model
precision_K, recall_K, f_measure_K = eval.get_precision_recall_F_measure()

# stores the predictions
spam_predictions_K, legitimate_predictions_K = get_test_predictions(spam_email_paths, legitimate_email_paths)



# ***** Approximate Randomization Test ****
# perform approximate_randomization test 
p_value = calculate_p_value(f_measure_all, f_measure_K, spam_predictions_all, spam_predictions_K, legitimate_predictions_all, legitimate_predictions_K)

if p_value < 0.05:
    print("reject the null hypothesis")
else:
    print("do not reject the null hypothesis")