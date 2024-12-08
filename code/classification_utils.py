from sklearn import metrics


def calc_metrics(y_trues, y_preds, sigmoid_threshold, save_path=None):
    """
    Calculate confusion matrix and its metrics
    @param y_trues:
    @param y_preds:
    @param sigmoid_threshold:
    @param save_path:
    @return:
    """
    # Calculate metrics
    confusion_matrix = metrics.confusion_matrix(y_trues, y_preds)

    TN, FP, FN, TP = confusion_matrix.ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1 score
    F1 = 2 * PPV * TPR / (PPV + TPR)

    return TNR, TPR, PPV, F1

def calculate_pred(pred, y_val,threshold=0.5):
    
    # accuracy
    print("acuracy:", metrics.accuracy_score(y_val, y_pred=pred))
    # precision score
    print("precision:", metrics.precision_score(y_val, y_pred=pred))
    # recall score
    print("recall", metrics.recall_score(y_val, y_pred=pred))
    # print(metrics.classification_report(y_val, y_pred=pred))

    TNR, TPR, PPV, F1 = calc_metrics(y_val, pred, threshold)
    print("sensitivity ", TPR)
    print("specificity ", TNR)