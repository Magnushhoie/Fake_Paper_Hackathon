from sklearn.decomposition import PCA
def PCA_transform(embeddings, n_components = 2):
    X = embeddings
    pca = PCA(n_components = n_components)
    pca.fit(X)
    X_PCA = pca.transform(X)
    return(X_PCA)


from sklearn.decomposition import PCA
def PCA_transform_loaded(pca, X):
    X_PCA = pca.transform(X)
    return(X_PCA)

def test_performance(test_x, test_y, verbose = 0):
    #Compute values, try to extract probabilities if possible
    y_pred = np.ones(len(test_x))
    y_true = return_false_true(test_x, test_y)
    #y_true = return_false_true(test_y, test_y)
    
    y_score = y_pred
    y_true_binary = y_true.astype(int)
    
    #Performance scores
    confusion_m =  confusion_matrix(y_true_binary, y_pred)
    try:
        tn, fp, fn, tp = confusion_m.ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, confusion_m[0]
    try: 
        neg_precision = tn / (tn + fn)
    except:
        neg_precision = 0
            
    #Note: Classifying AUC on y_pred and y_score has a large difference in AUC calculation
    # As the threshold becomes 0.5
    #fpr, tpr, thresholds = metrics.roc_curvey_true, y_pred, pos_label=1)
    #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    
    #auc = round(metrics.auc(fpr, tpr), 3)
    fpr, tpr, thresholds = metrics.roc_curve(y_true_binary, y_pred, pos_label=1)
    mcc = round(matthews_corrcoef(y_true_binary, y_pred), 10) 
    precision = precision_score(y_true_binary, y_pred, average='binary')
    #pearson = sc.stats.pearsonr(y_true, y_score)
    #pearson = (np.round(pearson[0], 3), pearson[1])
    

    #print("AUC", auc)
    #print("MCC", mcc)
    #print("Pearson", pearson)
    print("Accuracy:", np.round(accuracy_score(y_true, y_pred), 3))
    print("Correct positives:", tp, "/", tp+fn, "positives")
    print("False positives:", fp, "/", fp+tn, "negatives")
    if len(tpr) <= 2:
        print("TPR:", np.round(tpr, 2))
    else:
        print("TPR", np.round(tpr, 2)[1])            
    print("Positive Precision", round(precision, 3))
    print("Negative precision", round(neg_precision, 3))
    print("Confusion matrix:")
    print("[[tn, fp]]")
    print(confusion_m)    
    print("[[fn tp]]")