# util.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def plot_confusion_matrix(crosstab):
    """
    Plot the confusion matrix calculated using pd.crosstab.
    If save_path is provided, save the plot as an image, otherwise plot to the output.
    
    Args:
        crosstab (pd.DataFrame): The confusion matrix calculated using pd.crosstab.
        save_path (str or None): Path to save the image. If None, the plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel(crosstab.index.name or 'Actual')
    plt.xlabel(crosstab.columns.name or 'Predicted')

    plt.show()

def matchClusters(cfg, df, novel_col, target_col):
    # Extract the true and predicted labels
    y_true = df[target_col].to_numpy()
    y_pred = df[novel_col].to_numpy()
    
    # Create the cost matrix for the Hungarian algorithm
    cost_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    
    for i, valid_cluster in enumerate(set(y_true)):
        for j, novel_cluster in enumerate(set(y_pred)):
            cost_matrix[i, j] = -np.sum((y_true == valid_cluster) & (y_pred == novel_cluster))
    
    # Use the Hungarian algorithm to find the best cluster matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_clusters = list(zip(row_ind, col_ind))
    
    # Reorder the predicted labels according to the best matching
    y_pred_reordered = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        y_pred_reordered[y_pred == j] = i
    
    # Update the dataset with the reassigned cluster labels
    df[novel_col] = y_pred_reordered
    print(f"Best Cluster Matching: {matched_clusters}")
    return df

def compareClusterings(cfg, df, x_col, y_col):
    # Extract the true and predicted labels
    scores_x = df[x_col].to_numpy()
    scores_y = df[y_col].to_numpy()
    
    # Compute ARI and NMI
    ari_score = adjusted_rand_score(scores_x, scores_y)
    nmi_score = normalized_mutual_info_score(scores_x, scores_y)
    
    # Calculate the confusion matrix with the reordered predictions
    confusion = pd.crosstab(scores_y, scores_x, rownames=[y_col], colnames=[x_col])
    
    # Calculate Cluster Accuracy
    correct_assignments = np.sum(np.diag(confusion.values))
    total_assignments = np.sum(confusion.values)
    cluster_accuracy = correct_assignments / total_assignments

    # Print the results
    print(f"Adjusted Rand Index (ARI): {ari_score}")
    print(f"Normalized Mutual Information (NMI): {nmi_score}")
    print(f"Cluster Accuracy: {cluster_accuracy:.4f}")
    print("Confusion Matrix:")
    plot_confusion_matrix(confusion)