import numpy as np
import pandas as pd
from statistics import mean
from modeling_utils import getMetrics, createPipeline, TimeSeriesGroupKFold
from sklearn.model_selection import LeaveOneOut


# As we do not have probability of each category, use label to denote the probability directly.
# The probability will only be used to calculate the AUC value.
def baselineAccuracyOfMajorityClassClassifier(targets):
    majority_class = targets["target"].value_counts().idxmax()
    pred_y = [majority_class] * targets.shape[0]
    pred_y_prob = pred_y
    metrics = getMetrics(pred_y, pred_y_prob, targets["target"].values.ravel().tolist())
    return metrics, majority_class

def baselineMetricsOfRandomWeightedClassifier(targets, majority_ratio, majority_class, iter_times):
    metrics_all_iters = {"accuracy": [], "precision0":[], "recall0": [], "f10": [], "precision1": [], "recall1": [], "f11": [], "auc": [], "kappa": []}
    probabilities = [0, 0]
    probabilities[majority_class], probabilities[1 - majority_class] = majority_ratio, 1 - majority_ratio
    for i in range(iter_times):
        pred_y = np.random.RandomState(i).multinomial(1, probabilities, targets.shape[0])[:,1].tolist()
        pred_y_prob = pred_y
        metrics = getMetrics(pred_y, pred_y_prob, targets["target"].values.ravel().tolist())
        for key in metrics_all_iters.keys():
            metrics_all_iters[key].append(metrics[key].item())
    # Calculate average metrics across all iterations
    avg_metrics = {}
    for key in metrics_all_iters.keys():
        avg_metrics[key] = mean(metrics_all_iters[key])
    return avg_metrics

def baselineMetricsOfDTWithDayIdxFeatures(cv_method, data_x, data_y, oversampler_type):
    pred_y, true_y = [], []
    for train_index, test_index in cv_method.split(data_x):
        train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
        train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]
        clf = createPipeline("DT", oversampler_type)
        clf.fit(train_x, train_y.values.ravel())
        pred_y = pred_y + clf.predict(test_x).ravel().tolist()
        pred_y_prob = pred_y
        true_y = true_y + test_y.values.ravel().tolist()
    return getMetrics(pred_y, pred_y_prob, true_y)


data = pd.read_csv(snakemake.input[0], index_col=["pid", "local_date"])
cv_method = globals()[snakemake.params["cv_method"]](n_splits = data.shape[0])
rowsnan_colsnan_days_colsvar_threshold = snakemake.params["rowsnan_colsnan_days_colsvar_threshold"]

data_x, data_y = data.drop("target", axis=1), data[["target"]]
targets_value_counts = data_y["target"].value_counts()


baseline_metrics = pd.DataFrame(columns=["method", "fullMethodName", "accuracy", "precision0", "recall0", "f10", "precision1", "recall1", "f11", "auc", "kappa"])
if len(targets_value_counts) < 2:
    fout = open(snakemake.log[0], "w")
    fout.write(targets_value_counts.to_string())
    fout.close()

else:
    if min(targets_value_counts) >= 6:
        oversampler_type = "SMOTE"
    else:
        oversampler_type = "RandomOverSampler"
    # Baseline 1: majority class classifier => predict every sample as majority class
    baseline1_metrics, majority_class = baselineAccuracyOfMajorityClassClassifier(data_y)
    majority_ratio = baseline1_metrics["accuracy"]
    # Baseline 2: random weighted classifier => random classifier with binomial distribution
    baseline2_metrics = baselineMetricsOfRandomWeightedClassifier(data_y, majority_ratio, majority_class, 1000)
    # Baseline 3: decision tree with day_idx features
    baseline3_metrics = baselineMetricsOfDTWithDayIdxFeatures(cv_method, data_x[["day_idx"]], data_y, oversampler_type)
    
    baselines = [baseline1_metrics, baseline2_metrics, baseline3_metrics]

    baseline_metrics = pd.DataFrame({"method": ["majority", "rwc", "dt"],
                             "fullMethodName": ["MajorityClassClassifier", "RandomWeightedClassifier", "DecisionTreeWithDayIdxFeatures"],
                             "accuracy": [baseline["accuracy"] for baseline in baselines],
                             "precision0": [baseline["precision0"] for baseline in baselines],
                             "recall0": [baseline["recall0"] for baseline in baselines],
                             "f10": [baseline["f10"] for baseline in baselines],
                             "precision1": [baseline["precision1"] for baseline in baselines],
                             "recall1": [baseline["recall1"] for baseline in baselines],
                             "f11": [baseline["f11"] for baseline in baselines],
                             "auc": [baseline["auc"] for baseline in baselines],
                             "kappa": [baseline["kappa"] for baseline in baselines]})

baseline_metrics.to_csv(snakemake.output[0], index=False)
