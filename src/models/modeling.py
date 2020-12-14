import pandas as pd
import numpy as np
from modeling_utils import getMatchingColNames, getFittedScaler, getMetrics, getFeatureImportances, createPipeline, TimeSeriesGroupKFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import shap
import matplotlib.pyplot as plt

np.random.seed(0)

def computeAvgAndStd(metrics):
    return str(round(np.nanmean(metrics), 4)) + "(" + str(round(np.nanstd(metrics),4)) + ")"

def imputeNumericalFeaturesWithNearestTwoDays(train_numerical_features, test_numerical_features, flag):

    imputed_numerical_features = pd.DataFrame()

    if flag == "train":
        ordered_numerical_features = train_numerical_features
    elif flag == "test":
        ordered_numerical_features = test_numerical_features
    else:
        raise ValueError("flag should be 'train' or 'test'")
    
    # get unique pids with order preserved
    pids = ordered_numerical_features.index.get_level_values("pid")
    _, idx = np.unique(pids, return_index=True)
    pids = pids[np.sort(idx)]

    # fill nan with the avg value of closest two days for each participant
    for pid in pids:

        if flag == "train":
            train_numerical_features_for_pid = train_numerical_features.xs(pid, level="pid")
            if train_numerical_features_for_pid.isnull().values.any():
                numerical_features_for_pid = (train_numerical_features_for_pid.fillna(method="ffill") + train_numerical_features_for_pid.fillna(method="bfill"))/2
                numerical_features_for_pid = numerical_features_for_pid.fillna(method="ffill") if numerical_features_for_pid.isnull().values.any() else numerical_features_for_pid
                numerical_features_for_pid = numerical_features_for_pid.fillna(method="bfill") if numerical_features_for_pid.isnull().values.any() else numerical_features_for_pid
            else:
                numerical_features_for_pid = train_numerical_features_for_pid
        
        if flag == "test":
            test_numerical_features_for_pid = test_numerical_features.xs(pid, level="pid")
            if (test_numerical_features_for_pid.isnull().values.any()) and (pid in train_numerical_features.index.get_level_values("pid")):
                train_numerical_features_for_pid = train_numerical_features.xs(pid, level="pid")
                numerical_features_for_pid = test_numerical_features_for_pid.fillna(train_numerical_features_for_pid.iloc[-1])
            else:
                numerical_features_for_pid = test_numerical_features_for_pid
        # add pid as the first layer of index again
        numerical_features_for_pid = pd.concat({pid: numerical_features_for_pid}, names=["pid"])

        imputed_numerical_features = pd.concat([imputed_numerical_features, numerical_features_for_pid], axis=0)
    
    # fill the rest nan with the avg value of all participants
    imputed_numerical_features = imputed_numerical_features.fillna(train_numerical_features.mean())

    return imputed_numerical_features

def imputeCategoricalFeaturesWithMode(train_categorical_features, test_categorical_features, flag):

    if flag == "train":
        categorical_features = train_categorical_features
    elif flag == "test":
        categorical_features = test_categorical_features
    else:
        raise ValueError("flag should be 'train' or 'test'")    

    if not categorical_features.isnull().values.any():
        imputed_categorical_features = categorical_features
    else:
        categorical_features.reset_index(inplace=True)
        mode_values = train_categorical_features.groupby(["pid"]).agg(lambda x: pd.Series.mode(x)[0]).to_dict()
        for col in categorical_features.columns:
            nan_index = categorical_features[col].isnull()
            if nan_index.any() or col == "pid" or col == "local_date":
                continue
            categorical_features.loc[nan_index, col] = categorical_features.loc[nan_index, "pid"].map(mode_values[col])
        imputed_categorical_features = categorical_features.set_index(["pid", "local_date"])

    return imputed_categorical_features

def featureScaling(train_features, features, scaler, flag):

    scaled_features = pd.DataFrame()

    last_pid = ""
    for pid in features.index.get_level_values("pid"):

        if pid == last_pid:
            continue

        if pid in train_features.index.get_level_values("pid"):
            train_features_for_pid = train_features.xs(pid, level="pid")
        else:
            train_features_for_pid = train_features
        
        features_for_pid = features.xs(pid, level="pid")

        if scaler == "minmaxscaler" and flag == "test":
            features_for_pid = features_for_pid.clip(train_features_for_pid.min(), train_features_for_pid.max(), axis=1)


        fitted_scaler = getFittedScaler(train_features_for_pid, scaler)
        scaled_features_for_pid = pd.DataFrame(fitted_scaler.transform(features_for_pid), index=features_for_pid.index, columns=features_for_pid.columns)

        # add pid as the first layer of index again
        scaled_features_for_pid = pd.concat({pid: scaled_features_for_pid}, names=["pid"])

        scaled_features = pd.concat([scaled_features, scaled_features_for_pid], axis=0)
        last_pid = pid
    
    return scaled_features



def preprocessNumericalFeatures(train_numerical_features, test_numerical_features, scaler, flag):
    # fillna with avg value of nearest two days
    train_numerical_features = imputeNumericalFeaturesWithNearestTwoDays(train_numerical_features, test_numerical_features, "train")
    if flag == "train":
        numerical_features = train_numerical_features
    else:
        numerical_features = imputeNumericalFeaturesWithNearestTwoDays(train_numerical_features, test_numerical_features, flag)

    # normalize
    if scaler != "notnormalized":
        numerical_features = featureScaling(train_numerical_features, numerical_features, scaler, flag)

    return numerical_features

def preprocessCategoricalFeatures(train_categorical_features, test_categorical_features, flag):
    # fillna with mode
    categorical_features = imputeCategoricalFeaturesWithMode(train_categorical_features, test_categorical_features, flag)
    # one-hot encoding
    categorical_features = categorical_features.apply(lambda col: col.astype("category"))
    categorical_features = pd.get_dummies(categorical_features)
    return categorical_features

def splitNumericalCategoricalFeatures(features, categorical_feature_colnames):
    numerical_features = features.drop(categorical_feature_colnames, axis=1)
    categorical_features = features[categorical_feature_colnames].copy()
    return numerical_features, categorical_features

def preprocesFeatures(train_numerical_features, test_numerical_features, train_categorical_features, test_categorical_features, scaler, flag):
    numerical_features = preprocessNumericalFeatures(train_numerical_features, test_numerical_features, scaler, flag)
    if not train_categorical_features.empty:
        categorical_features = preprocessCategoricalFeatures(train_categorical_features, test_categorical_features, flag)
        features = pd.concat([numerical_features, categorical_features], axis=1)
    else:
        features = numerical_features
    return features


################################################################################################
# Summary of the workflow
# Step 1. Read parameters and data
# Step 2. Nested cross validation: LeaveOneOut(OuterCV) & 3-folds(InnerCV) + temporal order
# Step 3. Model evaluation
# Step 4. Save results, parameters, and metrics to CSV files
################################################################################################



# Step 1. Read parameters and data
# Read parameters
model = snakemake.params["model"]
source = snakemake.params["source"]
summarised = snakemake.params["summarised"]
day_segment = snakemake.params["day_segment"]
scaler = snakemake.params["scaler"]
cv_method = snakemake.params["cv_method"]
categorical_operators = snakemake.params["categorical_operators"]
model_hyperparams = snakemake.params["model_hyperparams"][model]
rowsnan_colsnan_days_colsvar_threshold = snakemake.params["rowsnan_colsnan_days_colsvar_threshold"] # thresholds for data cleaning


# Read data and split
if summarised == "summarised":
    data = pd.read_csv(snakemake.input["data"], index_col=["pid"])
elif summarised == "notsummarised":
    data = pd.read_csv(snakemake.input["data"], index_col=["pid", "local_date"])
else:
    raise ValueError("SUMMARISED parameter in config.yaml can only be 'summarised' or 'notsummarised'")


# drop highly correlated features
## line 1: >= 0.9; line 2: == 1
#correlated_cols = ["heartrate_daily_avghr", "step_daily_countepisodesedentarybout", "sleep_daily_sumdurationinbedall", "sleep_daily_countepisodenap", "sleep_daily_sumdurationinbednap", "sleep_daily_sumdurationinbedmain"] + ["heartrate_daily_medianhr", "sleep_daily_sumdurationasleepall", "sleep_daily_sumdurationawakemain", "step_daily_stdallsteps", "step_daily_sumallsteps", "step_daily_sumdurationactivebout", "screen_daily_stddurationunlock", "screen_daily_maxdurationunlock", "location_daily_meanlengthstayatclusters", "location_daily_numberlocationtransitions", "location_daily_loglocationvariance", "acc_daily_stddurationnonexertionalactivityepisode", "acc_daily_stddurationexertionalactivityepisode", "acc_daily_validsensedminutes"]
correlated_cols = ["heartrate_daily_avghr", "step_daily_countepisodesedentarybout", "sleep_daily_sumdurationinbedall", "sleep_daily_countepisodenap", "sleep_daily_sumdurationinbednap", "sleep_daily_sumdurationinbedmain"]
for col in correlated_cols:
    if col in data.columns:
        del data[col]



# for circadianmovement feature: the largest value is 6.34. We replace inf with 10.
data = data.replace(np.inf, 10)

data_x, data_y = data.drop("target", axis=1), data[["target"]]
categorical_feature_colnames = getMatchingColNames(categorical_operators, data_x)



# Step 2. Nested cross validation
cv_class = globals()[cv_method]
inner_cv = cv_class(n_splits=3)
outer_cv = cv_class(n_splits=875)

fold_id, fold_id_unique, pid, local_date, best_params, true_y, pred_y, pred_y_proba = [], [], [], [], [], [], [], []
feature_importances_all_folds = pd.DataFrame()
shap_all_folds, test_all_folds = pd.DataFrame(), pd.DataFrame()
metrics_all_folds = {"accuracy": [], "precision0": [], "recall0": [], "f10": [], "precision1": [], "recall1": [], "f11": [], "f1_macro": [], "auc": [], "kappa": []}
fold_count = 1

groups = data.index.get_level_values("pid").to_numpy()

# Outer cross validation
for train_index, test_index in outer_cv.split(data_x, groups=groups):

    # Split train and test, numerical and categorical features
    train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
    train_numerical_features, train_categorical_features = splitNumericalCategoricalFeatures(train_x, categorical_feature_colnames)
    train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]
    test_numerical_features, test_categorical_features = splitNumericalCategoricalFeatures(test_x, categorical_feature_colnames)

    # Preprocess: impute and normalize
    train_x = preprocesFeatures(train_numerical_features, None, train_categorical_features, None, scaler, "train")
    test_x = preprocesFeatures(train_numerical_features, test_numerical_features, train_categorical_features, test_categorical_features, scaler, "test")
    train_x, test_x = train_x.align(test_x, join='outer', axis=1, fill_value=0) # in case we get rid off categorical columns

    # Compute number of participants and features
    # values do not change between folds
    if fold_count == 1:
        num_of_rows = train_x.shape[0] + test_x.shape[0]
        num_of_features = train_x.shape[1]

    targets_value_counts = train_y["target"].value_counts()
    if len(targets_value_counts) < 2 or max(targets_value_counts) < 5:
        notes = open(snakemake.log[0], mode="w")
        notes.write(targets_value_counts.to_string())
        notes.close()
        break

    # Inner cross validation
    # Feature selection: mutual information
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    feature_selector = SelectKBest(mutual_info_classif, k=75)

    if min(targets_value_counts) >= 6:
        # SMOTE requires n_neighbors <= n_samples, the default value of n_neighbors is 6
        clf = RandomizedSearchCV(estimator=createPipeline(model, "SVMSMOTE", feature_selector=feature_selector), param_distributions=model_hyperparams, cv=inner_cv, scoring="roc_auc", refit=True, random_state=10, n_iter=3) # param_grid
    else:
        # RandomOverSampler: over-sample the minority class(es) by picking samples at random with replacement.
        clf = RandomizedSearchCV(estimator=createPipeline(model, "RandomOverSampler", feature_selector=feature_selector), param_distributions=model_hyperparams, cv=inner_cv, scoring="roc_auc", refit=True, random_state=10, n_iter=3)
    clf.fit(train_x, train_y.values.ravel())

    # plot: interpret our model
    feature_names = train_x.columns[clf.best_estimator_.steps[1][1].get_support(indices=True)]
    explainer = shap.TreeExplainer(clf.best_estimator_.steps[2][1])
    test_current_fold = test_x[feature_names]
    shap_values = explainer.shap_values(test_current_fold)

    shap_current_fold = pd.DataFrame(data=shap_values[1], columns=feature_names)
    shap_all_folds = pd.concat([shap_all_folds, shap_current_fold], axis=0, sort=False)
    test_all_folds = pd.concat([test_all_folds, test_current_fold], axis=0, sort=False)

    # Collect results and parameters
    best_params = best_params + [clf.best_params_] * test_x.shape[0]
    cur_fold_pred = clf.predict(test_x).tolist()
    pred_y = pred_y + cur_fold_pred

    proba_of_two_categories = clf.predict_proba(test_x).tolist()
    cur_fold_pred_proba = [probabilities[clf.classes_.tolist().index(1)] for probabilities in proba_of_two_categories]
    pred_y_proba = pred_y_proba + cur_fold_pred_proba

    true_y = true_y + test_y.values.ravel().tolist()
    pid = pid + test_y.index.get_level_values("pid").tolist()
    local_date = local_date + test_y.index.get_level_values("local_date").tolist()
    feature_importances_current_fold = getFeatureImportances(model, clf.best_estimator_.steps[2][1], train_x.columns[clf.best_estimator_.steps[1][1].get_support(indices=True)])
    feature_importances_all_folds = pd.concat([feature_importances_all_folds, feature_importances_current_fold], sort=False, axis=0)
    fold_id.extend([fold_count] * test_x.shape[0])
    fold_id_unique.append(fold_count)
    fold_count = fold_count + 1


# Step 3. Model evaluation
metrics = getMetrics(pred_y, pred_y_proba, true_y)
shap.summary_plot(shap_values=shap_all_folds.fillna(0).values, features=test_all_folds,  plot_size=(12, 8), show=False)
plt.tight_layout()
plt.savefig("summary_plot_allfolds.png")
plt.clf()



# Step 4. Save results, parameters, and metrics to CSV files
fold_predictions = pd.DataFrame({"fold_id": fold_id, "pid": pid, "local_date": local_date, "hyperparameters": best_params, "true_y": true_y, "pred_y": pred_y, "pred_y_proba": pred_y_proba})
fold_metrics = pd.DataFrame()
overall_results = pd.DataFrame({"num_of_rows": [num_of_rows], "num_of_features": [str(num_of_features)+">"+"75"], "accuracy": [metrics["accuracy"]], "precision0": [metrics["precision0"]], "recall0": [metrics["recall0"]], "f10": [metrics["f10"]], "precision1": [metrics["precision1"]], "recall1": [metrics["recall1"]], "f11": [metrics["f11"]], "f1_macro": [metrics["f1_macro"]], "auc": [metrics["auc"]], "kappa": [metrics["kappa"]]})
feature_importances_all_folds.insert(loc=0, column="fold_id", value=fold_id_unique)

fold_predictions.to_csv(snakemake.output["fold_predictions"], index=False)
fold_metrics.to_csv(snakemake.output["fold_metrics"], index=False)
overall_results.to_csv(snakemake.output["overall_results"], index=False)
feature_importances_all_folds.to_csv(snakemake.output["fold_feature_importances"], index=False)







metrics_all_pids = {"accuracy": [], "precision0": [], "recall0": [], "f10": [], "precision1": [], "recall1": [], "f11": [], "f1_macro": [], "auc": [], "kappa": []}
count_0, count_1 = [], []

pids = list(set(fold_predictions["pid"]))
for pid in pids:
    pid_pred = fold_predictions[fold_predictions["pid"] == pid]
    count_0.append(pid_pred[pid_pred["true_y"] == 0].shape[0])
    count_1.append(pid_pred[pid_pred["true_y"] == 1].shape[0])
    metrics_per_pid = getMetrics(pid_pred["pred_y"], pid_pred["pred_y_proba"], pid_pred["true_y"])
    for key in metrics_per_pid.keys():
        metrics_all_pids[key].append(metrics_per_pid[key])

participant_results = pd.DataFrame(data=metrics_all_pids)
participant_results.insert(0, "pid", pids)
participant_results.insert(1, "count_0", count_0)
participant_results.insert(2, "count_1", count_1)
participant_results.to_csv(snakemake.output["participant_results"], index=False)

