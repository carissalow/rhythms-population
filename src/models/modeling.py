import pandas as pd
import numpy as np
from modeling_utils import getMatchingColNames, dropZeroVarianceCols, getFittedScaler, getMetrics, getFeatureImportances, createPipeline, TimeSeriesGroupKFold
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV, cross_val_score, KFold
from mlxtend.feature_selection import SequentialFeatureSelector

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


##############################################################
# Summary of the workflow
# Step 1. Read parameters and data
# Step 2. Nested cross validation
# Step 3. Model evaluation
# Step 4. Save results, parameters, and metrics to CSV files
##############################################################



# Step 1. Read parameters and data
# Read parameters
model = snakemake.params["model"]
source = snakemake.params["source"]
summarised = snakemake.params["summarised"]
day_segment = snakemake.params["day_segment"]
scaler = snakemake.params["scaler"]
cv_method = snakemake.params["cv_method"]
categorical_operators = snakemake.params["categorical_operators"]
# categorical_colnames_demographic_features = snakemake.params["categorical_demographic_features"]
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
inner_cv = cv_class()
outer_cv = cv_class()

fold_id, fold_id_unique, pid, local_date, best_params, true_y, pred_y, pred_y_prob = [], [], [], [], [], [], [], []
feature_importances_all_folds = pd.DataFrame()
metrics_all_folds = {"accuracy": [], "precision0": [], "recall0": [], "f10": [], "precision1": [], "recall1": [], "f11": [], "f1_macro": [], "auc": [], "kappa": []}
fold_count = 1

# Outer cross validation
for train_index, test_index in outer_cv.split(data_x):

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
    # Feature selection: 
    """
    # method 1: sequential foward floating selection
    from lightgbm import LGBMClassifier
    feature_selector = SequentialFeatureSelector(estimator=LGBMClassifier(), 
           k_features=75,
           forward=True, 
           floating=True,
           scoring="f1_macro",
           cv=0)
    """
    # method 2: mutual information
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    feature_selector = SelectKBest(mutual_info_classif, k=70)

    if min(targets_value_counts) >= 6:
        # SMOTE requires n_neighbors <= n_samples, the default value of n_neighbors is 6
        clf = GridSearchCV(estimator=createPipeline(model, "SVMSMOTE", feature_selector=feature_selector), param_grid=model_hyperparams, cv=inner_cv, scoring="f1_macro")
    else:
        # RandomOverSampler: over-sample the minority class(es) by picking samples at random with replacement.
        clf = GridSearchCV(estimator=createPipeline(model, "RandomOverSampler", feature_selector=feature_selector), param_grid=model_hyperparams, cv=inner_cv, scoring="f1_macro")
    clf.fit(train_x, train_y.values.ravel())

    # Collect results and parameters
    best_params = best_params + [clf.best_params_] * test_x.shape[0]
    cur_fold_pred = clf.predict(test_x).tolist()
    pred_y = pred_y + cur_fold_pred

    proba_of_two_categories = clf.predict_proba(test_x).tolist()
    cur_fold_pred_prob = [probabilities[clf.classes_.tolist().index(1)] for probabilities in proba_of_two_categories]
    pred_y_prob = pred_y_prob + cur_fold_pred_prob

    # Step 3. Model evaluation
    metrics_current_fold = getMetrics(cur_fold_pred, cur_fold_pred_prob, test_y.values.ravel().tolist())
    for k, v in metrics_current_fold.items():
        metrics_all_folds[k].append(v)

    true_y = true_y + test_y.values.ravel().tolist()
    pid = pid + test_y.index.get_level_values("pid").tolist()
    local_date = local_date + test_y.index.get_level_values("local_date").tolist()
    #feature_importances_current_fold = getFeatureImportances(model, clf.best_estimator_.steps[2][1], clf.best_estimator_.steps[1][1].k_feature_names_)
    feature_importances_current_fold = getFeatureImportances(model, clf.best_estimator_.steps[2][1], train_x.columns[clf.best_estimator_.steps[1][1].get_support(indices=True)])
    feature_importances_all_folds = pd.concat([feature_importances_all_folds, feature_importances_current_fold], sort=False, axis=0)
    fold_id.extend([fold_count] * test_x.shape[0])
    fold_id_unique.append(fold_count)
    fold_count = fold_count + 1

# Step 4. Save results, parameters, and metrics to CSV files
fold_predictions = pd.DataFrame({"fold_id": fold_id, "pid": pid, "local_date": local_date, "hyperparameters": best_params, "true_y": true_y, "pred_y": pred_y, "pred_y_prob": pred_y_prob})
fold_metrics = pd.DataFrame({"fold_id": fold_id_unique, "accuracy": metrics_all_folds["accuracy"], "precision0": metrics_all_folds["precision0"], "recall0": metrics_all_folds["recall0"], "f10": metrics_all_folds["f10"], "precision1": metrics_all_folds["precision1"], "recall1": metrics_all_folds["recall1"], "f11": metrics_all_folds["f11"], "f1_macro": metrics_all_folds["f1_macro"], "auc": metrics_all_folds["auc"], "kappa": metrics_all_folds["kappa"]})
overall_results = pd.DataFrame({"num_of_rows": [num_of_rows], "num_of_features": [num_of_features], "rowsnan_colsnan_days_colsvar_threshold": [rowsnan_colsnan_days_colsvar_threshold], "model": [model], "cv_method": [cv_method], "source": [source], "scaler": [scaler], "day_segment": [day_segment], "summarised": [summarised], "accuracy": [computeAvgAndStd(metrics_all_folds["accuracy"])], "precision0": [computeAvgAndStd(metrics_all_folds["precision0"])], "recall0": [computeAvgAndStd(metrics_all_folds["recall0"])], "f10": [computeAvgAndStd(metrics_all_folds["f10"])], "precision1": [computeAvgAndStd(metrics_all_folds["precision1"])], "recall1": [computeAvgAndStd(metrics_all_folds["recall1"])], "f11": [computeAvgAndStd(metrics_all_folds["f11"])], "f1_macro": [computeAvgAndStd(metrics_all_folds["f1_macro"])], "auc": [computeAvgAndStd(metrics_all_folds["auc"])], "kappa": [computeAvgAndStd(metrics_all_folds["kappa"])]})
feature_importances_all_folds.insert(loc=0, column="fold_id", value=fold_id_unique)

fold_predictions.to_csv(snakemake.output["fold_predictions"], index=False)
fold_metrics.to_csv(snakemake.output["fold_metrics"], index=False)
overall_results.to_csv(snakemake.output["overall_results"], index=False)
feature_importances_all_folds.to_csv(snakemake.output["fold_feature_importances"], index=False)
