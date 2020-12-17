ruleorder: nan_cells_ratio_of_cleaned_features > merge_features_and_targets

rule days_to_analyse:
    input:
        participant_info = "data/raw/{pid}/" + config["PARAMS_FOR_ANALYSIS"]["GROUNDTRUTH_TABLE"] + "_raw.csv"
    params:
        days_before_surgery = "{days_before_surgery}",
        days_in_hospital = "{days_in_hospital}",
        days_after_discharge= "{days_after_discharge}"
    output:
        "data/interim/{pid}/days_to_analyse_{days_before_surgery}_{days_in_hospital}_{days_after_discharge}.csv"
    script:
        "../src/models/select_days_to_analyse.py"

# rule merged_targets:
#     input:
#         participant_symptoms = expand("data/raw/{pid}/" + config["PARAMS_FOR_ANALYSIS"]["TARGET_TABLE"] + "_with_datetime.csv", pid=['p01', 'p02', 'p12', 'p15', 'p16', 'p17', 'p18', 'p20', 'p22', 'p23', 'p24', 'p26', 'p28', 'p29', 'p30', 'p34', 'p36', 'p37', 'p38', 'p50', 'p58'])
#     params:
#         date_offset = config["PARAMS_FOR_ANALYSIS"]["DATE_OFFSET"],
#         symptom_cols = config["PARAMS_FOR_ANALYSIS"]["SYMPTOM_COLS"],
#         target_cols = config["PARAMS_FOR_ANALYSIS"]["TARGET_COLS"]        
#     output:
#         "data/processed/merged_targets.csv"
#     script:
#         "../src/models/merged_targets.py"

rule targets:
    input:
        participant_symptoms = "data/raw/{pid}/" + config["PARAMS_FOR_ANALYSIS"]["TARGET_TABLE"] + "_with_datetime.csv"
    params:
        summarised = "{summarised}",
        symptom_cols = config["PARAMS_FOR_ANALYSIS"]["SYMPTOM_COLS"],
        target_cols = config["PARAMS_FOR_ANALYSIS"]["TARGET_COLS"]
    output:
        "data/processed/{pid}/targets_{summarised}.csv"
    script:
        "../src/models/targets.py"

rule demographic_features:
    input:
        participant_info = "data/raw/{pid}/" + config["PARAMS_FOR_ANALYSIS"]["GROUNDTRUTH_TABLE"] + "_raw.csv"
    params:
        pid = "{pid}",
        features = config["PARAMS_FOR_ANALYSIS"]["DEMOGRAPHIC_FEATURES"]
    output:
        "data/processed/{pid}/demographic_features.csv"
    script:
        "../src/features/demographic_features.py"

rule merge_features_for_individual_model:
    input:
        feature_files = input_merge_features_of_single_participant,
        phone_valid_sensed_days = optional_input_valid_sensed_days,
        days_to_include = optional_input_days_to_include
    params:
        source = "{source}"
    output:
        "data/processed/{pid}/data_for_individual_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{source}_{day_segment}_original.csv"
    script:
        "../src/models/merge_features_for_individual_model.R"

rule merge_features_for_population_model:
    input:
        feature_files = expand("data/processed/{pid}/data_for_individual_model/{{min_valid_hours_per_day}}hours_{{min_valid_bins_per_hour}}bins/{{source}}_{{day_segment}}_original.csv", pid=config["PIDS"])
    output:
        "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{source}_{day_segment}_original.csv"
    script:
        "../src/models/merge_features_for_population_model.R"

rule merge_demographicfeatures_for_population_model:
    input:
        data_files = expand("data/processed/{pid}/demographic_features.csv", pid=config["PIDS"])
    output:
        "data/processed/data_for_population_model/demographic_features.csv"
    script:
        "../src/models/merge_data_for_population_model.py"

rule merge_targets_for_population_model:
    input:
        data_files = expand("data/processed/{pid}/targets_{{summarised}}.csv", pid=config["PIDS"])
    output:
        "data/processed/data_for_population_model/targets_{summarised}.csv"
    script:
        "../src/models/merge_data_for_population_model.py"

rule clean_features_for_individual_model:
    input:
        rules.merge_features_for_individual_model.output
    params:
        features_exclude_day_idx = config["PARAMS_FOR_ANALYSIS"]["FEATURES_EXCLUDE_DAY_IDX"],
        cols_nan_threshold = "{cols_nan_threshold}",
        cols_var_threshold = "{cols_var_threshold}",
        days_threshold = "{days_threshold}",
        rows_nan_threshold = "{rows_nan_threshold}",
    output:
        "data/processed/{pid}/data_for_individual_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_clean.csv"
    script:
        "../src/models/clean_features_for_model.R"

rule clean_features_for_population_model:
    input:
        rules.merge_features_for_population_model.output
    params:
        features_exclude_day_idx = config["PARAMS_FOR_ANALYSIS"]["FEATURES_EXCLUDE_DAY_IDX"],
        cols_nan_threshold = "{cols_nan_threshold}",
        cols_var_threshold = "{cols_var_threshold}",
        days_threshold = "{days_threshold}",
        rows_nan_threshold = "{rows_nan_threshold}",
    output:
        "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_clean.csv"
    script:
        "../src/models/clean_features_for_model.R"

rule nan_cells_ratio_of_cleaned_features:
    input:
        cleaned_features = "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_clean.csv"
    output:
        "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_nancellsratio.csv"
    script:
        "../src/models/nan_cells_ratio_of_cleaned_features.py"
 
rule merge_features_and_targets:
    input:
        cleaned_features = "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_clean.csv",
        # demographic_features = "data/processed/data_for_population_model/demographic_features.csv",
        targets = "data/processed/data_for_population_model/targets_{summarised}.csv",
    params:
        summarised = "{summarised}",
        cols_var_threshold = "{cols_var_threshold}",
        numerical_operators = config["PARAMS_FOR_ANALYSIS"]["NUMERICAL_OPERATORS"],
        categorical_operators = config["PARAMS_FOR_ANALYSIS"]["CATEGORICAL_OPERATORS"],
        features_exclude_day_idx = config["PARAMS_FOR_ANALYSIS"]["FEATURES_EXCLUDE_DAY_IDX"],
        date_offset = config["PARAMS_FOR_ANALYSIS"]["DATE_OFFSET"],
        days_threshold = config["PARAMS_FOR_ANALYSIS"]["PARTICIPANT_DAYS_THRESHOLD"]
    output:
        "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_{summarised}.csv"
    script:
        "../src/models/merge_features_and_targets.py"
 
rule baseline:
    input:
        "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_{summarised}.csv"
    params:
        cv_method = "{cv_method}",
        rowsnan_colsnan_days_colsvar_threshold = "{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}",
    output:
        "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/baseline/{cv_method}/{source}_{day_segment}_{summarised}.csv"
    log:
        "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/baseline/{cv_method}/{source}_{day_segment}_{summarised}_notes.log"
    script:
        "../src/models/baseline.py"
 
 
rule modeling:
    input:
        data = "data/processed/data_for_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{source}_{day_segment}_{summarised}.csv"
    params:
        model = "{model}",
        cv_method = "{cv_method}",
        source = "{source}",
        day_segment = "{day_segment}",
        summarised = "{summarised}",
        scaler = "{scaler}",
        categorical_operators = config["PARAMS_FOR_ANALYSIS"]["CATEGORICAL_OPERATORS"],
        model_hyperparams = config["PARAMS_FOR_ANALYSIS"]["MODEL_HYPERPARAMS"],
        rowsnan_colsnan_days_colsvar_threshold = "{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}"
    threads: 2
    output:
        fold_predictions = "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/fold_predictions.csv",
        fold_metrics = "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/fold_metrics.csv",
        overall_results = "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/overall_results.csv",
        fold_feature_importances = "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/fold_feature_importances.csv",
        participant_results = "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/participant_results.csv",
    log:
        "data/processed/output_population_model/{min_valid_hours_per_day}hours_{min_valid_bins_per_hour}bins/{rows_nan_threshold}|{cols_nan_threshold}_{days_threshold}_{cols_var_threshold}/{model}/{cv_method}/{source}_{day_segment}_{summarised}_{scaler}/notes.log"
    script:
        "../src/models/modeling.py"

