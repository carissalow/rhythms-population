import matplotlib.pyplot as plt
import pandas as pd
import shap



name_mappings = {
    "last_score": "most recent symptom score",
    "day_idx": "day index",
    "avg_score": "mean symptom burden score to date",
    "acc_daily_mediandurationnonexertionalactivityepisode": "median duration nonexertional activity episodes",
    "step_daily_countepisodeactivebout": "number of active bouts",
    "step_daily_sumdurationactivebout": "total duration of active bouts",
    "heartrate_daily_minhr": "minimum heart rate",
    "location_daily_movingtostaticratio": "moving-to-static ratio",
    "step_daily_maxdurationactivebout": "maximum duration active bout",
    "sleep_daily_sumdurationawakenap": "total time awake during naps",
    "step_daily_stddurationsedentarybout": "standard deviation of sedentary bout duration",
    "ar_daily_countuniqueactivities": "number of unique activities",
    "location_daily_timeattop1": "total time at most frequent location",
    "sleep_daily_sumdurationasleepmain": "total time asleep during main sleep",
    "acc_daily_sumdurationexertionalactivityepisode": "total duration exertional activity episodes",
    "heartrate_daily_minutesoncardiozone": "minutes in Fitbit cardio heart rate zone",
    "heartrate_daily_minutesonpeakzone": "minutes in Fitbit peak heart rate zone",
    "location_daily_maxlengthstayatclusters": "duration of longest stay at a significant location",
    "light_daily_maxlux": "maximum ambient luminance",
    "acc_daily_maxmagnitude": "maximum magnitude of acceleration"
}


shap_all_folds = pd.read_csv("rapids/data/processed/output_population_model/20hours_10bins/0.3|0.3_5_True/LightGBM/TimeSeriesGroupKFold/phone_fitbit_features_daily_notsummarised_notnormalized/shap_all_folds.csv").rename(columns=name_mappings)
shap_test_all_folds = pd.read_csv("rapids/data/processed/output_population_model/20hours_10bins/0.3|0.3_5_True/LightGBM/TimeSeriesGroupKFold/phone_fitbit_features_daily_notsummarised_notnormalized/shap_test_all_folds.csv").rename(columns=name_mappings)


shap_test_all_folds.set_index(["pid", "local_date"], inplace=True)

shap.summary_plot(shap_values=shap_all_folds.fillna(0).values, features=shap_test_all_folds,  plot_size=(12, 8), show=False)
plt.tight_layout()
plt.savefig("Figure 1.png")
plt.clf()
