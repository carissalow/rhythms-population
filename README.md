# Analysis code for Digital Biomarkers of Patient-Reported Symptom Burden in Pancreatic Surgery Patients

Installation instructions are in [RAPIDS beta installation](https://rapidspitt.readthedocs.io/en/latest/usage/installation.html)

Descriptions of features are in [RAPIDS beta features](https://rapidspitt.readthedocs.io/en/latest/features/extracted.html)

Check the end of the config.yaml file (`ANALYSIS` section) for all analysis parameters.

Check the rules/models.smk for all the rules and scripts involved in the analysis.

Check the tools/create_figure1_for_paper.py script for Figure 1.png of the paper. The script contains two steps: 1) rename top 20 important features with the readable names; 2) create density scatter plot which shows SHapley Additive exPlanation (SHAP) values for each feature, reflecting how much impact each feature has on model output.

# Reproducing results

1. Extract features and train models

Run `snakemake -j6`. Results are saved in the `data/processed/output_population_model/20hours_10bins/0.3|0.3_5_True` folder.


2. Create Figure 1.png figure

Run `python ./tools/create_figure1_for_paper.py`

# Notes

By default, the code is for next-day total symptom burden prediction (binary classification). You can change the `TARGET_COLS` parameter in config.yaml to predict next-day diarrhea or fatigue or pain symptom class.
