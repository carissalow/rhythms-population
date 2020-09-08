import pandas as pd

summarised = snakemake.params["summarised"]
symptom_cols = snakemake.params["symptom_cols"]
target_cols = snakemake.params["target_cols"]

if summarised == "summarised":
    raise ValueError("Do NOT support 'summarised' type currently.")

participant_symptoms = pd.read_csv(snakemake.input["participant_symptoms"])

# Set the M + 1 SD score as the threshold: round down
participant_symptoms["target_sum"] = participant_symptoms[target_cols].sum(axis=1)
if participant_symptoms.shape[0] <= 1:
    threshold = participant_symptoms["target_sum"].mean()
else:
    q1 = participant_symptoms["target_sum"].quantile(q=0.25)
    q3 = participant_symptoms["target_sum"].quantile(q=0.75)

    if q1 == 10:
        threshold = 9
    elif q3 == 0:
        threshold = 0
    else:
        threshold = participant_symptoms["target_sum"].mean() #+ participant_symptoms["target_sum"].std()


# Get target based on the threshold
participant_symptoms["target"] = participant_symptoms["target_sum"].apply(lambda score: 1 if score > threshold else 0)

# Only keep the target column
participant_symptoms = participant_symptoms[["pid", "local_date", "target"]]

participant_symptoms.to_csv(snakemake.output[0], index=False)