import pandas as pd

summarised = snakemake.params["summarised"]
symptom_cols = snakemake.params["symptom_cols"]
target_cols = snakemake.params["target_cols"]

if summarised == "summarised":
    raise ValueError("Do NOT support 'summarised' type currently.")

participant_symptoms = pd.read_csv(snakemake.input["participant_symptoms"])

# Set the average score of each participant as the threshold: round down
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
        threshold = participant_symptoms["target_sum"].mean()


# Get target based on the threshold
participant_symptoms["target"] = participant_symptoms["target_sum"].apply(lambda score: 1 if score > threshold else 0)
participant_symptoms["last_score"] = participant_symptoms["target_sum"].shift(periods=1, fill_value=0)
participant_symptoms["avg_score"] = (participant_symptoms["target_sum"].cumsum() / (participant_symptoms.index + 1)).shift(periods=1, fill_value=0)

# Only keep the target column
participant_symptoms = participant_symptoms[["pid", "local_date", "last_score", "avg_score", "target"]]

participant_symptoms.to_csv(snakemake.output[0], index=False)