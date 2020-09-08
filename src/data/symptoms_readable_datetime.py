import numpy as np
import pandas as pd
from datetime import datetime

participant_symptoms = pd.read_csv(snakemake.input["participant_symptoms"])
pid = snakemake.params["pid"]
symptom_cols = snakemake.params["symptom_cols"]


participant_symptoms.insert(0, "pid", pid)
 
# extract "local_date_time" and "local_date" from "timestamp" field
participant_symptoms.loc[:, "local_date_time"] = participant_symptoms["timestamp"].apply(lambda x: datetime.fromtimestamp(x/1000))
participant_symptoms.loc[:, "local_date"] = participant_symptoms["local_date_time"].dt.date
 
# sort by "local_date" and keep the first record of each day
participant_symptoms.sort_values(by=["timestamp"], ascending=True, inplace=True)
participant_symptoms.drop_duplicates(subset=["local_date"], keep="first", inplace=True)
 
# for each symptom score: replace -1 with NaN
participant_symptoms[symptom_cols] = participant_symptoms[symptom_cols].replace(-1, np.NaN)
# drop rows where all cols in symptom_cols equal to np.NaN
participant_symptoms.dropna(subset=symptom_cols, how="all", inplace=True)
# replace the rest np.NaN in symptom_cols with 0
participant_symptoms[symptom_cols] = participant_symptoms[symptom_cols].fillna(0)

participant_symptoms = participant_symptoms[["pid", "local_date"] + symptom_cols]
 
participant_symptoms.to_csv(snakemake.output[0], index=False)
