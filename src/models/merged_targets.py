import pandas as pd
from pandas.tseries.offsets import DateOffset

date_offset = int(snakemake.params["date_offset"])
symptom_cols = snakemake.params["symptom_cols"]
target_cols = snakemake.params["target_cols"]

selected_days = pd.read_csv("data/processed/selected_days.csv", index_col=["pid", "local_date"])

all_participants = pd.DataFrame()
for file_path in snakemake.input["participant_symptoms"]:
    participant_symptoms = pd.read_csv(file_path, parse_dates=["local_date"])
    participant_symptoms.loc[:, "local_date"] = (participant_symptoms["local_date"] - DateOffset(days = date_offset)).apply(lambda dt: dt.strftime("%Y-%m-%d"))
    all_participants = pd.concat([all_participants, participant_symptoms], axis=0)

all_participants.set_index(["pid", "local_date"], inplace=True)
selected_particiapnts = all_participants.loc[selected_days.index]
print(selected_particiapnts.shape)

selected_particiapnts.describe().to_csv("test.csv")

sys.exit(0)