import numpy as np
import pandas as pd
from datetime import timedelta

def excludeDaysInRange(days_to_analyse, start_date, end_date, last_analysis_date):
    num_of_days = (end_date - start_date).days
    if num_of_days > 0:
        exclude_dates = []
        for day in range(num_of_days + 1):
            exclude_dates.append(start_date + timedelta(days = day))
        days_to_analyse = days_to_analyse[~days_to_analyse["local_date"].isin(exclude_dates)]
        
        num_of_last_part_days = (last_analysis_date - end_date).days
        if num_of_last_part_days > 0:
            for day in range(num_of_last_part_days):
                curr_date = end_date + timedelta(days = day + 1)
                days_to_analyse.loc[days_to_analyse["local_date"] == curr_date, "is_readmitted"] = 1

    return days_to_analyse

def appendDaysInRange(days_to_analyse, start_date, end_date, day_type):
    num_of_days = (end_date - start_date).days
    if np.isnan(num_of_days):
        return days_to_analyse

    for day in range(num_of_days + 1):

        if day_type == -1:
            day_idx = (num_of_days - day + 1) * day_type
        elif day_type == 1:
            day_idx = day + 1
        else:
            day_idx = 0

        days_to_analyse = days_to_analyse.append({"local_date": start_date + timedelta(days = day), "day_idx": day_idx, "is_readmitted": 0}, ignore_index=True)
    
    return days_to_analyse

days_before_surgery = int(snakemake.params["days_before_surgery"])
days_in_hospital = str(snakemake.params["days_in_hospital"])
days_after_discharge = int(snakemake.params["days_after_discharge"])
participant_info = pd.read_csv(snakemake.input["participant_info"], parse_dates=["surgery_date", "discharge_date", "readmission1_date", "readmission1_discharge", "readmission2_date", "readmission2_discharge", "readmission3_date", "readmission3_discharge"])
days_to_analyse = pd.DataFrame(columns = ["local_date", "day_idx", "is_readmitted"])

try:
    surgery_date, discharge_date = participant_info["surgery_date"].iloc[0].date(), participant_info["discharge_date"].iloc[0].date()
except:
    pass
else:
    start_date = surgery_date - timedelta(days = days_before_surgery)
    end_date = discharge_date + timedelta(days = days_after_discharge)

    # days before surgery: -1; in hospital: 0; after discharge: 1
    days_to_analyse = appendDaysInRange(days_to_analyse, start_date, surgery_date - timedelta(days = 1), -1)
    if days_in_hospital == "T":
        days_to_analyse = appendDaysInRange(days_to_analyse, surgery_date, discharge_date, 0)
    days_to_analyse = appendDaysInRange(days_to_analyse, discharge_date + timedelta(days = 1), end_date, 1)
    
    # exclude readmission
    try:
        readmission1_date, readmission1_discharge = participant_info["readmission1_date"].iloc[0].date(), participant_info["readmission1_discharge"].iloc[0].date()
    except:
        pass
    else:
        days_to_analyse = excludeDaysInRange(days_to_analyse, readmission1_date, readmission1_discharge, end_date)
    
    try:
        readmission2_date, readmission2_discharge = participant_info["readmission2_date"].iloc[0].date(), participant_info["readmission2_discharge"].iloc[0].date()
    except:
        pass
    else:
        days_to_analyse = excludeDaysInRange(days_to_analyse, readmission2_date, readmission2_discharge, end_date)
    
    try:
        readmission3_date, readmission3_discharge = participant_info["readmission3_date"].iloc[0].date(), participant_info["readmission3_discharge"].iloc[0].date()
    except:
        pass
    else:
        days_to_analyse = excludeDaysInRange(days_to_analyse, readmission3_date, readmission3_discharge, end_date)
    
days_to_analyse.to_csv(snakemake.output[0], index=False)
