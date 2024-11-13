import pandas as pd
import numpy as np

ds = pd.read_csv("dataset/train_corrigido.csv", header=0)

ds["id_num"] = ds["id"].str[4:].astype(int)
ds = ds.sort_values(by=["p_num", "id_num"], ascending=[True, True])

vars = ["insulin"]
# vars = ["bg", "carbs", "hr", "steps", "cals"]
time_tags = [
    "0:05",
    "0:10",
    "0:15",
    "0:20",
    "0:25",
    "0:30",
    "0:35",
    "0:40",
    "0:45",
    "0:50",
    "0:55",
    "1:00",
    "1:05",
    "1:10",
    "1:15",
    "1:20",
    "1:25",
    "1:30",
    "1:35",
    "1:40",
    "1:45",
    "1:50",
    "1:55",
    "2:00",
    "2:05",
    "2:10",
    "2:15",
    "2:20",
    "2:25",
    "2:30",
    "2:35",
    "2:40",
    "2:45",
    "2:50",
    "2:55",
    "3:00",
    "3:05",
    "3:10",
    "3:15",
    "3:20",
    "3:25",
    "3:30",
    "3:35",
    "3:40",
    "3:45",
    "3:50",
    "3:55",
    "4:00",
    "4:05",
    "4:10",
    "4:15",
    "4:20",
    "4:25",
    "4:30",
    "4:35",
    "4:40",
    "4:45",
    "4:50",
    "4:55",
    "5:00",
    "5:05",
    "5:10",
    "5:15",
    "5:20",
    "5:25",
    "5:30",
    "5:35",
    "5:40",
    "5:45",
    "5:50",
]

for var in vars:
    ds_vazios = ds[ds[f"{var}-0:00"].isnull()]
    print(var, "vazios", ds_vazios.shape)
    count = 0
    for index, row in ds_vazios.iterrows():
        count += 1
        print(count)

        # Tratando da hora da linha
        cur_time = row["time"]
        split_cur_time = cur_time.split(":")
        cur_hour = int(split_cur_time[0])
        cur_min = int(split_cur_time[1])

        for time_tag in time_tags:
            col_name = f"{var}-{time_tag}"
            if row[col_name] is None:
                continue

            split_time_tag = time_tag.split(":")
            tag_hour = int(split_time_tag[0])
            tag_min = int(split_time_tag[1])

            # Achando o time da linha que pode se beneficiar da informação
            desired_time_min = cur_min + tag_min
            if desired_time_min >= 60:
                tag_hour += 1
                desired_time_min -= 60

            desired_time_hour = cur_hour + tag_hour
            desired_time = f"{desired_time_hour:02}:{desired_time_min:02}:00"

            # Achando a linha a procurar o dado
            desired_line = ds.loc[
                (ds["p_num"] == row["p_num"])
                & (ds["id_num"] > row["id_num"])
                & (ds["time"] == desired_time)
            ]
            if desired_line.shape[0] <= 0:
                continue

            if pd.isnull(desired_line.iloc[0][f"{var}-{time_tag}"]):
                continue

            ds.loc[index, f"{var}-0:00"] = desired_line.iloc[0][f"{var}-{time_tag}"]
            print("Corrigido", var)
            break

ds.to_csv("dataset/train_corrigido_bg_insulimn.csv", index=False)
