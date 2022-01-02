import pandas as pd
from random import sample
import random

VAL_CSV = "../hw4_data/mini/val.csv"
VAL_IMG = "../hw4_data/mini/val/"
OUTPUT_CSV = "../hw4_data/mini/val_testcase_5shot.csv"
N_SHOT = 10 # 
N_WAY = 5 # This can't be modified

df = pd.read_csv(VAL_CSV)
print(df)

# init NEW_FEATURE
NEW_FEATURE = {"episode_id" : []}
for c in range(N_WAY): # 5_way
    for s in range(N_SHOT):
        NEW_FEATURE[f"class{c}_support{s}"] = []
for q in range(75):
    NEW_FEATURE[f"query{q}"] = []

for episode_id in range(600):
    NEW_FEATURE["episode_id"].append(episode_id)
    sampled_classes = sample(range(16), N_WAY)
    # print(sampled_classes)

    # Sample support images
    query_list = []
    for c_idx, c in enumerate(sampled_classes):
        sampled_case = [c*600 + s for s in sample(range(600), 20)]

        # Use sampled_case[:5] as support 
        for s in range(N_SHOT):
            NEW_FEATURE[f"class{c_idx}_support{s}"].append(sampled_case[s])

        # Use sampled_case[5:] as query 
        query_list += sampled_case[5:]
    
    random.shuffle(query_list)

    for i in range(75):
        NEW_FEATURE[f"query{i}"].append(query_list[i])

df_output = pd.DataFrame.from_dict(NEW_FEATURE)
# Output result to csv
print(df_output)
df_output.to_csv(f'../hw4_data/mini/val_testcase_10shot.csv', index=False)

