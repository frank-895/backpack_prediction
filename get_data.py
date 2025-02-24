## Let's use Kaggle API to make this run on others computers

import pandas as pd

df = pd.read_csv("train.csv")
df_extra = pd.read_csv("training_extra.csv")

df.drop('id', axis=1, inplace=True)
df_extra.drop('id', axis=1, inplace=True)

df_final = pd.concat([df, df_extra])
df_final.to_csv("train_final.csv")