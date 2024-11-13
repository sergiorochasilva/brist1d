import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

df = pd.read_csv("dataset/train.csv", header=0)
print(df.shape)
df.head(5)

cols = ["cals-0:00", "bg+1:00"]

df_limpo = df.dropna(subset=cols)

selected_lags = []
start_lag = 1  # including it
final_lag = 73  # excluding it

# for each (feature, target), instead of 'chicken' and 'egg'
for lag in range(start_lag, final_lag):
    result = grangercausalitytests(df_limpo[cols], maxlag=iter([lag]))
    pvalue = result.get(lag)[0].get("ssr_ftest")[1]
    if (
        pvalue < 0.05
    ):  # coloquei 5%, mas esse valor pode ser ajustado. Quando menor, menos lags serão selecionados
        selected_lags.append(lag)
print(selected_lags)
