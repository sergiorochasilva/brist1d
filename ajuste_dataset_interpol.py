import pandas as pd

# import numpy as np


# Função para aplicar a média dos 5 valores mais próximos
def preencher_nan_com_interpolacao(df, coluna_valor, num_vizinhos=5):
    # Para cada índice `NaN`, encontra os vizinhos e calcula a média
    lista_nulls = df[df[coluna_valor].isna()].index
    print("Total", coluna_valor, len(lista_nulls))
    i = 0
    for idx in lista_nulls:
        i += 1
        print(i)

        # Seleciona os índices dos vizinhos mais próximos
        vizinhos_idx = df[
            (~df[coluna_valor].isna()) & (df["p_num"] == df.iloc[idx]["p_num"])
        ].index
        vizinhos_idx_proximos = vizinhos_idx[
            (vizinhos_idx > idx - num_vizinhos // 2)
            & (vizinhos_idx <= idx + num_vizinhos // 2)
        ]

        # Calcula a média dos valores dos vizinhos
        if len(vizinhos_idx_proximos) > 0:
            print("Corrigido, vizinhos usados", len(vizinhos_idx_proximos))
            df.at[idx, coluna_valor] = df.loc[
                vizinhos_idx_proximos, coluna_valor
            ].mean()

    return df


ds = pd.read_csv("dataset/train_corrigido.csv", header=0)

cols = [
    ("bg-0:00", 7),
    # ("insulin-0:00", 100000),
    # ("hr-0:00", 1000),
    ("cals-0:00", 7),
]

for col, qtd in cols:
    preencher_nan_com_interpolacao(ds, col, qtd)

ds.to_csv("dataset/train_corrigido_interpol.csv", index=False)
