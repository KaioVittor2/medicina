import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df["overweight"] = df["weight"] / (df["height"] / 100) ** 2 
df['overweight'] = (df["overweight"] > 25).astype(int)

# 3

# Aqui vamos usar Numpy, para facilitar essa normalização, o código faz os valores 1 virarem 0 e acima de 1 virarem 1
df["cholesterol"] = np.where(df["cholesterol"] == 1, 0, 1)

# Vai fazer a mesma coisa, verificar a condição, se for true substituir por 0 e se for falsa por 1
df["gluc"] = np.where(df["gluc"] == 1, 0, 1)

df = df.abs()

# 4
def draw_cat_plot():
    # 5 - Converto o DataFrame para o formato longo (long format)
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "overweight"], 
                     var_name="variable", value_name="Value")

    # 6 - Crio o gráfico categórico (catplot)
    df_cat = sns.catplot(x="variable", hue="Value", col="cardio", data=df_cat, kind="count")

    # 7 - Personalização do gráfico, se necessário (por exemplo, ajustando labels ou cores)
    df_cat.set_axis_labels("variable", "total")
    df_cat.set_titles("Cardio: {col_name}")
    df_cat.despine(left=True)

    # 8 - Criação da figura
    fig = df_cat.fig

    # 9 - Salvo o gráfico em um arquivo
    fig.savefig('catplot.png')
    
    return fig


# 10
def draw_heat_map():
    # 11 - Calcular a matriz de correlação
    df_heat = df.corr()

    # 12 - Criar uma máscara para o triângulo superior
    corr = df_heat
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13 - Definir o tamanho da figura e o layout
    fig, ax = plt.subplots(figsize=(10, 8))

    # 14 - Plotar o heatmap com seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", square=True, linewidths=.5, ax=ax)

    # 15 - Ajustes finais no gráfico (opcional)
    ax.set_title("Heatmap da Correlação")

    # 16 - Salvar o gráfico em um arquivo
    fig.savefig('heatmap.png')
    
    return fig
