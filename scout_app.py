import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Simulação de dados
data = {
    "nome": ["João Silva", "Carlos Lima", "Pedro Santos", "Lucas Costa", "Rafael Souza"],
    "idade": [19, 22, 21, 20, 23],
    "posicao": ["Zagueiro", "Meia", "Atacante", "Lateral", "Volante"],
    "gols": [3, 5, 12, 1, 2],
    "assistencias": [1, 7, 4, 3, 5],
    "minutos_jogados": [2345, 1980, 2670, 1880, 2100],
    "km_percorridos": [118, 109, 130, 105, 112],
    "passes_certos": [1325, 1542, 1120, 1450, 1480]
}

df = pd.DataFrame(data)

# Normalização
scaler = MinMaxScaler()
atributos = ["gols", "assistencias", "minutos_jogados", "km_percorridos", "passes_certos"]
df_norm = df.copy()
df_norm[atributos] = scaler.fit_transform(df[atributos])

# Score ponderado por posição
pesos = {
    "Zagueiro": {"gols": 0.1, "assistencias": 0.05, "minutos_jogados": 0.3, "km_percorridos": 0.25, "passes_certos": 0.3},
    "Meia": {"gols": 0.2, "assistencias": 0.3, "minutos_jogados": 0.2, "km_percorridos": 0.15, "passes_certos": 0.15},
    "Atacante": {"gols": 0.4, "assistencias": 0.25, "minutos_jogados": 0.15, "km_percorridos": 0.1, "passes_certos": 0.1},
    "Lateral": {"gols": 0.1, "assistencias": 0.2, "minutos_jogados": 0.25, "km_percorridos": 0.3, "passes_certos": 0.15},
    "Volante": {"gols": 0.1, "assistencias": 0.15, "minutos_jogados": 0.25, "km_percorridos": 0.2, "passes_certos": 0.3}
}

scores = []
for idx, row in df.iterrows():
    pos = row["posicao"]
    p = pesos[pos]
    score = sum(df_norm.loc[idx, attr] * peso for attr, peso in p.items())
    scores.append(score)

df["score_ponderado"] = scores

# Classificação
df["classificacao"] = pd.cut(df["score_ponderado"],
    bins=[0, 0.4, 0.6, 0.8, 1.0],
    labels=["Abaixo do esperado", "Regular", "Potencial alto", "Elite"]
)

# Clusterização
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df_norm[atributos])

# Média da posição
df["media_posicao"] = df.groupby("posicao")["score_ponderado"].transform("mean")

# Regressão para projeção
X = df[["idade", "minutos_jogados"]]
y = df["score_ponderado"]
reg = LinearRegression().fit(X, y)
df["score_projetado"] = reg.predict(X)

# Interface Streamlit
st.title("⚽ Scout Digital Inteligente — Versão Avançada")

nome = st.selectbox("Selecione um jogador", df["nome"])
jogador = df[df["nome"] == nome].iloc[0]

st.markdown(f"### {jogador['nome']} — {jogador['posicao']}")
st.write(f"**Idade:** {jogador['idade']}")
st.write(f"**Classificação atual:** `{jogador['classificacao']}`")
st.write(f"**Score Ponderado:** {jogador['score_ponderado']:.2f}")
st.write(f"**Score Projetado:** {jogador['score_projetado']:.2f}")
st.write(f"**Média da Posição:** {jogador['media_posicao']:.2f}")
st.write(f"**Cluster (estilo):** {jogador['cluster']}")

# Radar
atrib = atributos
valores = df_norm[df["nome"] == nome][atrib].values.flatten().tolist()
valores += [valores[0]]
labels = atrib + [atrib[0]]
angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]

fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
ax.plot(angles, valores, linewidth=2)
ax.fill(angles, valores, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(atrib)
ax.set_yticklabels([])
st.pyplot(fig)
