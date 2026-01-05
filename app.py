# app.py
import pandas as pd
import streamlit as st
from data import load_datalake_xlsx
from model import ProfileModel

st.set_page_config(page_title="Mega-Sena Lab", layout="wide")

st.title("Mega-Sena Lab (Acadêmico)")

xlsx = st.sidebar.text_input("Caminho do datalake Excel", "datalake_megasena.xlsx")
window_recent = st.sidebar.slider("Janela recente", 50, 1000, 300, step=50)
n_samples = st.sidebar.slider("Amostras (candidatos)", 50_000, 500_000, 200_000, step=50_000)
top_k = st.sidebar.slider("Top K palpites", 5, 100, 20, step=5)
seed = st.sidebar.number_input("Seed", value=123, step=1)

df = load_datalake_xlsx(xlsx)

last = df.iloc[-1]
last_nums = last["nums"]
next_concurso = int(last["concurso"]) + 1

st.subheader("Último concurso no datalake")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Concurso", int(last["concurso"]))
c2.metric("Data", str(last["data_sorteio"].date()) if pd.notna(last["data_sorteio"]) else "-")
c3.metric("Acumulado", bool(last["acumulado"]) if pd.notna(last["acumulado"]) else False)
c4.metric("Dezenas", " - ".join(map(str, last_nums)))

model = ProfileModel(df, window_recent=window_recent)

if st.button("Gerar previsões"):
    best = model.suggest(last_nums, n_samples=n_samples, top_k=top_k, seed=seed)
    out = []
    for rank, (sc, nums, feats) in enumerate(best, start=1):
        out.append({
            "rank": rank,
            "concurso_previsto": next_concurso,
            "nums": ",".join(map(str, nums)),
            "score": sc,
            **feats,
        })
    pred = pd.DataFrame(out)

    st.subheader("Top previsões")
    st.dataframe(pred, use_container_width=True)

    st.download_button(
        "Baixar previsões (CSV)",
        pred.to_csv(index=False).encode("utf-8"),
        file_name=f"predicoes_concurso_{next_concurso}.csv",
        mime="text/csv"
    )