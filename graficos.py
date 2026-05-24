import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

resultados = "resultados"
graficos = "graficos_resultados"

os.makedirs(graficos, exist_ok=True)

df = pd.read_csv(
    os.path.join(
        resultados,
        "resumo_explicabilidade.csv"
    )
)

nomes = [
    "Min5\nFrozen",
    "Min5\nFT",
    "Min10\nFrozen",
    "Min10\nFT",
    "Min5\nFrozen\nRec",
    "Min5\nFT\nRec",
    "Min10\nFrozen\nRec",
    "Min10\nFT\nRec"
]

x = np.arange(len(nomes))

# accuracy 

plt.figure(figsize=(12,6))

bars = plt.bar(
    x,
    df["accuracy"]
)

plt.xticks(x, nomes)

plt.ylabel("Accuracy (%)")

plt.title(
    "Accuracy dos Modelos",
    fontsize=14,
    fontweight='bold'
)

plt.ylim(0,100)

for bar in bars:

    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        y + 1,
        f"{y:.1f}%",
        ha='center'
    )

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/01_accuracy_modelos.png",
    dpi=300
)

plt.close()

# cara vs fundo
face = df["face_total_corretas"]
fundo = df["nao_face_corretas"]

plt.figure(figsize=(12,6))

plt.bar(
    x,
    face,
    label="Face Interna"
)

plt.bar(
    x,
    fundo,
    bottom=face,
    label="Face Externa / Fundo"
)

plt.xticks(x, nomes)

plt.ylabel("Ativação Média (%)")

plt.title(
    "Distribuição da Atenção do Modelo",
    fontsize=14,
    fontweight='bold'
)

plt.ylim(0,100)

plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/02_face_vs_fundo.png",
    dpi=300
)

plt.close()

# regiões da cara
olhos = df["olhos_corretas"]
nariz = df["nariz_corretas"]
boca = df["boca_corretas"]

plt.figure(figsize=(12,6))

plt.bar(
    x,
    olhos,
    label="Olhos"
)

plt.bar(
    x,
    nariz,
    bottom=olhos,
    label="Nariz"
)

plt.bar(
    x,
    boca,
    bottom=olhos + nariz,
    label="Boca"
)

plt.xticks(x, nomes)

plt.ylabel("Ativação Média (%)")

plt.title(
    "Distribuição da Atenção Facial",
    fontsize=14,
    fontweight='bold'
)

plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/03_regioes_faciais.png",
    dpi=300
)

plt.close()

# imagens corretas vs incorretas
w = 0.35

plt.figure(figsize=(12,6))

plt.bar(
    x - w/2,
    df["face_total_corretas"],
    width=w,
    label="Corretas"
)

plt.bar(
    x + w/2,
    df["face_total_incorretas"],
    width=w,
    label="Incorretas"
)

plt.xticks(x, nomes)

plt.ylabel("Ativação Facial (%)")

plt.title(
    "Corretas vs Incorretas",
    fontsize=14,
    fontweight='bold'
)

plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/04_corretas_vs_incorretas.png",
    dpi=300
)

plt.close()
# accuracy vs atenção
plt.figure(figsize=(8,6))

plt.scatter(
    df["face_total_corretas"],
    df["accuracy"],
    s=120
)

for i, nome in enumerate(nomes):

    plt.text(
        df["face_total_corretas"][i] + 0.2,
        df["accuracy"][i],
        nome.replace("\n"," ")
    )

plt.xlabel("Ativação Facial (%)")

plt.ylabel("Accuracy (%)")

plt.title(
    "Accuracy vs Atenção Facial",
    fontsize=14,
    fontweight='bold'
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/05_accuracy_vs_face.png",
    dpi=300
)

plt.close()

# fundo corretas vs incorretas
plt.figure(figsize=(12,6))

plt.bar(
    x - w/2,
    df["nao_face_corretas"],
    width=w,
    label="Corretas"
)

plt.bar(
    x + w/2,
    df["nao_face_incorretas"],
    width=w,
    label="Incorretas"
)

plt.xticks(x, nomes)

plt.ylabel("Ativação Face Externa/Fundo (%)")

plt.title(
    "Dependência de Contexto",
    fontsize=14,
    fontweight='bold'
)

plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{graficos}/06_dependencia_contexto.png",
    dpi=300
)

plt.close()