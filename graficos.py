import os
import pandas as pd
import matplotlib.pyplot as plt

PASTA_RESULTADOS = "resultados"

PASTA_GRAFICOS = "graficos"

os.makedirs(PASTA_GRAFICOS, exist_ok=True)

csvs = [
    f for f in os.listdir(PASTA_RESULTADOS)
    if f.endswith(".csv")
    and "resultados_explicabilidade" in f
]

print(f"{len(csvs)} CSVs")

def guardar(nome):

    caminho = os.path.join(PASTA_GRAFICOS, nome)

    plt.tight_layout()
    plt.savefig(caminho, dpi=300)

    plt.close()

for csv_file in csvs:

    caminho_csv = os.path.join(PASTA_RESULTADOS, csv_file)

    print(f"\nProcessando: {csv_file}")

    df = pd.read_csv(caminho_csv)

    nome = csv_file.replace(".csv", "")

    if "correta" in df.columns:

        accuracy = df["correta"].mean() * 100

        plt.figure(figsize=(5,5))

        plt.bar(["Accuracy"], [accuracy])

        plt.ylim(0, 100)

        plt.ylabel("Percentagem (%)")

        plt.title(f"Accuracy\n{nome}")

        guardar(f"{nome}_accuracy.png")

        print(f"Accuracy: {accuracy:.2f}%")

    regioes = ["olhos", "nariz", "boca", "fundo"]

    if all(r in df.columns for r in regioes):

        medias = [df[r].mean() for r in regioes]

        plt.figure(figsize=(8,5))

        plt.bar(regioes, medias)

        plt.ylabel("Ativação Média")

        plt.title(f"Atenção por Região\n{nome}")

        guardar(f"{nome}_regioes.png")

        print("Gráfico de regiões criado")

    if "confidence" in df.columns:

        plt.figure(figsize=(8,5))

        plt.hist(df["confidence"], bins=20)

        plt.xlabel("Confidence")

        plt.ylabel("Número de Imagens")

        plt.title(f"Distribuição da Confiança\n{nome}")

        guardar(f"{nome}_confidence.png")

        print("Histograma criado")

    if "fundo" in df.columns and "correta" in df.columns:

        corretos = df[df["correta"] == 1]
        erros = df[df["correta"] == 0]

        if len(corretos) > 0 and len(erros) > 0:

            valores = [
                corretos["fundo"].mean(),
                erros["fundo"].mean()
            ]

            labels = ["Corretos", "Incorretos"]

            plt.figure(figsize=(6,5))

            plt.bar(labels, valores)

            plt.ylabel("Ativação Média no Fundo")

            plt.title(f"Dependência do Fundo\n{nome}")

            guardar(f"{nome}_fundo.png")

            print("Gráfico do fundo criado")

