import pandas as pd

files = {
    "min5_frozen": "resultados_explicabilidade_dataset_5_frozen.csv",
    "min5_ft": "resultados_explicabilidade_dataset_5_FT.csv",
    "min10_frozen": "resultados_explicabilidade_dataset_10_frozen.csv",
    "min10_ft": "resultados_explicabilidade_dataset_10_FT.csv"
}

resultados = []

for nome_modelo, ficheiro in files.items():

    df = pd.read_csv(ficheiro)

    correct = df[df["correta"] == 1]
    incorrect = df[df["correta"] == 0]

    # MÉDIAS CORRETAS
    olhos_c = correct["olhos"].mean()
    nariz_c = correct["nariz"].mean()
    boca_c = correct["boca"].mean()
    fundo_c = correct["fundo"].mean()

    # MÉDIAS INCORRETAS
    olhos_i = incorrect["olhos"].mean()
    nariz_i = incorrect["nariz"].mean()
    boca_i = incorrect["boca"].mean()
    fundo_i = incorrect["fundo"].mean()

    # FACE TOTAL
    face_total_c = olhos_c + nariz_c + boca_c
    face_total_i = olhos_i + nariz_i + boca_i

    # NÃO FACE
    nao_face_c = 100 - face_total_c
    nao_face_i = 100 - face_total_i

    resultado = {
        "modelo": nome_modelo,

        "accuracy": df["correta"].mean() * 100,

        "num_corretas": len(correct),
        "num_incorretas": len(incorrect),

        "olhos_corretas": olhos_c,
        "nariz_corretas": nariz_c,
        "boca_corretas": boca_c,
        "fundo_corretas": fundo_c,

        "olhos_incorretas": olhos_i,
        "nariz_incorretas": nariz_i,
        "boca_incorretas": boca_i,
        "fundo_incorretas": fundo_i,

        "face_total_corretas": face_total_c,
        "face_total_incorretas": face_total_i,

        "nao_face_corretas": nao_face_c,
        "nao_face_incorretas": nao_face_i
    }

    resultados.append(resultado)

df_final = pd.DataFrame(resultados)

# arredondar
df_final = df_final.round(2)

# guardar
df_final.to_csv("resumo_explicabilidade.csv", index=False)

print(df_final)