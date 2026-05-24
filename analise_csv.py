import pandas as pd

ficheiros = {
    "min5_frozen": "resultados/resultados_explicabilidade_dataset_5_frozen.csv",
    "min5_ft": "resultados/resultados_explicabilidade_dataset_5_FT.csv",
    "min10_frozen": "resultados/resultados_explicabilidade_dataset_10_frozen.csv",
    "min10_ft": "resultados/resultados_explicabilidade_dataset_10_FT.csv",

    "min5_frozen_recortado": "resultados/resultados_explicabilidade_dataset_5_frozen_recortado.csv",
    "min5_ft_recortado": "resultados/resultados_explicabilidade_dataset_5_FT_recortado.csv",
    "min10_frozen_recortado": "resultados/resultados_explicabilidade_dataset_10_frozen_recortado.csv",
    "min10_ft_recortado": "resultados/resultados_explicabilidade_dataset_10_FT_recortado.csv"
}

resultados = []

for nome_modelo, ficheiro in ficheiros.items():

    df = pd.read_csv(ficheiro)

    corretas = df[df["correta"] == 1]
    incorretas = df[df["correta"] == 0]

    # médias das corretas
    olhos_c = corretas["olhos"].mean()
    nariz_c = corretas["nariz"].mean()
    boca_c = corretas["boca"].mean()
    fundo_c = corretas["fundo"].mean()

    # médias das incorretas
    olhos_i = incorretas["olhos"].mean()
    nariz_i = incorretas["nariz"].mean()
    boca_i = incorretas["boca"].mean()
    fundo_i = incorretas["fundo"].mean()

    # cara total
    face_total_c = olhos_c + nariz_c + boca_c
    face_total_i = olhos_i + nariz_i + boca_i

    # sem ser a cara
    nao_face_c = 100 - face_total_c
    nao_face_i = 100 - face_total_i

    resultado = {
        "modelo": nome_modelo,

        "accuracy": df["correta"].mean() * 100,

        "num_corretas": len(corretas),
        "num_incorretas": len(incorretas),

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