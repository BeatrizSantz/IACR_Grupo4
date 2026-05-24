import os
import cv2
import matplotlib.pyplot as plt

# pessoa
pessoa = "Jackie_Chan"

# imagem
imagem = "Jackie_Chan_0011.jpg"

# pasta output
pasta_output = "comparacoes"

os.makedirs(pasta_output, exist_ok=True)

modelos = {

    "Min5 Frozen":
    "heatmaps/min5_frozen/corretas",

    "Min5 FT":
    "heatmaps/min5_FT/corretas",

    "Min10 Frozen":
    "heatmaps/min10_frozen/corretas",

    "Min10 FT":
    "heatmaps/min10_FT/corretas",

    "Min5 Frozen Rec":
    "heatmaps/min5_frozen_recortado/corretas",

    "Min5 FT Rec":
    "heatmaps/min5_FT_recortado/corretas",

    "Min10 Frozen Rec":
    "heatmaps/min10_frozen_recortado/corretas",

    "Min10 FT Rec":
    "heatmaps/min10_FT_recortado/corretas"
}

original_path = os.path.join(
    "data/dataset_10/val",
    pessoa,
    imagem
)

original = cv2.imread(original_path)

# verifica se existe
if original is None:

    print("\nERRO:")
    print("Imagem original não encontrada.")

    exit()

# converter BGR em RGB
original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

fig, axs = plt.subplots(
    3,
    3,
    figsize=(16,14)
)

axs = axs.flatten()

axs[0].imshow(original)

axs[0].set_title(
    "Imagem Original",
    fontsize=13,
    fontweight='bold',
    pad=10
)

axs[0].axis("off")

for idx, (nome_modelo, pasta_modelo) in enumerate(modelos.items()):

    caminho_heatmap = os.path.join(
        pasta_modelo,
        pessoa,
        imagem
    )

    ax = axs[idx + 1]

    if not os.path.exists(caminho_heatmap):

        ax.text(
            0.5,
            0.5,
            "Não encontrado",
            ha='center',
            va='center',
            fontsize=12
        )

        ax.set_title(
            nome_modelo,
            fontsize=11
        )

        ax.axis("off")

        continue

    img = cv2.imread(caminho_heatmap)

    # verifica leitura
    if img is None:

        ax.text(
            0.5,
            0.5,
            "Erro ao carregar",
            ha='center',
            va='center',
            fontsize=12
        )

        ax.set_title(nome_modelo)

        ax.axis("off")

        continue

    # converter BGR em RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(img)

    ax.set_title(
        nome_modelo,
        fontsize=11,
        pad=8
    )

    ax.axis("off")

fig.suptitle(

    f"Comparação GradCAM\n{pessoa}",

    fontsize=10,

    fontweight='bold',

    y=0.98
)

plt.subplots_adjust(

    top=0.90,

    wspace=0.15,

    hspace=0.25
)

output_path = os.path.join(
    pasta_output,
    f"{pessoa}_comparacao.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight'
)

print(output_path)

# mostrar figura
plt.show()

# fechar figura
plt.close()