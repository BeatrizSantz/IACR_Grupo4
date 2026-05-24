import os
import csv
import cv2
import shutil
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch import nn

from facenet_pytorch import InceptionResnetV1

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = [

    {
        "nome": "min5_frozen",
        "dataset": "data/dataset_5/val",
        "modelo": "modelo_min5_dataset_frozen.pth"
    },

    {
        "nome": "min5_FT",
        "dataset": "data/dataset_5/val",
        "modelo": "modelo_min5_dataset_FT.pth"
    },

    {
        "nome": "min10_frozen",
        "dataset": "data/dataset_10/val",
        "modelo": "modelo_min10_dataset_frozen.pth"
    },

    {
        "nome": "min10_FT",
        "dataset": "data/dataset_10/val",
        "modelo": "modelo_min10_dataset_FT.pth"
    },

    # Dataset recortado

    {
        "nome": "min5_frozen_recortado",
        "dataset": "data_recortado/dataset_5_recortado/val",
        "modelo": "modelo_min5_dataset_frozen_recortado.pth"
    },

    {
        "nome": "min5_FT_recortado",
        "dataset": "data_recortado/dataset_5_recortado/val",
        "modelo": "modelo_min5_dataset_FT_recortado.pth"
    },

    {
        "nome": "min10_frozen_recortado",
        "dataset": "data_recortado/dataset_10_recortado/val",
        "modelo": "modelo_min10_dataset_frozen_recortado.pth"
    },

    {
        "nome": "min10_FT_recortado",
        "dataset": "data_recortado/dataset_10_recortado/val",
        "modelo": "modelo_min10_dataset_FT_recortado.pth"
    }

]

transformacoes = transforms.Compose([
    transforms.Resize((160, 160)), # todas as imagens ficam 160x160
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5]*3,
        std=[0.5]*3
    )
])

os.makedirs("resultados", exist_ok=True) # pasta csvs
os.makedirs("heatmaps", exist_ok=True) # pasta heatmaps

for config in configs:

    caminho_dataset = config["dataset"]
    caminho_modelo = config["modelo"]

    output_csv = f"resultados/resultados_explicabilidade_{config['nome']}.csv"

    heatmap_dir = f"heatmaps/{config['nome']}"

    if os.path.exists(heatmap_dir): # remove pasta antiga se existir
        shutil.rmtree(heatmap_dir)

    os.makedirs(heatmap_dir) # cria pasta nova

    classes = sorted(os.listdir(caminho_dataset)) # cada pasta corresponde a uma pessoa

    class_to_idx = { # nome da pessoa corresponde a um indice numérico
        cls_name: idx
        for idx, cls_name in enumerate(classes)
    }

    modelo = InceptionResnetV1(
        pretrained='vggface2',
        classify=True
    ).to(DEVICE)

    modelo.logits = nn.Linear(
        512,
        len(classes)
    ).to(DEVICE)

    modelo.load_state_dict( # carrega os pesos treinados
        torch.load(
            caminho_modelo,
            map_location=DEVICE
        )
    )

    modelo.eval()

    target_layers = [modelo.block8.branch1]

    cam = GradCAM(
        model=modelo,
        target_layers=target_layers
    )

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)

        writer.writerow([
            "imagem",
            "real",
            "classe_prevista",
            "confianca",
            "correta",
            "olhos",
            "nariz",
            "boca",
            "fundo"
        ])

        contador = 0
        corretas = 0
        incorretas = 0

        for pessoa in classes:

            pasta_pessoa = os.path.join(
                caminho_dataset,
                pessoa
            )

            if not os.path.isdir(pasta_pessoa):
                continue

            for nome_img in os.listdir(pasta_pessoa):

                caminho = os.path.join(
                    pasta_pessoa,
                    nome_img
                )

                try:

                    img = Image.open(caminho).convert('RGB')

                    input_tensor = transformacoes(img)
                    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

                    output = modelo(input_tensor)

                    classe_prevista = output.argmax(dim=1).item()

                    classe_verdadeira = class_to_idx[pessoa]

                    prob = torch.softmax(output, dim=1)

                    confianca = prob[0][classe_prevista].item()

                    correta = int(classe_prevista == classe_verdadeira) # verfica se acertou 

                    if correta:
                        corretas += 1
                    else:
                        incorretas += 1

                    targets = [ClassifierOutputTarget(classe_prevista)]

                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets=targets
                    )

                    cam_map = grayscale_cam[0]

                    img_resized = img.resize((160,160))

                    img_np = np.array(img_resized).astype(np.float32) / 255.0

                    visualization = show_cam_on_image(
                        img_np,
                        cam_map,
                        use_rgb=True
                    )

                    tipo = "corretas" if correta else "incorretas"

                    pasta_saida = os.path.join(
                        heatmap_dir,
                        tipo,
                        pessoa
                    )

                    os.makedirs(
                        pasta_saida,
                        exist_ok=True
                    )

                    caminho_saida = os.path.join(
                        pasta_saida,
                        nome_img
                    )

                    cv2.imwrite(
                        caminho_saida,
                        cv2.cvtColor(
                            visualization,
                            cv2.COLOR_RGB2BGR
                        )
                    )

                    h, w = cam_map.shape

                    mask_olhos = np.zeros((h, w))
                    mask_nariz = np.zeros((h, w))
                    mask_boca = np.zeros((h, w))
                    mask_fundo = np.ones((h, w))

                    # olhos
                    mask_olhos[30:70, 35:125] = 1

                    # nariz
                    mask_nariz[65:105, 55:105] = 1

                    # boca
                    mask_boca[100:140, 45:115] = 1

                    # fundo
                    mask_fundo -= (
                        mask_olhos +
                        mask_nariz +
                        mask_boca
                    )

                    mask_fundo = np.clip(
                        mask_fundo,
                        0,
                        1
                    )
                    # ativações
                    olhos_score = np.sum(
                        cam_map * mask_olhos
                    )

                    nariz_score = np.sum(
                        cam_map * mask_nariz
                    )

                    boca_score = np.sum(
                        cam_map * mask_boca
                    )

                    fundo_score = np.sum(
                        cam_map * mask_fundo
                    )

                    total = (
                        olhos_score +
                        nariz_score +
                        boca_score +
                        fundo_score
                    )

                    # evitar divisão por zero
                    if total == 0:
                        continue

                    olhos_pct = 100 * olhos_score / total
                    nariz_pct = 100 * nariz_score / total
                    boca_pct = 100 * boca_score / total
                    fundo_pct = 100 * fundo_score / total

                    writer.writerow([
                        caminho,
                        pessoa,
                        classes[classe_prevista],
                        confianca,
                        correta,
                        olhos_pct,
                        nariz_pct,
                        boca_pct,
                        fundo_pct
                    ])

                    contador += 1

                    if contador % 50 == 0:

                        print(
                            f"{contador} imagens processadas"
                        )

                except Exception as e:

                    print(f"\nErro em: {caminho}")
                    print(e)

    total_imgs = corretas + incorretas

    accuracy = 100 * corretas / total_imgs

    print(f"Total: {total_imgs}")
    print(f"Corretas: {corretas}")
    print(f"Incorretas: {incorretas}")
    print(f"Accuracy: {accuracy:.2f}%")

