import os
import csv
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch import nn

from facenet_pytorch import InceptionResnetV1

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = "data/dataset_5/val"

MODEL_PATH = "modelo_min5_dataset_frozen.pth"

CSV_OUTPUT = "resultados_explicabilidade_dataset_5_frozen.csv"

classes = sorted(os.listdir(DATASET_PATH))

class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(classes)
}

transformacoes = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

modelo = InceptionResnetV1(
    pretrained='vggface2',
    classify=True
).to(DEVICE)

modelo.logits = nn.Linear(512, len(classes)).to(DEVICE)

modelo.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

modelo.eval()

target_layers = [modelo.block8.branch1]

cam = GradCAM(
    model=modelo,
    target_layers=target_layers
)

with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as file:

    writer = csv.writer(file)

    writer.writerow([
        "imagem",
        "real",
        "pred",
        "confidence",
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

        pasta_pessoa = os.path.join(DATASET_PATH, pessoa)

        if not os.path.isdir(pasta_pessoa):
            continue

        for nome_img in os.listdir(pasta_pessoa):

            caminho = os.path.join(pasta_pessoa, nome_img)

            try:

                img = Image.open(caminho).convert('RGB')

                input_tensor = transformacoes(img).unsqueeze(0).to(DEVICE)

                output = modelo(input_tensor)

                pred = output.argmax(dim=1).item()

                label_real = class_to_idx[pessoa]

                prob = torch.softmax(output, dim=1)

                confidence = prob[0][pred].item()

                correta = int(pred == label_real)

                if correta:
                    corretas += 1
                else:
                    incorretas += 1

                targets = [ClassifierOutputTarget(pred)]

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets
                )

                cam_map = grayscale_cam[0]

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

                mask_fundo = np.clip(mask_fundo, 0, 1)

                olhos_score = np.sum(cam_map * mask_olhos)
                nariz_score = np.sum(cam_map * mask_nariz)
                boca_score = np.sum(cam_map * mask_boca)
                fundo_score = np.sum(cam_map * mask_fundo)

                total = (
                    olhos_score +
                    nariz_score +
                    boca_score +
                    fundo_score
                )

                olhos_pct = 100 * olhos_score / total
                nariz_pct = 100 * nariz_score / total
                boca_pct = 100 * boca_score / total
                fundo_pct = 100 * fundo_score / total

                writer.writerow([
                    caminho,
                    pessoa,
                    classes[pred],
                    confidence,
                    correta,
                    olhos_pct,
                    nariz_pct,
                    boca_pct,
                    fundo_pct
                ])

                contador += 1

                if contador % 50 == 0:
                    print(f"{contador} imagens processadas")

            except Exception as e:

                print(f"Erro em {caminho}")
                print(e)

accuracy = 100 * corretas / (corretas + incorretas)

print("RESULTADOS FINAIS")

print(f"Total: {corretas + incorretas}")
print(f"Corretas: {corretas}")
print(f"Incorretas: {incorretas}")
print(f"Accuracy: {accuracy:.2f}%")

print(f"\nCSV guardado em: {CSV_OUTPUT}")