import os
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = "data_recortado/dataset_5_recortado/val"
modelo = "modelo_min5_dataset_frozen_recortado.pth"

classes = sorted(os.listdir(dataset))

class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(classes)
}

modelo = InceptionResnetV1(
    pretrained='vggface2',
    classify=True
).to(DEVICE)

modelo.logits = nn.Linear(512, len(classes)).to(DEVICE)

modelo.load_state_dict(
    torch.load(modelo, map_location=DEVICE)
)

modelo.eval()

transformacoes = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5]*3,
        std=[0.5]*3
    )
])

corretas = []
incorretas = []

for pessoa in classes:

    pasta_pessoa = os.path.join(dataset, pessoa)

    for nome_img in os.listdir(pasta_pessoa):

        caminho = os.path.join(pasta_pessoa, nome_img)

        try:
            img = Image.open(caminho).convert('RGB')

            input_tensor = transformacoes(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = modelo(input_tensor)

            pred = output.argmax(dim=1).item()

            label_real = class_to_idx[pessoa]

            if pred == label_real:
                corretas.append(caminho)
            else:
                incorretas.append(caminho)

        except:
            pass

print(f"Corretas: {len(corretas)}")
print(f"Incorretas: {len(incorretas)}")