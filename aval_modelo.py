import os
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = "data/dataset_5/val"
MODEL_PATH = "modelo_min5_dataset_frozen.pth"

classes = sorted(os.listdir(DATASET_PATH))

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
    torch.load(MODEL_PATH, map_location=DEVICE)
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

correct = []
incorrect = []

for pessoa in classes:

    pasta_pessoa = os.path.join(DATASET_PATH, pessoa)

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
                correct.append(caminho)
            else:
                incorrect.append(caminho)

        except:
            pass

print(f"Corretas: {len(correct)}")
print(f"Incorretas: {len(incorrect)}")