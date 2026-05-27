import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch import nn

from facenet_pytorch import InceptionResnetV1

from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

dataset = "data/dataset_10/val"

caminho_modelo = "modelo_min10_dataset_FT.pth"

imagem = (
    "data/dataset_10/val/"
    "Carlos_Moya/"
    "Carlos_Moya_0001.jpg"
)

classes = sorted([
    pasta
    for pasta in os.listdir(dataset)
])

modelo = InceptionResnetV1(
    pretrained='vggface2',
    classify=True
).to(DEVICE)

modelo.logits = nn.Linear(
    512,
    len(classes)
).to(DEVICE)

modelo.load_state_dict(
    torch.load(
        caminho_modelo,
        map_location=DEVICE
    )
)

modelo.eval()

transformacoes = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5] * 3,
        std=[0.5] * 3
    )
])
# função utilizada pelo LIME para obter probabilidades de classificação
def predict(images):

    batch = []

    for img in images:

        pil = Image.fromarray(
            img.astype(np.uint8)
        )

        tensor = transformacoes(pil)

        batch.append(tensor)

    batch = torch.stack(batch).to(DEVICE)

    with torch.no_grad():

        outputs = modelo(batch)

        probs = torch.softmax(
            outputs,
            dim=1
        )

    return probs.cpu().numpy()

img = Image.open(
    imagem
).convert("RGB")

img_np = np.array(img)

input_tensor = transformacoes(img)

input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

#obtém a previsão do modelo
with torch.no_grad():

    output = modelo(input_tensor)

    pred = output.argmax(dim=1).item()

    conf = torch.softmax(
        output,
        dim=1
    )[0][pred].item()

# mostrar a classe prevista e a confiança
print(
    f"\nClasse prevista: {classes[pred]}"
)

print(
    f"Confiança: {conf:.4f}"
)
# inicializar explicador LIME
explainer = lime_image.LimeImageExplainer()
# gera a explicação
explanation = explainer.explain_instance(
    img_np,
    predict,
    top_labels=1,
    hide_color=0,
    num_samples=5000,
    segmentation_fn=lambda x: slic(
        x,
        n_segments=100,
        compactness=10,
        sigma=1
    )
)
# obtem as regiões mais importantes
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=20,
    hide_rest=False
)

os.makedirs(
    "lime",
    exist_ok=True
)

plt.figure(figsize=(8,8))

# desenha os limites dos superpixels relevantes
plt.imshow(
    mark_boundaries(
        temp / 255.0,
        mask
    )
)

plt.title(
    f"LIME - {classes[pred]} "
    f"({conf:.2%})"
)

plt.axis("off")

plt.tight_layout()

output_path = os.path.join(
    "lime",
    "lime_Carlos_Moya.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()