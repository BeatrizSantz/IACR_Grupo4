import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
import time
import pandas as pd 
# CONFIGURAÇÕES
historico = []

DIR_TREINO = 'data/dataset_10/train'
DIR_VALIDACAO = 'data/dataset_10/val'

BATCH_SIZE = 32
EPOCAS = 50
paciencia = 5
melhor_val_loss = float('inf')
contador_paciencia = 0

#Configurar Placa Gráfica (CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'A usar o dispositivo: {device}')

#PREPARAÇÃO DOS DADOS
transformacoes = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("\nA carregar as imagens...")
dataset_treino = datasets.ImageFolder(DIR_TREINO, transform=transformacoes)
dataset_val = datasets.ImageFolder(DIR_VALIDACAO, transform=transformacoes)

num_classes = len(dataset_treino.classes)
print(f'Total de pessoas diferentes a classificar: {num_classes}')
print(f'Imagens de treino: {len(dataset_treino)}')
print(f'Imagens de validação: {len(dataset_val)}')

loader_treino = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# PREPARAR O MODELO
print("\nA carregar o InceptionResnetV1 (FaceNet)...")
modelo = InceptionResnetV1(pretrained='vggface2', classify=True).to(device)

# Congelar ou não as camadas iniciais
for param in modelo.parameters():
    param.requires_grad = False

modelo.logits = nn.Linear(512, num_classes).to(device)

# CONFIGURAR TREINO
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.logits.parameters(), lr=0.001)

# CICLO DE TREINO E VALIDAÇÃO

for epoca in range(EPOCAS):
    inicio_epoca = time.time()

    # FASE DE TREINO
    modelo.train()
    loss_treino = 0.0
    acertos_treino = 0

    for imagens, labels in loader_treino:
        imagens, labels = imagens.to(device), labels.to(device)

        otimizador.zero_grad()
        outputs = modelo(imagens)
        loss = criterio(outputs, labels)

        loss.backward()
        otimizador.step()

        loss_treino += loss.item() * imagens.size(0)
        _, previsoes = torch.max(outputs, 1)
        acertos_treino += torch.sum(previsoes == labels.data).item() # .item() para somar como número

    acc_treino = acertos_treino / len(dataset_treino)
    loss_media_treino = loss_treino / len(dataset_treino)

    # FASE DE VALIDAÇÃO
    modelo.eval()
    loss_val = 0.0
    acertos_val = 0

    with torch.no_grad():
        for imagens, labels in loader_val:
            imagens, labels = imagens.to(device), labels.to(device)

            outputs = modelo(imagens)
            loss = criterio(outputs, labels)

            loss_val += loss.item() * imagens.size(0)
            _, previsoes = torch.max(outputs, 1)
            acertos_val += torch.sum(previsoes == labels.data).item()
        # Cálculos finais da época
        acc_val = acertos_val / len(dataset_val)
        loss_media_val = loss_val / len(dataset_val)

        # Guardar no histórico para o CSV
        historico.append({
            'epoca': epoca + 1,
            'treino_loss': loss_media_treino,
            'treino_acc': acc_treino,
            'val_loss': loss_media_val,
            'val_acc': acc_val
        })

        tempo_epoca = time.time() - inicio_epoca
        print(f'Época {epoca + 1}/{EPOCAS} [{tempo_epoca:.1f}s]')
        print(f'  Treino    -> Loss: {loss_media_treino:.4f} | Accuracy: {acc_treino:.4f}')
        print(f'  Validação -> Loss: {loss_media_val:.4f} | Accuracy: {acc_val:.4f}')

    if loss_media_val < melhor_val_loss:
        melhor_val_loss = loss_media_val
        contador_paciencia = 0
    else:
        contador_paciencia += 1
        print(f"  [Paciência: {contador_paciencia}/{paciencia}]")

    if contador_paciencia >= paciencia:
        print(f"\nTreino interrompido por Early Stopping na época {epoca + 1}!")
        break

df = pd.DataFrame(historico)
NOME_TESTE = "min10_dataset_frozen"
caminho_csv = f"log_{NOME_TESTE}.csv"
df.to_csv(caminho_csv, index=False)
caminho_modelo = f"modelo_{NOME_TESTE}.pth"
torch.save(modelo.state_dict(), caminho_modelo)