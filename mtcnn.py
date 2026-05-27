import os
from facenet_pytorch import MTCNN
from PIL import Image

# Inicializar o detetor MTCNN
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    select_largest=True,
    device='cuda'  # Mudem para 'cuda' se tiverem a placa gráfica configurada
)

# Definir as pastas principais
pasta_origem = "data/dataset_10/"  # A pasta principal onde estão as subpastas das pessoas
pasta_destino_base = "data/dataset_10_recortado"  # Onde o novo dataset vai ser criado

print(f"A iniciar o processamento da pasta: {pasta_origem} ...\n")

# Percorrer todas as pastas, subpastas e ficheiros
for root, dirs, files in os.walk(pasta_origem):
    for ficheiro in files:
        # Verificar se é realmente uma imagem
        if ficheiro.lower().endswith(('.png', '.jpg', '.jpeg')):

            caminho_original = os.path.join(root, ficheiro)

            # Descobrir qual é a subpasta atual (ex: "pessoa_A") para replicar no destino
            caminho_relativo = os.path.relpath(root, pasta_origem)
            pasta_destino = os.path.join(pasta_destino_base, caminho_relativo)

            # Criar a pasta de destino se ela ainda não existir
            if not os.path.exists(pasta_destino):
                os.makedirs(pasta_destino)

            caminho_destino = os.path.join(pasta_destino, ficheiro)

            try:
                # Abrir a imagem original
                img = Image.open(caminho_original).convert('RGB')

                # O MTCNN processa a imagem e guarda automaticamente
                face_tensor = mtcnn(img, save_path=caminho_destino)

            except Exception as e:
                print(f"[ERRO] Falha ao processar {caminho_original}: {e}")

print("\nProcessamento concluído!")