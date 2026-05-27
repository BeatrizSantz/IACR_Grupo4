import pandas as pd
import matplotlib.pyplot as plt

# Dicionário com os títulos para os gráficos e os respetivos nomes dos ficheiros CSV
files = {
    "Min 5 Imagens_rec - Frozen": "log_min5_dataset_frozen_recortado.csv",
    "Min 5 Imagens_rec - Fine-Tuning": "log_min5_dataset_FT_recortado.csv",
    "Min 10 Imagens_rec - Frozen": "log_min10_dataset_frozen_recortado.csv",
    "Min 10 Imagens_rec - Fine-Tuning": "log_min10_dataset_FT_recortado.csv"
}

# Criar uma grelha de 2x2 para os gráficos (tamanho 16x12 para boa legibilidade)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Iterar sobre cada ficheiro e desenhar o respetivo gráfico
for i, (title, file) in enumerate(files.items()):
    # Ler o ficheiro CSV
    df = pd.read_csv(file)
    ax = axes[i]

    # 1. Desenhar a Accuracy (Eixo Y principal - Esquerda)
    ax.plot(df['epoca'], df['treino_acc'], label='Treino Acc', color='blue', linestyle='-')
    ax.plot(df['epoca'], df['val_acc'], label='Validação Acc', color='blue', linestyle='--')
    ax.set_ylabel('Accuracy', color='blue', fontsize=12)
    ax.tick_params(axis='y', labelcolor='blue')

    # 2. Desenhar a Loss (Eixo Y secundário - Direita)
    # Utilizamos twinx() para partilhar o mesmo eixo X (épocas) mas com uma escala Y diferente
    ax2 = ax.twinx()
    ax2.plot(df['epoca'], df['treino_loss'], label='Treino Loss', color='red', linestyle='-')
    ax2.plot(df['epoca'], df['val_loss'], label='Validação Loss', color='red', linestyle='--')
    ax2.set_ylabel('Loss', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # Formatação do gráfico
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Época', fontsize=12)

    # Juntar as legendas de ambos os eixos (Accuracy e Loss)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='center right')

    # Adicionar uma grelha subtil para facilitar a leitura
    ax.grid(True, alpha=0.3)

# Ajustar o layout para que os gráficos não se sobreponham
plt.tight_layout()

# Guardar o gráfico como imagem
plt.savefig("facenet_comparacao_treino_recortados.png", dpi=300, bbox_inches='tight')

# Mostrar o gráfico no ecrã
plt.show()