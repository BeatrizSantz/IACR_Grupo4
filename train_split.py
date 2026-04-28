import splitfolders


pasta_origem = "data/lfw_filtered"
pasta_destino = "data/dataset_5"

# Divide o dataset: 80% para treino e 20% para validação
splitfolders.ratio(pasta_origem, 
                   output=pasta_destino, 
                   seed=42, 
                   ratio=(0.8, 0.2))

print("Dataset dividido com sucesso!")