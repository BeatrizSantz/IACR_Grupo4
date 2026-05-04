import splitfolders


pasta_origem = "data/lfw_filtered10"
pasta_destino = "data/dataset_10cd"


splitfolders.ratio(pasta_origem, 
                   output=pasta_destino, 
                   seed=42, 
                   ratio=(0.8, 0.2))

print("Dataset dividido com sucesso!")