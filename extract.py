import os
import shutil

input_dir = "data/lfw_filtered"
output_dir = "data/lfw_filtered10"
min_images =10

# limpar pasta de saída
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

count_people = 0

for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)

    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(person, len(images))  # debug

    if len(images) >= min_images:
        shutil.copytree(person_path, os.path.join(output_dir, person))
        count_people += 1

print(f"Total de pessoas usadas: {count_people}")