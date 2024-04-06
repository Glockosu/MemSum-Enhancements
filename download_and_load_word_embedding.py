import numpy as np
import pickle
import torch
import os
import requests
import zipfile
from io import BytesIO

# Define the GloVe URL and the desired file name
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_filename = "glove.6B.zip"

# Ensure the target directory exists and change into it
glove_dir = "model/glove/"
os.makedirs(glove_dir, exist_ok=True)
os.chdir(glove_dir)

# Download GloVe embeddings using requests
print("Downloading GloVe embeddings...")
response = requests.get(glove_url)
zip_content = BytesIO(response.content)

# Extract the GloVe embeddings
print("Extracting GloVe embeddings...")
with zipfile.ZipFile(zip_content) as zip_ref:
    zip_ref.extractall(".")

# Process the embeddings
print("Processing GloVe embeddings...")
embed = []
with open("glove.6B.200d.txt", "r", encoding="utf-8") as f:
    for line in f:
        embed.append(line.strip().split()[1:])
embed = np.asarray(embed).astype(np.float32)
embed = np.concatenate([np.zeros((3, 200)), embed], axis=0).astype(np.float32)

# Save the processed embeddings
with open("unigram_embeddings_200dim.pkl", "wb") as f:
    pickle.dump(embed, f, protocol=pickle.HIGHEST_PROTOCOL)

# Return to the original directory
os.chdir("../../")

# Update and save model checkpoints
print("Updating model checkpoints...")
all_model_files = []
for root, dirs, files in os.walk("./"):
    for fname in files:
        if fname.endswith(".pt"):
            all_model_files.append(os.path.join(root, fname))

for model_name in all_model_files:
    model_ckpt = torch.load(model_name, map_location="cpu")
    model_ckpt["local_sentence_encoder"]["word_embedding"] = torch.from_numpy(embed)
    torch.save(model_ckpt, model_name)

print("All models loaded!")