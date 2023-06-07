import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)



    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

def narrations2vec(model, data_list):
    vectors = []
    for i, row in data_list.iterrows():
        narration = row['narration']
        print(narration)
        text = clip.tokenize([narration]).to(device)
        with torch.no_grad():
            vector = model.encode_text(text)
            print(vector.shape)
        vectors.append(vector)
    print('successful encoding, number of vectors: ', len(vectors))
    return np.row_stack(vectors)
if __name__ == '__main__':
    train_list = pd.read_pickle('EPIC_train.pkl')
    test_list = pd.read_pickle('EPIC_val.pkl')

    # train_vector = narrations2vec(model, train_list)
    # np.save('train_vector.npy', train_vector)

    # test_vector = narrations2vec(model, test_list)
    # np.save('test_vector.npy', test_vector)