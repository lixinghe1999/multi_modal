from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd 

# sentences = ["I'm happy", "I'm full of happiness"]
sentences = ['I open the door', 'I wash my hand']
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
print('miniLM example output')
print(embedding_1.shape, embedding_2.shape)
print(util.pytorch_cos_sim(embedding_1, embedding_2))

def narrations2vec(model, data_list):
    vectors = []
    for i, row in data_list.iterrows():
        narration = row['narration']
        vector = model.encode(narration, convert_to_tensor=False)
        vectors.append(vector)
    print('successful encoding, number of vectors: ', len(vectors))
    return np.row_stack(vectors)
if __name__ == '__main__':
    train_list = pd.read_pickle('EPIC_train.pkl')
    test_list = pd.read_pickle('EPIC_val.pkl')

    train_vector = narrations2vec(model, train_list)
    np.save('train_vector.npy', train_vector)

    test_vector = narrations2vec(model, test_list)
    np.save('test_vector.npy', test_vector)