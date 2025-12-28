import numpy as np
import os
from sklearn.svm import SVR

class OnlineSVR:
    def __init__(self, data_path="./svr_data.npz"):
        self.data_path = data_path
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        self.svr = SVR(kernel='linear', C=1.0)
        self.svr_data_len = 0
    
    def save_data(self, embeddings: np.ndarray, labels: list, pre_file:str, filename: str):
        assert embeddings.shape[0] == len(labels), f"The number of the embeddings and labels should be same! but get embeddings: {embeddings.shape[0]}, labels: {len(labels)}"
        if os.path.exists(pre_file):
            pre_emb, pre_lab = self.load_data(filename)
            embeddings = np.concatenate([pre_emb, embeddings], axis=0)
            labels = np.concatenate([pre_lab, labels], axis=0)
            print(embeddings.shape)
            self.svr_data_len = embeddings.shape[0]
        np.savez_compressed(filename, embeddings=embeddings, labels=labels)

    def load_data(self, filename: str):
        data = np.load(filename)
        return data['embeddings'], data['labels']
    
    def online_train(self, emb, lab):
        self.save_data(emb, lab, self.data_path, self.data_path)
        emb, lab = self.load_data(self.data_path)
        self.svr = SVR(kernel='linear', C=1.0) # refresh the model
        self.svr.fit(emb, lab)
    
    def predict(self, emb):
        return self.svr.predict(emb)


