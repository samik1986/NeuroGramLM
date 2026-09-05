import os
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import logging

class MultimodalVQ:
    def __init__(self, config):
        self.config = config['vector_quantization']
        self.model_dir = self.config['model_dir']
        self.vocab_sizes = self.config['vocab_sizes']
        self.codebooks = {}
        self.is_trained = False
        self.buffer = {k: [] for k in self.vocab_sizes.keys()}
        self.buffer_size = 2048
        
        # Initialize MiniBatchKMeans for each modality
        for modality, vocab_size in self.vocab_sizes.items():
            self.codebooks[modality] = MiniBatchKMeans(
                n_clusters=vocab_size,
                random_state=42,
                batch_size=1024,
                n_init='auto'
            )
            
    def load(self):
        """Loads trained codebooks if they exist."""
        all_loaded = True
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = self.model_dir if os.path.isabs(self.model_dir) else os.path.join(base_dir, self.model_dir.lstrip('./'))
        for modality in self.vocab_sizes.keys():
            path = os.path.join(model_dir, f"kmeans_{modality}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.codebooks[modality] = pickle.load(f)
            else:
                all_loaded = False
        self.is_trained = all_loaded
        return self.is_trained

    def save(self):
        """Saves trained codebooks."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = self.model_dir if os.path.isabs(self.model_dir) else os.path.join(base_dir, self.model_dir.lstrip('./'))
        os.makedirs(model_dir, exist_ok=True)
        for modality, model in self.codebooks.items():
            path = os.path.join(model_dir, f"kmeans_{modality}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(model, f)

    def fit_partial(self, features_dict):
        """Incrementally trains the codebooks on new features using a buffer."""
        for modality, data in features_dict.items():
            if modality in self.codebooks and data is not None and len(data) > 0:
                # Ensure data is 2D and float32 for MiniBatchKMeans
                X = np.array(data, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                # Filter out NaNs if any
                X = X[~np.isnan(X).any(axis=1)]
                if len(X) > 0:
                    self.buffer[modality].extend(X)
                    
                if len(self.buffer[modality]) >= self.buffer_size:
                    X_batch = np.array(self.buffer[modality])
                    self.codebooks[modality].partial_fit(X_batch)
                    self.buffer[modality] = []
                
    def flush_buffer(self):
        """Processes any remaining samples in the buffer."""
        for modality, data in self.buffer.items():
            # partial_fit requires at least n_clusters samples for the very first initialization
            # but if it has already been initialized, fewer samples are fine.
            # To be safe, we only fit if it meets the cluster size or if it's already initialized.
            if len(data) >= self.vocab_sizes[modality] or (hasattr(self.codebooks[modality], 'cluster_centers_') and len(data) > 0):
                X_batch = np.array(data, dtype=np.float32)
                self.codebooks[modality].partial_fit(X_batch)
            self.buffer[modality] = []

    def quantize(self, token_features):
        """
        Maps continuous token features to discrete IDs.
        Takes a single token's feature dictionary.
        Returns a dictionary of discrete IDs.
        """
        if not self.is_trained:
            return {}
            
        discrete_ids = {}
        for modality, value in token_features.items():
            if modality in self.codebooks and value is not None:
                # Reshape to 2D and ensure float32
                X = np.array(value, dtype=np.float32)
                if X.ndim == 0:
                    X = X.reshape(1, -1)
                elif X.ndim == 1:
                    X = X.reshape(1, -1)
                
                # Check for NaNs
                if np.isnan(X).any():
                    discrete_ids[modality] = -1 # Or some padding ID
                else:
                    cluster_id = self.codebooks[modality].predict(X)[0]
                    discrete_ids[modality] = int(cluster_id)
        return discrete_ids
