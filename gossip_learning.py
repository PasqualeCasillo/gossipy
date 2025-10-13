import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class GossipNode:
    """Singolo nodo nel sistema di gossip learning"""
    
    def __init__(self, node_id: int, features: int, learning_rate: float = 0.01, 
                 gossip_probability: float = 0.1):
        self.node_id = node_id
        self.learning_rate = learning_rate
        self.gossip_probability = gossip_probability
        
        # Inizializza i pesi del modello (regressione logistica)
        self.weights = np.random.normal(0, 0.1, features + 1)  # +1 per il bias
        self.neighbors: List[int] = []
        self.local_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        # Metriche per il monitoraggio
        self.loss_history = []
        self.accuracy_history = []
        self.gossip_count = 0
        
    def set_local_data(self, X: np.ndarray, y: np.ndarray):
        """Assegna i dati locali al nodo"""
        self.local_data = (X, y)
        
    def add_neighbor(self, neighbor_id: int):
        """Aggiunge un vicino al nodo"""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
            
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Funzione sigmoidale per la regressione logistica"""
        # Clipping per evitare overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice le probabilità"""
        # Aggiungi bias
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        z = X_with_bias @ self.weights
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice le classi"""
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcola la loss di cross-entropy"""
        probas = self.predict_proba(X)
        # Evita log(0) aggiungendo epsilon
        epsilon = 1e-15
        probas = np.clip(probas, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(probas) + (1 - y) * np.log(1 - probas))
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calcola i gradienti per la regressione logistica"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        probas = self.predict_proba(X)
        error = probas - y
        gradients = X_with_bias.T @ error / X.shape[0]
        return gradients
    
    def local_update(self):
        """Aggiorna i pesi usando i dati locali"""
        if self.local_data is None:
            return
            
        X, y = self.local_data
        gradients = self.compute_gradients(X, y)
        self.weights -= self.learning_rate * gradients
        
        # Calcola metriche
        loss = self.compute_loss(X, y)
        accuracy = accuracy_score(y, self.predict(X))
        
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
    
    def gossip_weights(self, neighbor_weights: np.ndarray, aggregation_method: str = 'average'):
        """Aggrega i pesi ricevuti dai vicini"""
        if aggregation_method == 'average':
            # Media tra i propri pesi e quelli del vicino
            self.weights = (self.weights + neighbor_weights) / 2
        elif aggregation_method == 'weighted_average':
            # Media ponderata (può essere estesa con pesi specifici)
            alpha = 0.7  # Peso per i propri parametri
            self.weights = alpha * self.weights + (1 - alpha) * neighbor_weights
        
        self.gossip_count += 1


class GossipLearningNetwork:
    """Rete di nodi per il gossip learning"""
    
    def __init__(self, n_nodes: int, features: int, learning_rate: float = 0.01,
                 gossip_probability: float = 0.1, topology: str = 'ring'):
        self.n_nodes = n_nodes
        self.features = features
        self.topology = topology
        
        # Crea i nodi
        self.nodes = [GossipNode(i, features, learning_rate, gossip_probability) 
                     for i in range(n_nodes)]
        
        # Crea la topologia della rete
        self._create_topology()
        
        # Metriche globali
        self.global_loss_history = []
        self.global_accuracy_history = []
        
    def _create_topology(self):
        """Crea la topologia della rete"""
        if self.topology == 'ring':
            # Topologia ad anello
            for i in range(self.n_nodes):
                self.nodes[i].add_neighbor((i + 1) % self.n_nodes)
                self.nodes[i].add_neighbor((i - 1) % self.n_nodes)
                
        elif self.topology == 'complete':
            # Topologia completa
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        self.nodes[i].add_neighbor(j)
                        
        elif self.topology == 'random':
            # Topologia casuale
            for i in range(self.n_nodes):
                # Ogni nodo ha 3-5 vicini casuali
                n_neighbors = random.randint(3, min(5, self.n_nodes - 1))
                neighbors = random.sample([j for j in range(self.n_nodes) if j != i], 
                                        n_neighbors)
                for neighbor in neighbors:
                    self.nodes[i].add_neighbor(neighbor)
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray, 
                       distribution: str = 'uniform', overlap: float = 0.0):
        """Distribuisce i dati tra i nodi"""
        n_samples = X.shape[0]
        
        if distribution == 'uniform':
            # Distribuzione uniforme
            samples_per_node = n_samples // self.n_nodes
            indices = np.random.permutation(n_samples)
            
            for i in range(self.n_nodes):
                start_idx = i * samples_per_node
                end_idx = (i + 1) * samples_per_node if i < self.n_nodes - 1 else n_samples
                node_indices = indices[start_idx:end_idx]
                
                self.nodes[i].set_local_data(X[node_indices], y[node_indices])
                
        elif distribution == 'non_iid':
            # Distribuzione non-IID (ogni nodo ha più samples di una classe)
            class_0_indices = np.where(y == 0)[0]
            class_1_indices = np.where(y == 1)[0]
            
            # Mescola gli indici
            np.random.shuffle(class_0_indices)
            np.random.shuffle(class_1_indices)
            
            for i in range(self.n_nodes):
                if i < self.n_nodes // 2:
                    # Primi nodi: più samples della classe 0
                    n_class_0 = len(class_0_indices) // (self.n_nodes // 2)
                    n_class_1 = len(class_1_indices) // (self.n_nodes * 2)
                    
                    start_0 = i * n_class_0
                    end_0 = (i + 1) * n_class_0
                    start_1 = i * n_class_1
                    end_1 = (i + 1) * n_class_1
                    
                    node_indices = np.concatenate([
                        class_0_indices[start_0:end_0],
                        class_1_indices[start_1:end_1]
                    ])
                else:
                    # Ultimi nodi: più samples della classe 1
                    node_idx = i - self.n_nodes // 2
                    n_class_0 = len(class_0_indices) // (self.n_nodes * 2)
                    n_class_1 = len(class_1_indices) // (self.n_nodes // 2)
                    
                    start_0 = node_idx * n_class_0
                    end_0 = (node_idx + 1) * n_class_0
                    start_1 = node_idx * n_class_1
                    end_1 = (node_idx + 1) * n_class_1
                    
                    node_indices = np.concatenate([
                        class_0_indices[start_0:end_0],
                        class_1_indices[start_1:end_1]
                    ])
                
                np.random.shuffle(node_indices)
                self.nodes[i].set_local_data(X[node_indices], y[node_indices])
    
    def train_epoch(self, gossip_round: bool = True):
        """Esegue un'epoca di training"""
        # Fase 1: Aggiornamento locale
        for node in self.nodes:
            node.local_update()
        
        # Fase 2: Gossip round
        if gossip_round:
            for node in self.nodes:
                if random.random() < node.gossip_probability and node.neighbors:
                    # Scegli un vicino casuale
                    neighbor_id = random.choice(node.neighbors)
                    neighbor = self.nodes[neighbor_id]
                    
                    # Scambia i pesi
                    node.gossip_weights(neighbor.weights.copy())
    
    def evaluate_global_performance(self, X_test: np.ndarray, y_test: np.ndarray):
        """Valuta le performance globali della rete"""
        # Media dei pesi di tutti i nodi
        avg_weights = np.mean([node.weights for node in self.nodes], axis=0)
        
        # Predizioni usando i pesi mediati
        X_test_with_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        z = X_test_with_bias @ avg_weights
        probas = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        predictions = (probas > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, predictions)
        loss = -np.mean(y_test * np.log(np.clip(probas, 1e-15, 1-1e-15)) + 
                       (1 - y_test) * np.log(np.clip(1 - probas, 1e-15, 1-1e-15)))
        
        self.global_accuracy_history.append(accuracy)
        self.global_loss_history.append(loss)
        
        return accuracy, loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 100, verbose: bool = True):
        """Allena la rete per il numero specificato di epoche"""
        
        # Distribuisci i dati
        self.distribute_data(X_train, y_train, distribution='non_iid')
        
        for epoch in range(epochs):
            self.train_epoch(gossip_round=True)
            
            if epoch % 10 == 0:
                accuracy, loss = self.evaluate_global_performance(X_test, y_test)
                if verbose:
                    print(f"Epoca {epoch}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        
        # Valutazione finale
        final_accuracy, final_loss = self.evaluate_global_performance(X_test, y_test)
        
        return final_accuracy, final_loss
    
    def plot_training_metrics(self):
        """Visualizza le metriche di training"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy globale
        axes[0, 0].plot(self.global_accuracy_history)
        axes[0, 0].set_title('Accuracy Globale')
        axes[0, 0].set_xlabel('Epoca')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Loss globale
        axes[0, 1].plot(self.global_loss_history)
        axes[0, 1].set_title('Loss Globale')
        axes[0, 1].set_xlabel('Epoca')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Accuracy per nodo
        for i, node in enumerate(self.nodes):
            if i < 5:  # Mostra solo i primi 5 nodi
                axes[1, 0].plot(node.accuracy_history, label=f'Nodo {i}')
        axes[1, 0].set_title('Accuracy per Nodo')
        axes[1, 0].set_xlabel('Epoca')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Conteggio gossip
        gossip_counts = [node.gossip_count for node in self.nodes]
        axes[1, 1].bar(range(len(gossip_counts)), gossip_counts)
        axes[1, 1].set_title('Conteggio Gossip per Nodo')
        axes[1, 1].set_xlabel('Nodo ID')
        axes[1, 1].set_ylabel('Numero di Gossip')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carica e preprocessa il dataset"""
    
    # Carica i dati
    df = pd.read_csv(file_path)
    
    # Separa features e target
    X = df.drop('marker', axis=1)
    y = df['marker']
    
    # Codifica il target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Gestisce valori problematici
    print(f"Shape originale: {X.shape}")
    print(f"Valori NaN: {X.isnull().sum().sum()}")
    print(f"Valori infiniti: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Sostituisce NaN con la mediana
    X = X.fillna(X.median())
    
    # Sostituisce valori infiniti
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Rimuove colonne con tutti valori identici (varianza = 0)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    constant_cols = []
    for col in numeric_cols:
        if X[col].std() == 0:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Rimosse {len(constant_cols)} colonne costanti: {constant_cols[:5]}...")
        X = X.drop(columns=constant_cols)
    
    # Clipping per valori estremi
    for col in X.select_dtypes(include=[np.number]).columns:
        q1 = X[col].quantile(0.01)
        q99 = X[col].quantile(0.99)
        X[col] = np.clip(X[col], q1, q99)
    
    print(f"Shape dopo preprocessing: {X.shape}")
    
    # Normalizza le features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Verifica finale
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("⚠️ Ancora valori problematici, applicando fix finale...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    
    print("✅ Preprocessing completato")
    
    return X_scaled, y_encoded


def compare_learning_approaches(X_train, y_train, X_test, y_test, 
                               n_nodes=5, epochs=100):
    """Confronta diverse configurazioni di gossip learning"""
    
    results = {}
    
    # Configurazioni da testare
    configs = [
        {'topology': 'ring', 'gossip_prob': 0.1, 'name': 'Ring Topology'},
        {'topology': 'complete', 'gossip_prob': 0.1, 'name': 'Complete Topology'},
        {'topology': 'random', 'gossip_prob': 0.1, 'name': 'Random Topology'},
        {'topology': 'ring', 'gossip_prob': 0.3, 'name': 'Ring High Gossip'},
    ]
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        network = GossipLearningNetwork(
            n_nodes=n_nodes,
            features=X_train.shape[1],
            learning_rate=0.01,
            gossip_probability=config['gossip_prob'],
            topology=config['topology']
        )
        
        start_time = time.time()
        final_accuracy, final_loss = network.train(
            X_train, y_train, X_test, y_test, 
            epochs=epochs, verbose=False
        )
        training_time = time.time() - start_time
        
        results[config['name']] = {
            'accuracy': final_accuracy,
            'loss': final_loss,
            'time': training_time,
            'network': network
        }
        
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Training Time: {training_time:.2f}s")
    
    return results


# Esempio di utilizzo
if __name__ == "__main__":
    # Carica e preprocessa i dati
    print("Caricamento e preprocessing dei dati...")
    X, y = load_and_preprocess_data('archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dimensioni training set: {X_train.shape}")
    print(f"Dimensioni test set: {X_test.shape}")
    print(f"Distribuzione classi train: {np.bincount(y_train)}")
    print(f"Distribuzione classi test: {np.bincount(y_test)}")
    
    # Crea e allena la rete
    print("\nCreazione della rete di gossip learning...")
    network = GossipLearningNetwork(
        n_nodes=8,
        features=X_train.shape[1],
        learning_rate=0.01,
        gossip_probability=0.2,
        topology='ring'
    )
    
    print("\nInizio training...")
    final_accuracy, final_loss = network.train(
        X_train, y_train, X_test, y_test, 
        epochs=150, verbose=True
    )
    
    print(f"\nRisultati finali:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Loss: {final_loss:.4f}")
    
    # Visualizza i risultati
    network.plot_training_metrics()
    
    # Confronta diverse configurazioni
    print("\nConfronto di diverse configurazioni...")
    comparison_results = compare_learning_approaches(
        X_train, y_train, X_test, y_test, 
        n_nodes=6, epochs=100
    )
    
    # Stampa il confronto
    print("\n--- CONFRONTO RISULTATI ---")
    for name, result in comparison_results.items():
        print(f"{name}: Accuracy={result['accuracy']:.4f}, "
              f"Loss={result['loss']:.4f}, Time={result['time']:.2f}s")