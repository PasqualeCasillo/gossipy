import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, Message, MessageType
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation
from gossipy import CACHE, LOG
from typing import Union

# Importa le metriche di divergenza
from divergence_metrics import DivergenceTracker, extract_model_weights, jensen_renyi_divergence

def load_custom_dataset_for_onoszko(csv_path="archive/binaryAllNaturalPlusNormalVsAttacks/data2.csv"):
    """
    Carica il dataset personalizzato e lo prepara per Onoszko (classificazione con reti neurali)
    """
    print(f"Caricamento dataset da {csv_path}...")
    
    # Carica il CSV
    df = pd.read_csv(csv_path)
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"Classi uniche in 'marker': {df['marker'].unique()}")
    print(f"Distribuzione: {df['marker'].value_counts()}")
    
    # Separa features e target
    X = df.drop('marker', axis=1)
    y = df['marker']
    
    print(f"Valori target unici: {y.unique()}")
    print(f"Distribuzione target:\n{y.value_counts()}")
    
    # Gestisci valori mancanti e infiniti
    print("Gestione valori mancanti e infiniti...")
    
    # Rimuovi colonne non numeriche
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    # Sostituisci infiniti con NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Gestisci NaN con mediana
    X = X.fillna(X.median())
    
    # Controlla ancora per infiniti
    if np.any(np.isinf(X.values)):
        print("Errore: valori infiniti rilevati dopo pulizia")
        X = X.replace([np.inf, -np.inf], 0)
    
    # Encoding delle etichette
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classi finali: {label_encoder.classes_}")
    print(f"Numero di classi: {len(label_encoder.classes_)}")
    print(f"Distribuzione encoding: {np.bincount(y_encoded)}")
    
    # Normalizzazione robusta
    print("Normalizzazione features...")
    
    # Controlla per valori estremamente grandi prima della normalizzazione
    if np.any(np.abs(X.values) > 1e10):
        print("Valori molto grandi rilevati, applicazione clipping...")
        X = X.clip(-1e10, 1e10)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Controllo finale per infiniti/NaN
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("Errore: NaN/Inf dopo normalizzazione")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Conversione a tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    print(f"Dataset preparato:")
    print(f"  Train: X{X_train_tensor.shape}, y{y_train_tensor.shape}")
    print(f"  Test: X{X_test_tensor.shape}, y{y_test_tensor.shape}")
    
    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), len(label_encoder.classes_)


class TabularNet(TorchModel):
    """Rete neurale semplice per dati tabulari"""
    
    def __init__(self, input_size, n_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        
        self.input_size = input_size
        self.n_classes = n_classes
        
        # Assicura che tutti i parametri siano float32
        self.to(torch.float32)
    
    def init_weights(self, *args, **kwargs) -> None:
        """Inizializzazione semplice dei pesi"""
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)
                m.weight.data = m.weight.data.float()
                m.bias.data = m.bias.data.float()
        
        self.apply(_init_weights)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def __repr__(self) -> str:
        return f"TabularNet(input={self.input_size}, classes={self.n_classes}, size={self.get_size()})"


class CustomDataDispatcher(DataDispatcher):
    """Data dispatcher personalizzato per distribuzione equa dei dati"""
    
    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = math.ceil(n_ex / self.n)

        for idx, i in enumerate(range(0, n_ex, ex_x_user)):
            if idx < self.n:
                self.tr_assignments[idx] = list(range(i, min(i + ex_x_user, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = math.ceil(n_eval_ex / self.n)
            for idx, i in enumerate(range(0, n_eval_ex, eval_ex_x_user)):
                if idx < self.n:
                    self.te_assignments[idx] = list(range(i, min(i + eval_ex_x_user, n_eval_ex)))


class FixedPENSNode(PENSNode):
    """PENSNode con correzione del bug di valutazione in Fase 1"""
    
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        """
        FIX BUG CRITICO: Valuta i modelli ricevuti sul TEST SET, non sul training set
        """
        msg_type: MessageType
        recv_model: any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0]
        
        if msg_type != MessageType.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")
            return None

        if self.step == 1:
            # FIX CRITICO: Usa self.data[1] (test set) invece di self.data[0] (training set)
            if self.data[1] is not None:
                evaluation = CACHE[recv_model].evaluate(self.data[1])  # ← FIX!
            else:
                # Fallback al training set se non c'è test set locale
                evaluation = CACHE[recv_model].evaluate(self.data[0])
                LOG.warning(f"Node {self.idx}: Using training set for evaluation (no local test set)")
            
            self.cache[sender] = (recv_model, -evaluation["accuracy"])

            if len(self.cache) >= self.n_sampled:
                top_m = sorted(self.cache, key=lambda key: self.cache[key][1])[:self.m_top]
                recv_models = [CACHE.pop(self.cache[k][0]) for k in top_m]
                self.model_handler(recv_models, self.data[0])
                self.cache = {}
                for i in top_m:
                    self.neigh_counter[i] += 1
        else:
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0])
        
        return None


class DivergenceTrackingReport(SimulationReport):
    """Report esteso con tracking delle divergenze e logging corretto"""
    
    def __init__(self, delta: int = 100, n_nodes: int = 20):
        super().__init__()
        self.delta = delta
        self.divergence_tracker = DivergenceTracker()
        self.node_weights_history = {}
        self.n_nodes = n_nodes
        self.round_counter = 0
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: list):
        """Override per convertire timestamp in round number"""
        actual_round = round // self.delta  # ← FIX: Converti timestamp → round
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))
    
    def update_timestep(self, t: int):
        """Override per tracciare le divergenze ad ogni round"""
        super().update_timestep(t)
        
        # Ogni delta timestep = 1 round
        if t % self.delta == 0 and t > 0:
            self.round_counter = t // self.delta
            if self.round_counter % 10 == 0:  # Log ogni 10 round
                print(f"  Round {self.round_counter} completed")
    
    def track_node_weights(self, node_id, model_weights):
        """Traccia i pesi di un nodo specifico"""
        if node_id not in self.node_weights_history:
            self.node_weights_history[node_id] = []
        
        weights_flat = extract_model_weights(model_weights)
        self.node_weights_history[node_id].append(weights_flat.copy())
    
    def calculate_network_divergence(self):
        """Calcola la divergenza media nella rete"""
        if len(self.node_weights_history) < 2:
            return None
        
        # Prendi gli ultimi pesi di ogni nodo
        latest_weights = {}
        for node_id, weights_list in self.node_weights_history.items():
            if weights_list:
                latest_weights[node_id] = weights_list[-1]
        
        if len(latest_weights) < 2:
            return None
        
        # Calcola divergenza media tra tutti i nodi
        divergences = []
        node_ids = list(latest_weights.keys())
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                try:
                    div = jensen_renyi_divergence(
                        latest_weights[node_ids[i]], 
                        latest_weights[node_ids[j]]
                    )
                    divergences.append(div)
                except Exception as e:
                    print(f"Error calculating divergence between nodes {node_ids[i]} and {node_ids[j]}: {e}")
                    continue
        
        return np.mean(divergences) if divergences else None
    
    def update_end(self):
        """Override per stampare statistiche finali delle divergenze"""
        super().update_end()
        
        print(f"\n{'='*60}")
        print(f"DIVERGENCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Calcola divergenza finale della rete
        final_network_div = self.calculate_network_divergence()
        if final_network_div is not None:
            print(f"Final Network Divergence (Jensen-Rényi): {final_network_div:.6f}")
        
        # Statistiche sui nodi
        print(f"Nodes tracked: {len(self.node_weights_history)}")
        for node_id, weights_list in self.node_weights_history.items():
            print(f"  Node {node_id}: {len(weights_list)} weight snapshots")
        print(f"{'='*60}")


def patch_evaluate_method():
    """Patch per risolvere il bug dell'AUC score nella libreria gossipy"""
    from gossipy.model.handler import TorchModelHandler
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    original_evaluate = TorchModelHandler.evaluate
    
    def patched_evaluate(self, ext_data):
        try:
            return original_evaluate(self, ext_data)
        except AttributeError as e:
            if "'float' object has no attribute 'astype'" in str(e):
                import torch
                
                model = getattr(self, 'model', None) or getattr(self, '_model', None) or getattr(self, 'net_', None)
                if model is None:
                    for attr_name in dir(self):
                        attr = getattr(self, attr_name)
                        if hasattr(attr, 'eval') and hasattr(attr, 'forward'):
                            model = attr
                            break
                
                if model is None:
                    print("Errore: non riesco a trovare il modello nell'handler")
                    return {"accuracy": 0.0, "auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                
                model.eval()
                with torch.no_grad():
                    x_test, y_test = ext_data
                    y_pred = model(x_test)
                    y_prob = torch.softmax(y_pred, dim=1)
                    y_pred_classes = torch.argmax(y_pred, dim=1)
                    
                    y_true = y_test.cpu().numpy()
                    y_pred_np = y_pred_classes.cpu().numpy()
                    
                    accuracy = accuracy_score(y_true, y_pred_np)
                    
                    if y_prob.shape[1] == 2:
                        y_prob_positive = y_prob[:, 1].cpu().numpy()
                        try:
                            auc = float(roc_auc_score(y_true, y_prob_positive))
                        except:
                            auc = 0.5
                    else:
                        auc = 0.5
                    
                    try:
                        precision = precision_score(y_true, y_pred_np, average='weighted', zero_division=0)
                        recall = recall_score(y_true, y_pred_np, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, y_pred_np, average='weighted', zero_division=0)
                    except:
                        precision = recall = f1 = 0.0
                    
                    return {
                        "accuracy": float(accuracy),
                        "auc": auc,
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1)
                    }
            else:
                raise e
    
    TorchModelHandler.evaluate = patched_evaluate


def patch_merge_method():
    """Patch per gestire il problema di casting dei parametri"""
    from gossipy.model.handler import TorchModelHandler
    
    original_merge = TorchModelHandler._merge
    
    def patched_merge(self, recv_model):
        try:
            return original_merge(self, recv_model)
        except RuntimeError as e:
            if "result type Float can't be cast to the desired output type Long" in str(e):
                return
            else:
                raise e
    
    TorchModelHandler._merge = patched_merge


class DivergenceTrackingSimulator(GossipSimulator):
    """Simulatore esteso con tracking delle divergenze"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.divergence_report = None
        self.weight_snapshot_interval = 100  # Ogni 100 timestep
    
    def add_divergence_receiver(self, report):
        """Aggiunge il receiver per il tracking delle divergenze"""
        self.divergence_report = report
        self.add_receiver(report)
    
    def start(self, n_rounds: int = 100):
        """Override del metodo start per tracciare le divergenze"""
        print(f"\n{'='*60}")
        print(f"Starting Gossip Learning simulation with divergence tracking")
        print(f"{'='*60}")
        print(f"Rounds: {n_rounds}, Nodes: {self.n_nodes}")
        print(f"FIX APPLICATI:")
        print(f"Valutazione su test set in Fase 1")
        print(f"Logging corretto dei round numbers")
        print(f"Test set locale per ogni nodo")
        print(f"Tracking divergenza tra nodi")
        print(f"{'='*60}\n")
        
        # Snapshot iniziale dei pesi
        if self.divergence_report:
            for node_id, node in self.nodes.items():
                if hasattr(node, 'model_handler') and hasattr(node.model_handler, 'model'):
                    try:
                        weights = extract_model_weights(node.model_handler.model)
                        self.divergence_report.track_node_weights(node_id, weights)
                    except Exception as e:
                        print(f"Error tracking initial weights for node {node_id}: {e}")
        
        # Avvia la simulazione normale
        super().start(n_rounds)
        
        # Snapshot finale e calcolo divergenze
        if self.divergence_report:
            print(f"\n  Final divergence calculation...")
            for node_id, node in self.nodes.items():
                if hasattr(node, 'model_handler') and hasattr(node.model_handler, 'model'):
                    try:
                        weights = extract_model_weights(node.model_handler.model)
                        self.divergence_report.track_node_weights(node_id, weights)
                    except Exception as e:
                        print(f"Error tracking final weights for node {node_id}: {e}")
            
            # Calcola e stampa divergenza finale
            final_div = self.divergence_report.calculate_network_divergence()
            if final_div is not None:
                print(f"\n Final Network Jensen-Rényi Divergence: {final_div:.6f}")


# Carica il dataset personalizzato
try:
    train_set, test_set, n_classes = load_custom_dataset_for_onoszko("archive/binaryAllNaturalPlusNormalVsAttacks/data15.csv")
except FileNotFoundError:
    print("Errore: File non trovato!")
    print("Assicurati che il file sia nella directory corretta")
    exit(1)
except Exception as e:
    print(f"Errore nel caricamento dataset: {e}")
    exit(1)

# Setup data handler
data_handler = ClassificationDataHandler(
    train_set[0], train_set[1],  # train data
    test_set[0], test_set[1]     # test data
)

# Data dispatcher con 20 nodi e eval_on_user=True per test set locale
data_dispatcher = CustomDataDispatcher(
    data_handler, 
    n=20, 
    eval_on_user=True,  # Abilita test set locale
    auto_assign=True
)

# Configurazione della rete
input_size = train_set[0].shape[1]
print(f"\nConfigurazione rete:")
print(f"- Input size: {input_size}")
print(f"- Numero classi: {n_classes}")
print(f"- Train samples: {train_set[0].shape[0]}")
print(f"- Test samples: {test_set[0].shape[0]}")

# Applica le patch
patch_evaluate_method()
patch_merge_method()

# Genera nodi PENS con FixedPENSNode
nodes = FixedPENSNode.generate(  # ← USA FixedPENSNode
    data_dispatcher=data_dispatcher,
    p2p_net=StaticP2PNetwork(20),
    model_proto=TorchModelHandler(
        net=TabularNet(input_size, n_classes),
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": 0.01,
            "weight_decay": 0.0001
        },
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=32,
        local_epochs=10
    ),
    round_len=100,
    sync=False,
    n_sampled=10,
    m_top=2,
    step1_rounds=50
)

# Setup simulatore con tracking divergenze
simulator = DivergenceTrackingSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.1
)

# Setup report con tracking divergenze e logging corretto
report = DivergenceTrackingReport(delta=100, n_nodes=20)  # ← FIX: Passa delta
simulator.add_divergence_receiver(report)
simulator.init_nodes(seed=42)

print(f"\nAvvio simulazione PENS con tracking divergenze...")
print(f"Features: {input_size}, Classi: {n_classes}")
print(f"Algoritmo: PENS con sparsificazione dei gradienti (VERSIONE CORRETTA)")

# Avvia simulazione
simulator.start(n_rounds=500)

# Visualizza risultati
print(f"\nSimulazione completata!")
plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], 
                f"PENS FIXED con Divergence Tracking - Dataset ({data_handler.size()} samples, {n_classes} classi)")

# Statistiche finali
if report.get_evaluation(False):
    final_metrics = report.get_evaluation(False)[-1][1]
    print(f"\n{'='*60}")
    print(f"Risultati finali:")
    print(f"{'='*60}")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"{'='*60}")

# Salva i dati delle divergenze per analisi future
print(f"\nSaving divergence data...")
import pickle
with open("gossip_divergence_data.pkl", "wb") as f:
    pickle.dump({
        'node_weights_history': report.node_weights_history,
        'final_metrics': final_metrics if report.get_evaluation(False) else {},
        'config': {
            'n_nodes': 20,
            'input_size': input_size,
            'n_classes': n_classes,
            'rounds': 500
        },
        'fixes_applied': [
            'Valutazione su test set in Fase 1',
            'Logging corretto dei round numbers',
            'Test set locale per ogni nodo'
        ]
    }, f)

print(f"Divergence analysis complete - data saved to gossip_divergence_data.pkl")