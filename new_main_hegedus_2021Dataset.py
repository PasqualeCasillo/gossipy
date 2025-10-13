import pandas as pd
import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from networkx import to_numpy_array
from networkx.generators.random_graphs import random_regular_graph
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gossipy import set_seed
from gossipy.core import UniformDelay, AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode, PartitioningBasedNode, SamplingBasedNode
from gossipy.model.handler import PartitionedTMH, SamplingTMH, TorchModelHandler
from gossipy.model.sampling import TorchModelPartition
from gossipy.model.nn import LogisticRegression
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, SimulationReport, TokenizedGossipSimulator
from gossipy.flow_control import RandomizedTokenAccount
from gossipy.utils import plot_evaluation

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

def load_custom_dataset_for_hegedus(csv_path="archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv"):
    """
    Carica il dataset personalizzato e lo prepara per Hegedus (classificazione multi-classe con CrossEntropyLoss)
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
        # Sostituisci infiniti rimanenti con 0
        X = X.replace([np.inf, -np.inf], 0)
    
    # Per Hegedus, manteniamo la classificazione multi-classe
    # Encoding: converti a indici 0, 1, 2, ... per CrossEntropyLoss
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
    
    # Conversione a tensori PyTorch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)  # long per CrossEntropyLoss
    
    print(f"Dataset preparato: X{X_tensor.shape}, y{y_tensor.shape}")
    
    return X_tensor, y_tensor, len(label_encoder.classes_)

# Imposta seed per riproducibilità
set_seed(98765)

# Carica il dataset personalizzato invece di spambase
try:
    X, y, n_classes = load_custom_dataset_for_hegedus("archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv")
except FileNotFoundError:
    print("Errore: File 'data7.csv' non trovato!")
    print("Assicurati che il file sia nella directory 'archive/binaryAllNaturalPlusNormalVsAttacks/'")
    exit(1)
except Exception as e:
    print(f"Errore nel caricamento dataset: {e}")
    exit(1)

# Setup data handler e dispatcher
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False, auto_assign=True)

# Topologia di rete
topology = StaticP2PNetwork(100, to_numpy_array(random_regular_graph(20, 100, seed=42)))

# Rete neurale adattata al numero di features e classi del dataset
net = LogisticRegression(data_handler.Xtr.shape[1], n_classes)

print(f"\nConfigurazione:")
print(f"- Features: {data_handler.Xtr.shape[1]}")
print(f"- Classi: {n_classes}")
print(f"- Samples training: {data_handler.Xtr.shape[0]}")
print(f"- Samples test: {data_handler.Xte.shape[0]}")
print(f"- Nodi: 100")

# Genera nodi con partizionamento
nodes = PartitioningBasedNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    round_len=100,
    model_proto=PartitionedTMH(
        net=net,
        tm_partition=TorchModelPartition(net, 4),
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": 1,
            "weight_decay": .001
        },
        criterion=CrossEntropyLoss(),
        create_model_mode=CreateModelMode.UPDATE #CreateModelMode.MERGE_UPDATE
    ),
    sync=True
)

# Setup simulatore con token
simulator = TokenizedGossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    token_account=RandomizedTokenAccount(C=20, A=10),
    utility_fun=lambda mh1, mh2, msg: 1, #The utility function is always = 1 (i.e., utility is not used)
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    delay=UniformDelay(0, 10),
    #online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
    #drop_prob=.1, #Simulates the possibility of message dropping
    sampling_eval=.1
)

# Patch per risolvere il bug dell'AUC score
def patch_evaluate_method():
    """Patch per risolvere il bug dell'AUC score nella libreria gossipy"""
    from gossipy.model.handler import PartitionedTMH
    from sklearn.metrics import roc_auc_score
    
    original_evaluate = PartitionedTMH.evaluate
    
    def patched_evaluate(self, ext_data):
        try:
            return original_evaluate(self, ext_data)
        except AttributeError as e:
            if "'float' object has no attribute 'astype'" in str(e):
                # Riproviamo con gestione manuale dell'AUC
                import torch
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                self.model.eval()
                with torch.no_grad():
                    x_test, y_test = ext_data
                    y_pred = self.model(x_test)
                    y_prob = torch.softmax(y_pred, dim=1)
                    y_pred_classes = torch.argmax(y_pred, dim=1)
                    
                    # Converti in numpy per sklearn
                    y_true = y_test.cpu().numpy()
                    y_pred_np = y_pred_classes.cpu().numpy()
                    
                    # Calcola metriche base
                    accuracy = accuracy_score(y_true, y_pred_np)
                    
                    # Per AUC, usa solo la probabilità della classe positiva (per classificazione binaria)
                    if y_prob.shape[1] == 2:
                        y_prob_positive = y_prob[:, 1].cpu().numpy()
                        try:
                            auc = float(roc_auc_score(y_true, y_prob_positive))
                        except:
                            auc = 0.5  # Valore di default se AUC non calcolabile
                    else:
                        auc = 0.5
                    
                    # Calcola altre metriche
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
    
    PartitionedTMH.evaluate = patched_evaluate

# Applica la patch
patch_evaluate_method()

# Setup report e avvio
report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)

print(f"\nAvvio simulazione Hegedus con {len(nodes)} nodi...")
print(f"Features: {data_handler.Xtr.shape[1]}, Samples: {data_handler.size()}")

simulator.start(n_rounds=1000)

# Visualizza risultati
print(f"\nSimulazione completata!")
plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], 
                f"Hegedus - Dataset Personalizzato ({data_handler.size()} samples, {n_classes} classi)")

# Statistiche finali
if report.get_evaluation(False):
    final_metrics = report.get_evaluation(False)[-1][1]
    print(f"\nRisultati finali:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")