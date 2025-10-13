import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformDelay
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import AdaLine
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, SimulationReport
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

def load_custom_dataset(csv_path="archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv"):
    """
    Carica il dataset personalizzato e lo prepara per Pegasos (classificazione binaria)
    """
    print(f"Caricamento dataset da {csv_path}...")
    
    # Carica il CSV
    df = pd.read_csv(csv_path)
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    # Aggiungi prima del caricamento:
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
    
    # Per Pegasos, convertiamo a classificazione binaria
    # Se hai più di 2 classi, prendi le 2 più frequenti
    if len(y.unique()) > 2:
        top_2_classes = y.value_counts().head(2).index
        mask = y.isin(top_2_classes)
        X = X[mask]
        y = y[mask]
        print(f"Ridotto a classificazione binaria con classi: {top_2_classes.tolist()}")
    
    # Encoding: converti a -1/+1 per Pegasos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_binary = 2 * y_encoded - 1  # Converte 0,1 in -1,+1
    
    print(f"Classi finali: {label_encoder.classes_}")
    print(f"Distribuzione binaria: -1={np.sum(y_binary == -1)}, +1={np.sum(y_binary == 1)}")
    
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
    y_tensor = torch.tensor(y_binary, dtype=torch.float32)
    
    print(f"Dataset preparato: X{X_tensor.shape}, y{y_tensor.shape}")
    
    return X_tensor, y_tensor

# Imposta seed per riproducibilità
set_seed(42)

# Carica il dataset personalizzato invece di spambase
try:
    X, y = load_custom_dataset("archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv")
except FileNotFoundError:
    print("Errore: File 'data.csv' non trovato!")
    print("Assicurati che il file sia nella stessa directory dello script")
    exit(1)
except Exception as e:
    print(f"Errore nel caricamento dataset: {e}")
    exit(1)

# Resto del codice identico a main_ormandi_2013.py
data_handler = ClassificationDataHandler(X, y, test_size=.1)
data_dispatcher = DataDispatcher(data_handler, eval_on_user=False, auto_assign=True)
topology = StaticP2PNetwork(data_dispatcher.size(), None)
model_handler = PegasosHandler(net=AdaLine(data_handler.size(1)),
                               learning_rate=.01,
                               create_model_mode=CreateModelMode.MERGE_UPDATE)

# Genera nodi
nodes = GossipNode.generate(data_dispatcher=data_dispatcher,
                            p2p_net=topology,
                            model_proto=model_handler,
                            round_len=100,
                            sync=False)

# Setup simulatore
simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=UniformDelay(0,10),
    online_prob=.2, # Simula condizioni reali di rete mobile
    drop_prob=.1,   # Simula perdita di messaggi
    sampling_eval=.1
)

# Setup report e avvio
report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)

print(f"\nAvvio simulazione Pegasos con {len(nodes)} nodi...")
print(f"Features: {data_handler.size(1)}, Samples: {data_handler.size()}")

simulator.start(n_rounds=200)

# Visualizza risultati
print(f"\nSimulazione completata!")
plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], 
                f"Pegasos - Dataset Personalizzato ({data_handler.size()} samples)")

# Statistiche finali
if report.get_evaluation(False):
    final_metrics = report.get_evaluation(False)[-1][1]
    print(f"\nRisultati finali:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")