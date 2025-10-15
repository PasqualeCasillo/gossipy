import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, ConstantDelay
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import AdaLine
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation

__version__ = "0.0.3"
__author__ = "Fixed Pegasos"

def load_custom_dataset(csv_path="archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv"):
    """Carica e preprocessa il dataset"""
    print(f"Caricamento dataset da {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # Separa features e target
    X = df.drop('marker', axis=1)
    y = df['marker']
    
    print(f"Distribuzione classi:")
    print(y.value_counts())
    print(f"Sbilanciamento: {y.value_counts().max() / len(y):.2%}")
    
    # Preprocessing
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    if np.any(np.abs(X.values) > 1e10):
        X = X.clip(-1e10, 1e10)
    
    # Normalizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encoding per Pegasos (-1/+1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_binary = 2 * y_encoded - 1
    
    print(f"Mapping classi: {dict(zip(label_encoder.classes_, [-1, 1]))}")
    
    # Conversione a tensori
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_binary, dtype=torch.float32)
    
    print(f"Dataset preparato: X{X_tensor.shape}, y{y_tensor.shape}\n")
    
    return X_tensor, y_tensor, label_encoder

set_seed(42)

# Carica dataset
X, y, label_encoder = load_custom_dataset("archive/binaryAllNaturalPlusNormalVsAttacks/data7.csv")

# Calcola baseline
n_neg = (y == -1).sum().item()
n_pos = (y == 1).sum().item()
baseline_accuracy = max(n_neg, n_pos) / len(y)

print("="*60)
print(f" BASELINE (predict majority): {baseline_accuracy:.4f}")
print(f"   Pegasos DEVE superare questo valore!")
print("="*60 + "\n")

#  PARAMETRI OTTIMIZZATI PER PEGASOS
N_NODES = 20
LEARNING_RATE = 0.1  #  Aumentato 10x (era 0.01)
N_ROUNDS = 500       #  Più rounds per convergenza (era 200)
LAMBDA_REG = 0.01    #  Regolarizzazione (implicita in Pegasos)

print("="*60)
print("PARAMETRI PEGASOS (OTTIMIZZATI):")
print("="*60)
print(f"Numero nodi:       {N_NODES}")
print(f"Samples per nodo:  ~{len(X) // N_NODES}")
print(f"Learning rate:     {LEARNING_RATE}  (era 0.01)")
print(f"Rounds:            {N_ROUNDS}  (era 200)")
print(f"Protocol:          PUSH")
print("="*60 + "\n")

# Setup
data_handler = ClassificationDataHandler(X, y, test_size=.1)
data_dispatcher = DataDispatcher(
    data_handler, 
    n=N_NODES,
    eval_on_user=False, 
    auto_assign=True
)

topology = StaticP2PNetwork(N_NODES, None)

#  Model handler con learning rate aumentato
model_handler = PegasosHandler(
    net=AdaLine(data_handler.size(1)),
    learning_rate=LEARNING_RATE,  #  0.1 invece di 0.01
    create_model_mode=CreateModelMode.MERGE_UPDATE
)

# Genera nodi
nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=model_handler,
    round_len=100,
    sync=False
)

# Simulatore
simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=ConstantDelay(0),
    online_prob=1.0,
    drop_prob=0.0,
    sampling_eval=.2  #  Aumentato per più feedback
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)

print("="*60)
print("AVVIO SIMULAZIONE PEGASOS")
print("="*60)
print(f"Nodi:     {len(nodes)}")
print(f"Features: {data_handler.size(1)}")
print(f"Rounds:   {N_ROUNDS}")
print("="*60 + "\n")

simulator.start(n_rounds=N_ROUNDS)

# Risultati
print("\n" + "="*60)
print("SIMULAZIONE COMPLETATA!")
print("="*60 + "\n")

if report.get_evaluation(False):
    results = report.get_evaluation(False)
    
    #  Mostra progresso (inizio, metà, fine)
    if len(results) > 100:
        print(" Progresso Learning:")
        milestones = [len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
        for idx in milestones:
            round_num, metrics = results[idx]
            print(f"  Round {round_num//100:3d}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        print()
    
    # Risultati finali
    final_round, final_metrics = results[-1]
    
    print(" Risultati Finali:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Confronto baseline
    final_accuracy = final_metrics.get('accuracy', 0)
    improvement = ((final_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    
    print(f"\n{'='*60}")
    print(" CONFRONTO CON BASELINE:")
    print("="*60)
    print(f"Baseline Accuracy:  {baseline_accuracy:.4f}")
    print(f"Pegasos Accuracy:   {final_accuracy:.4f}")
    print(f"Miglioramento:      {improvement:+.2f}%")
    
    if final_accuracy <= baseline_accuracy + 0.02:
        print("  WARNING: Pegasos NON supera il baseline!")
        print("    Possibili cause:")
        print("    - Learning rate ancora troppo basso")
        print("    - Dataset non linearmente separabile")
        print("    - Serve più rounds per convergenza")
    else:
        print(f" Pegasos supera il baseline del {improvement:.2f}%")
    print("="*60 + "\n")

plot_evaluation(
    [[ev for _, ev in report.get_evaluation(False)]], 
    f"Pegasos - data7.csv ({N_NODES} nodi, LR={LEARNING_RATE})"
)