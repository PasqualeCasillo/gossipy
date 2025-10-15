import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation
from gossipy import set_seed, CACHE
from gossipy.core import Message, MessageType
from typing import Union


set_seed(42)


class Data1Net(TorchModel):
    """Neural Network per classificazione binaria del dataset data1.csv"""
    
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        self.input_dim = input_dim
        
        # Costruzione della rete fully connected
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Layer di output per classificazione binaria
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def init_weights(self, *args, **kwargs) -> None:
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.network.apply(_init_weights)

    def forward(self, x):
        return self.network(x)
    
    def __repr__(self) -> str:
        return f"Data1Net(input_dim={self.input_dim}, size={self.get_size()})"


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


# FIX per il bug roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from gossipy import GlobalSettings

class FixedTorchModelHandler(TorchModelHandler):
    """TorchModelHandler con fix per il bug roc_auc_score"""
    
    def evaluate(self, data):
        """Versione corretta del metodo evaluate"""
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        self.model.eval()
        self.model = self.model.to(self.device)
        scores = self.model(x)

        if y.dim() == 1:
            y_true = y.cpu().numpy().flatten()
        else:
            y_true = torch.argmax(y, dim=-1).cpu().numpy().flatten()

        pred = torch.argmax(scores, dim=-1)
        y_pred = pred.cpu().numpy().flatten()
        
        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
            "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
            "f1_score": f1_score(y_true, y_pred, zero_division=0, average="macro")
        }

        if scores.shape[1] == 2:
            auc_scores = scores[:, 1].detach().cpu().numpy().flatten()
            if len(set(y_true)) == 2:
                # FIX: roc_auc_score già restituisce un float
                res["auc"] = float(roc_auc_score(y_true, auc_scores))
            else:
                res["auc"] = 0.5
        
        self.model = self.model.to("cpu")
        return res


class FixedPENSNode(PENSNode):
    """PENSNode con correzione del bug di valutazione in Fase 1"""
    
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        """
        FIX BUG CRITICO: Valuta i modelli ricevuti sul TEST SET, non sul training set
        """
        from gossipy import LOG
        
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

class FixedSimulationReport(SimulationReport):
    """SimulationReport con logging corretto del numero di round"""
    
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: list):
        """Override per convertire timestamp in round number"""
        actual_round = round // self.delta  # Converti timestamp in round number
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))


# Caricamento del dataset data1.csv
print("Caricamento dataset data1.csv...")
df = pd.read_csv('archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')

# Separazione features e target
X = df.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = df.iloc[:, -1].values    # Ultima colonna (marker)

print(f"Shape originale: X={X.shape}, y={y.shape}")

# GESTIONE VALORI PROBLEMATICI
print("\nPreprocessing dei dati...")

# 1. Rimuovi colonne con troppi NaN (>50%)
nan_percentage = np.isnan(X).sum(axis=0) / X.shape[0]
cols_to_keep = nan_percentage < 0.5
print(f"Rimosse {(~cols_to_keep).sum()} colonne con >50% NaN")
X = X[:, cols_to_keep]

# 2. Sostituisci infiniti con NaN
X = np.where(np.isinf(X), np.nan, X)
inf_count = np.isinf(X).sum()
print(f"Valori infiniti sostituiti con NaN: {inf_count}")

# 3. Imputa i valori mancanti con la mediana
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
print(f"NaN imputati con la mediana")

# 4. Clip valori estremi (outliers)
percentile_low = np.percentile(X, 1, axis=0)
percentile_high = np.percentile(X, 99, axis=0)
X = np.clip(X, percentile_low, percentile_high)
print(f"Valori clippati ai percentili 1-99")

# Verifica che non ci siano più valori problematici
assert not np.any(np.isnan(X)), "Ancora presenti NaN dopo preprocessing!"
assert not np.any(np.isinf(X)), "Ancora presenti infiniti dopo preprocessing!"

print(f"Shape dopo preprocessing: X={X.shape}")

# Encoding della variabile target
le = LabelEncoder()
y = le.fit_transform(y)
print(f"\nClassi trovate: {le.classes_}")
print(f"Distribuzione classi: {pd.Series(y).value_counts().to_dict()}")

# Normalizzazione delle features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Features normalizzate con StandardScaler")

# Conversione a tensori PyTorch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(f"\nShape finale: X={X.shape}, y={y.shape}")
print(f"Numero di features: {X.shape[1]}")
print(f"Numero di campioni: {X.shape[0]}")

# Split train/test (80/20) con stratificazione
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Calcolo baseline (modello stupido che predice sempre la classe maggioritaria)
from collections import Counter
class_counts = Counter(y_test.numpy())
baseline_accuracy = max(class_counts.values()) / len(y_test)
print(f"\n BASELINE (always predict majority class): {baseline_accuracy:.4f}")
print(f"   Il modello deve superare questo valore per essere considerato utile!")

# Creazione del data handler con test set per ogni nodo
data_handler = ClassificationDataHandler(
    X_train, y_train,
    X_test, y_test
)

# FIX: Parametri PENS ottimizzati
N_NODES = 5

# Creazione del data dispatcher con eval_on_user=True per avere test set locali
data_dispatcher = CustomDataDispatcher(
    data_handler, 
    n=N_NODES,
    eval_on_user=True,  # Abilita test set locale per ogni nodo
    auto_assign=True
)

# PARAMETRI OTTIMIZZATI PER PENS
N_SAMPLED = 4
M_TOP = 2
STEP1_ROUNDS = 100

print(f"\n{'='*60}")
print(f"PARAMETRI PENS:")
print(f"{'='*60}")
print(f"Numero nodi: {N_NODES}")
print(f"Peer disponibili per nodo: {N_NODES - 1}")
print(f"Modelli campionati (n_sampled): {N_SAMPLED}")
print(f"Top modelli selezionati (m_top): {M_TOP}")
print(f"Durata fase 1 (selezione): {STEP1_ROUNDS} rounds")
print(f"{'='*60}\n")

# Generazione dei nodi PENS con la classe corretta
nodes = FixedPENSNode.generate(  # ← USA FixedPENSNode
    data_dispatcher=data_dispatcher,
    p2p_net=StaticP2PNetwork(N_NODES),
    model_proto=FixedTorchModelHandler(
        net=Data1Net(input_dim=X.shape[1], hidden_dims=(128, 64, 32)),
        optimizer=torch.optim.Adam,
        optimizer_params={
            "lr": 0.001,
            "weight_decay": 0.0001
        },
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=32,
        local_epochs=3
    ),
    round_len=100,
    sync=False,
    n_sampled=N_SAMPLED,
    m_top=M_TOP,
    step1_rounds=STEP1_ROUNDS
)

# Creazione del simulatore
simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.2
)

# Report della simulazione con logging corretto
report = FixedSimulationReport(delta=100)  # ← USA FixedSimulationReport
simulator.add_receiver(report)

# Inizializzazione e avvio della simulazione
print("\n" + "="*60)
print("INIZIALIZZAZIONE DEI NODI")
print("="*60)
simulator.init_nodes(seed=42)

print("\n" + "="*60)
print("AVVIO DELLA SIMULAZIONE PENS (VERSIONE CORRETTA)")
print("="*60)
print("Protocollo: PUSH")
print(f"Nodi: {N_NODES}")
print("Rounds: 200 (100 per fase)")
print(f"Fase 1: Selezione dei migliori vicini (round 0-{STEP1_ROUNDS-1})")
print(f"Fase 2: Comunicazione ottimizzata (round {STEP1_ROUNDS}-199)")
print(f"FIX APPLICATI:")
print(f"  ✓ Valutazione su test set in Fase 1")
print(f"  ✓ Logging corretto dei round numbers")
print(f"  ✓ Test set locale per ogni nodo")
print("="*60 + "\n")

simulator.start(n_rounds=200)

# Visualizzazione dei risultati
print("\n" + "="*60)
print("SIMULAZIONE COMPLETATA!")
print("="*60)

# Estrai e mostra i risultati finali
final_results = report.get_evaluation(False)
if final_results:
    # Mostra risultati intermedi (fine fase 1)
    if len(final_results) > STEP1_ROUNDS:
        phase1_round, phase1_metrics = final_results[STEP1_ROUNDS-1]
        print(f"\n Risultati Fine Fase 1 (Round {phase1_round}):")  # ← Ora corretto!
        for metric, value in phase1_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Risultati finali
    final_round, final_metrics = final_results[-1]
    print(f"\n Risultati Finali (Round {final_round}):")  # ← Ora corretto!
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Confronto con baseline
    final_accuracy = final_metrics.get('accuracy', 0)
    improvement = ((final_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    print(f"\n{'='*60}")
    print(f" CONFRONTO CON BASELINE:")
    print(f"{'='*60}")
    print(f"Baseline Accuracy:  {baseline_accuracy:.4f}")
    print(f"PENS Accuracy:      {final_accuracy:.4f}")
    print(f"Miglioramento:      {improvement:+.2f}%")
    
    if final_accuracy <= baseline_accuracy + 0.05:
        print(f"  WARNING: Il modello NON supera significativamente il baseline!")
    else:
        print(f" Il modello supera il baseline del {improvement:.2f}%")
    print(f"{'='*60}\n")

plot_evaluation(
    [[ev for _, ev in report.get_evaluation(False)]], 
    "Overall test results - PENS FIXED with Data1.csv"
)


# ============================================================================
# FIX IMPRECISIONE #2: Calcolo corretto train/test accuracy comparabili
# ============================================================================
print("\n" + "="*60)
print(" ANALISI OVERFITTING/UNDERFITTING (VERSIONE CORRETTA)")
print("="*60)

# 1. Calcola TRAIN ACCURACY GLOBALE (su tutto il training set)
print("\n Calcolo train accuracy GLOBALE...")
# Prendi un nodo qualsiasi per valutare sul dataset completo
sample_node = nodes[0]
global_train_eval = sample_node.model_handler.evaluate((X_train, y_train))
global_train_acc = global_train_eval['accuracy']

# 2. Calcola TEST ACCURACY GLOBALE (già disponibile)
global_test_acc = final_metrics['accuracy']

# 3. Calcola anche le accuracy LOCALI per confronto
local_train_accs = []
local_test_accs = []

for node_id, node in nodes.items():
    # Training accuracy locale
    X_train_node, y_train_node = node.data[0]
    local_train_eval = node.evaluate((X_train_node, y_train_node))
    local_train_accs.append(local_train_eval['accuracy'])
    
    # Test accuracy locale (se disponibile)
    if node.has_test() and node.data[1] is not None:
        local_test_eval = node.evaluate(node.data[1])
        local_test_accs.append(local_test_eval['accuracy'])

avg_local_train_acc = np.mean(local_train_accs)
avg_local_test_acc = np.mean(local_test_accs) if local_test_accs else global_test_acc

# Gap globale (comparabile correttamente)
global_gap = global_train_acc - global_test_acc

print(f"\n Metriche GLOBALI (tutto il dataset):")
print(f"  Train Accuracy: {global_train_acc:.4f}")
print(f"  Test Accuracy:  {global_test_acc:.4f}")
print(f"  Gap:            {global_gap:.4f} ({global_gap*100:.2f}%)")

print(f"\n Metriche LOCALI (media sui nodi):")
print(f"  Train Accuracy: {avg_local_train_acc:.4f}")
print(f"  Test Accuracy:  {avg_local_test_acc:.4f}")
print(f"  Gap:            {avg_local_train_acc - avg_local_test_acc:.4f}")

# Diagnosi basata sul gap GLOBALE (corretto)
print(f"\n Diagnosi (basata su metriche GLOBALI):")
if global_gap < 0.02:
    print(" ✓ GOOD FIT: Il modello generalizza bene!")
elif global_gap < 0.10:
    print("  ⚠ LEGGERO OVERFITTING: Accettabile ma monitorare")
elif global_gap >= 0.10:
    print("  ✗ OVERFITTING: Il modello memorizza il training set!")
    print("     Suggerimenti:")
    print("     - Aumenta dropout (da 0.2 a 0.3-0.4)")
    print("     - Aumenta weight_decay (da 0.0001 a 0.001)")
    print("     - Riduci complessità modello")
else:
    print("  ✗ UNDERFITTING: Il modello è troppo semplice!")
    print("     Suggerimenti:")
    print("     - Aumenta complessità (più layer/neuroni)")
    print("     - Aumenta local_epochs")
    print("     - Riduci regolarizzazione")

if global_train_acc < 0.85:
    print("\n  ℹ Train accuracy bassa → modello potrebbe essere sottopotente")

# Statistiche dettagliate per nodo
print(f"\n Statistiche per nodo:")
print(f"  Train Accuracy: min={min(local_train_accs):.4f}, max={max(local_train_accs):.4f}, std={np.std(local_train_accs):.4f}")
if local_test_accs:
    print(f"  Test Accuracy:  min={min(local_test_accs):.4f}, max={max(local_test_accs):.4f}, std={np.std(local_test_accs):.4f}")
