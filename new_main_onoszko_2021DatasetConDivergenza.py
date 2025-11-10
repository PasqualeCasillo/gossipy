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
import pickle

# Importa le metriche di divergenza
from divergence_metrics import DivergenceTracker, extract_model_weights, jensen_renyi_divergence

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


class DivergenceTrackingReport(SimulationReport):
    """SimulationReport esteso con tracking delle divergenze e logging corretto"""
    
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta
        self.divergence_tracker = DivergenceTracker()
        self.node_weights_history = {}
        self.round_divergences = []  # [(round, avg_divergence)]
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: list):
        """Override per convertire timestamp in round number"""
        actual_round = round // self.delta  # Converti timestamp in round number
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))
    
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
                    print(f"Warning: Error calculating divergence between nodes {node_ids[i]} and {node_ids[j]}: {e}")
                    continue
        
        return np.mean(divergences) if divergences else None
    
    def update_end(self) -> None:
        """Override per stampare statistiche finali delle divergenze"""
        super().update_end()
        
        print(f"\n{'='*60}")
        print(f"DIVERGENCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Calcola divergenza finale della rete
        final_network_div = self.calculate_network_divergence()
        if final_network_div is not None:
            print(f"Final Network Divergence (Jensen-Rényi): {final_network_div:.6f}")
        else:
            print(f"Warning: Could not calculate final network divergence")
        
        # Statistiche sui nodi
        print(f"Nodes tracked: {len(self.node_weights_history)}")
        for node_id, weights_list in self.node_weights_history.items():
            print(f"  Node {node_id}: {len(weights_list)} weight snapshots")
        
        # Mostra evoluzione divergenza se disponibile
        if self.round_divergences:
            print(f"\nDivergence evolution:")
            for round_num, div_value in self.round_divergences[-5:]:  # Ultimi 5 round
                print(f"  Round {round_num}: {div_value:.6f}")
        
        print(f"{'='*60}")


class DivergenceTrackingSimulator(GossipSimulator):
    """Simulatore esteso con tracking delle divergenze"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.divergence_report = None
        self.weight_snapshot_interval = 10  # Snapshots ogni 10 round
    
    def add_divergence_receiver(self, report):
        """Aggiunge il receiver per il tracking delle divergenze"""
        self.divergence_report = report
        self.add_receiver(report)
    
    def start(self, n_rounds: int = 100):
        """Override del metodo start per tracciare le divergenze"""
        print(f"\n{'='*60}")
        print(f"Starting PENS simulation with divergence tracking")
        print(f"{'='*60}")
        print(f"Rounds: {n_rounds}, Nodes: {self.n_nodes}")
        print(f"FIX APPLICATI:")
        print(f"  - Valutazione su test set in Fase 1")
        print(f"  - Logging corretto dei round numbers")
        print(f"  - Test set locale per ogni nodo")
        print(f"  - Tracking divergenza tra nodi")
        print(f"{'='*60}\n")
        
        # Snapshot iniziale dei pesi
        if self.divergence_report:
            print("Taking initial weight snapshots...")
            for node_id, node in self.nodes.items():
                if hasattr(node, 'model_handler') and hasattr(node.model_handler, 'model'):
                    try:
                        weights = extract_model_weights(node.model_handler.model)
                        self.divergence_report.track_node_weights(node_id, weights)
                    except Exception as e:
                        print(f"Warning: Error tracking initial weights for node {node_id}: {e}")
        
        # Variabili per tracking durante la simulazione
        last_snapshot_round = 0
        
        # Hook nel loop principale per snapshot periodici
        from collections import defaultdict
        from rich.progress import track
        from numpy.random import shuffle, random, choice
        
        assert self.initialized, "The simulator is not initialized. Please, call the method 'init_nodes'."
        
        node_ids = np.arange(self.n_nodes)
        pbar = track(range(n_rounds * self.delta), description="Simulating...")
        msg_queues = defaultdict(list)
        rep_queues = defaultdict(list)

        try:
            for t in pbar:
                if t % self.delta == 0: 
                    shuffle(node_ids)
                    current_round = t // self.delta
                    
                    # Snapshot periodico delle divergenze
                    if self.divergence_report and current_round > 0 and current_round % self.weight_snapshot_interval == 0:
                        for node_id, node in self.nodes.items():
                            if hasattr(node, 'model_handler') and hasattr(node.model_handler, 'model'):
                                try:
                                    weights = extract_model_weights(node.model_handler.model)
                                    self.divergence_report.track_node_weights(node_id, weights)
                                except:
                                    pass
                        
                        # Calcola divergenza corrente
                        current_div = self.divergence_report.calculate_network_divergence()
                        if current_div is not None:
                            self.divergence_report.round_divergences.append((current_round, current_div))
                            print(f"  Round {current_round}: Network divergence = {current_div:.6f}")
                    
                for i in node_ids:
                    node = self.nodes[i]
                    if node.timed_out(t):
                        peer = node.get_peer()
                        if peer is None:
                            break
                        msg = node.send(t, peer, self.protocol)
                        self.notify_message(False, msg)
                        if msg:
                            if random() >= self.drop_prob:
                                d = self.delay.get(msg)
                                msg_queues[t + d].append(msg)
                            else:
                                self.notify_message(True)
                
                is_online = random(self.n_nodes) <= self.online_prob
                for msg in msg_queues[t]:
                    if is_online[msg.receiver]:
                        reply = self.nodes[msg.receiver].receive(t, msg)
                        if reply:
                            if random() > self.drop_prob:
                                d = self.delay.get(reply)
                                rep_queues[t + d].append(reply)
                            else:
                                self.notify_message(True)
                    else:
                        self.notify_message(True)
                del msg_queues[t]

                for reply in rep_queues[t]:
                    if is_online[reply.receiver]:
                        self.notify_message(False, reply)
                        self.nodes[reply.receiver].receive(t, reply)
                    else:
                        self.notify_message(True)
                    
                del rep_queues[t]

                if (t+1) % self.delta == 0:
                    if self.sampling_eval > 0:
                        sample = choice(list(self.nodes.keys()),
                                        max(int(self.n_nodes * self.sampling_eval), 1))
                        ev = [self.nodes[i].evaluate() for i in sample if self.nodes[i].has_test()]
                    else:
                        ev = [n.evaluate() for _, n in self.nodes.items() if n.has_test()]
                    if ev:
                        self.notify_evaluation(t, True, ev)
                    
                    if self.data_dispatcher.has_test():
                        if self.sampling_eval > 0:
                            ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                                for i in sample]
                        else:
                            ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                                for _, n in self.nodes.items()]
                        if ev:
                            self.notify_evaluation(t, False, ev)
                self.notify_timestep(t)

        except KeyboardInterrupt:
            from gossipy import LOG
            LOG.warning("Simulation interrupted by user.")
        
        pbar.close()
        
        # Snapshot finale dei pesi
        if self.divergence_report:
            print(f"\n  Taking final weight snapshots...")
            for node_id, node in self.nodes.items():
                if hasattr(node, 'model_handler') and hasattr(node.model_handler, 'model'):
                    try:
                        weights = extract_model_weights(node.model_handler.model)
                        self.divergence_report.track_node_weights(node_id, weights)
                    except Exception as e:
                        print(f"Warning: Error tracking final weights for node {node_id}: {e}")
            
            # Calcola e stampa divergenza finale
            final_div = self.divergence_report.calculate_network_divergence()
            if final_div is not None:
                print(f"\n Final Network Jensen-Rényi Divergence: {final_div:.6f}")
                self.divergence_report.round_divergences.append((n_rounds, final_div))
        
        self.notify_end()
        return


# ============================================================================
# CARICAMENTO E PREPROCESSING DEL DATASET
# ============================================================================

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

# ============================================================================
# CONFIGURAZIONE PENS
# ============================================================================

N_NODES = 10

# Creazione del data dispatcher con eval_on_user=True per avere test set locali
data_dispatcher = CustomDataDispatcher(
    data_handler, 
    n=N_NODES,
    eval_on_user=True,  # Abilita test set locale per ogni nodo
    auto_assign=True
)

# PARAMETRI OTTIMIZZATI PER PENS
N_SAMPLED = 10
M_TOP = 3
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
nodes = FixedPENSNode.generate(
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

# ============================================================================
# SIMULAZIONE CON DIVERGENCE TRACKING
# ============================================================================

# Creazione del simulatore con tracking divergenze
simulator = DivergenceTrackingSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=1
)

# Report della simulazione con tracking divergenze e logging corretto
report = DivergenceTrackingReport(delta=100)
simulator.add_divergence_receiver(report)

# Inizializzazione e avvio della simulazione
print("\n" + "="*60)
print("INIZIALIZZAZIONE DEI NODI")
print("="*60)
simulator.init_nodes(seed=42)

simulator.start(n_rounds=400)

# ============================================================================
# VISUALIZZAZIONE RISULTATI
# ============================================================================

print("\n" + "="*60)
print("SIMULAZIONE COMPLETATA!")
print("="*60)

# Estrai e mostra i risultati finali
final_results = report.get_evaluation(False)
if final_results:
    # Mostra risultati intermedi (fine fase 1)
    if len(final_results) > STEP1_ROUNDS:
        phase1_round, phase1_metrics = final_results[STEP1_ROUNDS-1]
        print(f"\n Risultati Fine Fase 1 (Round {phase1_round}):")
        for metric, value in phase1_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Risultati finali
    final_round, final_metrics = final_results[-1]
    print(f"\n Risultati Finali (Round {final_round}):")
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
    "Overall test results - PENS with Divergence Tracking (Data1.csv)"
)

# ============================================================================
# ANALISI OVERFITTING/UNDERFITTING
# ============================================================================

print("\n" + "="*60)
print(" ANALISI OVERFITTING/UNDERFITTING")
print("="*60)

# 1. Calcola TRAIN ACCURACY GLOBALE (su tutto il training set)
print("\n Calcolo train accuracy GLOBALE...")
sample_node = nodes[0]
global_train_eval = sample_node.model_handler.evaluate((X_train, y_train))
global_train_acc = global_train_eval['accuracy']

# 2. Calcola TEST ACCURACY GLOBALE
global_test_acc = final_metrics['accuracy']

# 3. Calcola anche le accuracy LOCALI
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

# Gap globale
global_gap = global_train_acc - global_test_acc

print(f"\n Metriche GLOBALI (tutto il dataset):")
print(f"  Train Accuracy: {global_train_acc:.4f}")
print(f"  Test Accuracy:  {global_test_acc:.4f}")
print(f"  Gap:            {global_gap:.4f} ({global_gap*100:.2f}%)")

print(f"\n Metriche LOCALI (media sui nodi):")
print(f"  Train Accuracy: {avg_local_train_acc:.4f}")
print(f"  Test Accuracy:  {avg_local_test_acc:.4f}")
print(f"  Gap:            {avg_local_train_acc - avg_local_test_acc:.4f}")

# Diagnosi
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

# ============================================================================
# SALVATAGGIO DATI DIVERGENZA
# ============================================================================

print(f"\n{'='*60}")
print(f"Saving divergence data...")
print(f"{'='*60}")

# Salva i dati delle divergenze per analisi future
with open("pens_divergence_data_data1.pkl", "wb") as f:
    pickle.dump({
        'node_weights_history': report.node_weights_history,
        'round_divergences': report.round_divergences,
        'final_metrics': final_metrics if final_results else {},
        'config': {
            'n_nodes': N_NODES,
            'input_size': X.shape[1],
            'n_classes': len(le.classes_),
            'rounds': 400,
            'n_sampled': N_SAMPLED,
            'm_top': M_TOP,
            'step1_rounds': STEP1_ROUNDS
        },
        'baseline_accuracy': baseline_accuracy,
        'global_train_acc': global_train_acc,
        'global_test_acc': global_test_acc,
        'fixes_applied': [
            'Valutazione su test set in Fase 1',
            'Logging corretto dei round numbers',
            'Test set locale per ogni nodo',
            'Tracking divergenza Jensen-Rényi'
        ]
    }, f)

print(f"Divergence analysis complete!")
print(f"Data saved to: pens_divergence_data_data1.pkl")
print(f"{'='*60}\n")