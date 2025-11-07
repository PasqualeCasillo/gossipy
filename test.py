import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from collections import Counter

from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, Message, MessageType
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation
from gossipy import set_seed, CACHE

set_seed(42)


class Data1Net(TorchModel):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
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


class CustomDataDispatcher(DataDispatcher):
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


# ============================================================================
# PENS CON LOGGING DELLA SELEZIONE
# ============================================================================
class FixedPENSNodeWithLogging(PENSNode):
    """PENS con logging dettagliato della selezione"""
    
    def _select_neighbors(self) -> None:
        """Override per aggiungere logging"""
        self.best_nodes = []
        
        print(f"\n{'='*60}")
        print(f"NODO {self.idx}: SELEZIONE FINALE DEI MIGLIORI VICINI")
        print(f"{'='*60}")
        print(f"Peer disponibili: {list(self.neigh_counter.keys())}")
        print(f"\nStatistiche Fase 1:")
        
        for i, cnt in self.neigh_counter.items():
            threshold = self.selected[i] * (self.m_top / self.n_sampled)
            selected_pct = (cnt / self.selected[i] * 100) if self.selected[i] > 0 else 0
            
            is_best = cnt > threshold
            marker = "✅" if is_best else "❌"
            
            print(f"  Peer {i}: selezionato {cnt}/{self.selected[i]} volte ({selected_pct:.1f}%) | "
                  f"threshold={threshold:.1f} {marker}")
            
            if is_best:
                self.best_nodes.append(i)
        
        print(f"\n🎯 MIGLIORI VICINI SELEZIONATI: {self.best_nodes}")
        print(f"{'='*60}\n")
    
    def receive(self, t: int, msg: Message):
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0]
        
        if msg_type != MessageType.PUSH:
            return None

        if self.step == 1:
            if self.data[1] is not None:
                evaluation = CACHE[recv_model].evaluate(self.data[1])
            else:
                evaluation = CACHE[recv_model].evaluate(self.data[0])
            
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


class FixedReport(SimulationReport):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
    
    def update_evaluation(self, round, on_user, evaluation):
        actual_round = round // self.delta
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))


# CARICAMENTO DATASET (versione compatta)
print("Caricamento dataset data1.csv...")
df = pd.read_csv('archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

nan_percentage = np.isnan(X).sum(axis=0) / X.shape[0]
X = X[:, nan_percentage < 0.5]
X = np.where(np.isinf(X), np.nan, X)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X = np.clip(X, np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0))

le = LabelEncoder()
y = le.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_accuracy = max(Counter(y_test.numpy()).values()) / len(y_test)
print(f"Baseline: {baseline_accuracy:.4f}\n")

# CONFIGURAZIONE
N_NODES = 5
N_SAMPLED = 4
M_TOP = 2
STEP1_ROUNDS = 100

data_handler = ClassificationDataHandler(X_train, y_train, X_test, y_test)
data_dispatcher = CustomDataDispatcher(data_handler, n=N_NODES, eval_on_user=True, auto_assign=True)

# USA LA VERSIONE CON LOGGING
nodes = FixedPENSNodeWithLogging.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=StaticP2PNetwork(N_NODES),
    model_proto=TorchModelHandler(
        net=Data1Net(input_dim=X.shape[1], hidden_dims=(128, 64, 32)),
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
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

simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.2
)

report = FixedReport(delta=100)
simulator.add_receiver(report)

print("AVVIO SIMULAZIONE CON LOGGING SELEZIONE NODI\n")
simulator.init_nodes(seed=42)
simulator.start(n_rounds=200)

# RISULTATI
print("\n" + "="*60)
print("RISULTATI FINALI")
print("="*60)

results = report.get_evaluation(False)
if results:
    phase1_metrics = results[99][1]
    final_metrics = results[-1][1]
    
    print(f"\nFine Fase 1 (Round 99):  accuracy={phase1_metrics['accuracy']:.4f}")
    print(f"Finale (Round 199):      accuracy={final_metrics['accuracy']:.4f}")
    print(f"Miglioramento vs baseline: +{(final_metrics['accuracy']-baseline_accuracy)*100:.2f}%")

plot_evaluation([[ev for _, ev in results]], "PENS with Selection Logging")