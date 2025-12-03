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

# ============================================================
# RETE NEURALE
# ============================================================
class Data1Net(TorchModel):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def init_weights(self, *args, **kwargs) -> None:
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.network.apply(_init_weights)

    def forward(self, x):
        return self.network(x)

# ============================================================
# DATA DISPATCHER PERSONALIZZATO
# ============================================================
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

# ============================================================
# FIX HANDLER E NODE
# ============================================================
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from gossipy import GlobalSettings

class FixedTorchModelHandler(TorchModelHandler):
    def evaluate(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        self.model.eval()
        self.model = self.model.to(self.device)
        scores = self.model(x)
        y_true = y.cpu().numpy().flatten()
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
                res["auc"] = float(roc_auc_score(y_true, auc_scores))
            else:
                res["auc"] = 0.5
        self.model = self.model.to("cpu")
        return res

class FixedPENSNode(PENSNode):
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        from gossipy import LOG
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0]
        if msg_type != MessageType.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")
            return None
        if self.step == 1:
            if self.data[1] is not None:
                evaluation = CACHE[recv_model].evaluate(self.data[1])
            else:
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

# ============================================================
# FIXED SIMULATION REPORT - EVITA DOPPIA STAMPA
# ============================================================
class FixedSimulationReport(SimulationReport):
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta
        self._report_printed = False  # ← FIX: aggiungi flag
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: list):
        actual_round = round // self.delta
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))
    
    # ← FIX: override update_end per evitare doppia stampa
    def update_end(self) -> None:
        if not self._report_printed:
            from gossipy import LOG
            LOG.info("# Sent messages: %d" % self._sent_messages)
            LOG.info("# Failed messages: %d" % self._failed_messages)
            LOG.info("Total size: %d" % self._total_size)
            self._report_printed = True

# ============================================================
# CARICAMENTO DATASET E PREPROCESSING
# ============================================================
print("Caricamento dataset data1.csv...")
df = pd.read_csv('archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(f"Shape originale: X={X.shape}, y={y.shape}")

# Rimozione e imputazione
nan_percentage = np.isnan(X).sum(axis=0) / X.shape[0]
cols_to_keep = nan_percentage < 0.5
X = X[:, cols_to_keep]
X = np.where(np.isinf(X), np.nan, X)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
percentile_low = np.percentile(X, 1, axis=0)
percentile_high = np.percentile(X, 99, axis=0)
X = np.clip(X, percentile_low, percentile_high)
scaler = StandardScaler()
X = scaler.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline
from collections import Counter
baseline = max(Counter(y_test.numpy()).values()) / len(y_test)
print(f"Baseline accuracy: {baseline:.4f}")

# ============================================================
# DATA HANDLER E PARAMETRI
# ============================================================
data_handler = ClassificationDataHandler(X_train, y_train, X_test, y_test)
N_NODES = 10
data_dispatcher = CustomDataDispatcher(data_handler, n=N_NODES, eval_on_user=True, auto_assign=True)
N_SAMPLED, M_TOP, STEP1_ROUNDS = 10, 3, 50

# ============================================================
# CICLO ESPERIMENTI PER DIVERSI drop_prob
# ============================================================
drop_probs = [0.0, 0.1, 0.3, 0.5, 0.7]
results_by_drop = {}

for drop_p in drop_probs:
    print("\n" + "="*80)
    print(f" AVVIO ESPERIMENTO con drop_prob = {drop_p}")
    print("="*80)

    # ← FIX: crea nuovi nodi per ogni esperimento
    nodes = FixedPENSNode.generate(
        data_dispatcher=data_dispatcher,
        p2p_net=StaticP2PNetwork(N_NODES),
        model_proto=FixedTorchModelHandler(
            net=Data1Net(input_dim=X.shape[1], hidden_dims=(128, 64, 32)),
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
            criterion=F.cross_entropy,
            create_model_mode=CreateModelMode.MERGE_UPDATE,
            batch_size=32,
            local_epochs=4
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
        sampling_eval=1,
        drop_prob=drop_p
    )

    # ← FIX: crea UN SOLO report per simulazione
    report = FixedSimulationReport(delta=100)
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=500)

    eval_results = report.get_evaluation(False)
    if eval_results:
        rounds = [r for r, _ in eval_results]
        accuracies = [ev.get("accuracy", 0) for _, ev in eval_results]
        results_by_drop[drop_p] = {"rounds": rounds, "accuracies": accuracies}
    
    # ← FIX: pulisci la cache tra esperimenti
    CACHE.clear()

# ============================================================
# CONFRONTO TRA ESPERIMENTI
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
colors = {0.0: "red", 0.1: "blue", 0.3: "purple", 0.5: "orange", 0.7: "green"}

for drop_p, res in results_by_drop.items():
    plt.plot(res["rounds"], res["accuracies"], label=f"drop_prob={drop_p}", linewidth=2, color=colors[drop_p])

plt.xlabel("Round", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Confronto Accuracy per diversi valori di drop_prob", fontsize=14, fontweight="bold")
plt.axvline(x=STEP1_ROUNDS, color="gray", linestyle=":", label=f"Fine Fase 1 ({STEP1_ROUNDS})")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("accuracy_comparison_drop_prob.png", dpi=300, bbox_inches="tight")
print("\nGrafico confronto accuracy salvato in: accuracy_comparison_drop_prob.png")
plt.show()

print("\n" + "="*60)
print(" RISULTATI FINALI PER OGNI drop_prob")
print("="*60)
for drop_p, res in results_by_drop.items():
    final_acc = res["accuracies"][-1]
    print(f" drop_prob={drop_p}: Accuracy finale = {final_acc:.4f} (Baseline={baseline:.4f})")