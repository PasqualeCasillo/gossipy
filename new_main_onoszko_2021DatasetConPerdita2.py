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
from typing import Union, Tuple, Dict, List
import matplotlib.pyplot as plt

from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, Message, MessageType
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy import set_seed, CACHE, LOG, GlobalSettings

set_seed(42)


# ============================================================================
# MODELLO E CLASSI CORRETTE
# ============================================================================

class Data1Net(TorchModel):
    """Neural Network per classificazione binaria del dataset data1.csv"""
    
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


class FixedTorchModelHandler(TorchModelHandler):
    """TorchModelHandler con fix per il bug roc_auc_score"""
    
    def evaluate(self, data):
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
                res["auc"] = float(roc_auc_score(y_true, auc_scores))
            else:
                res["auc"] = 0.5
        
        self.model = self.model.to("cpu")
        return res


class FixedPENSNode(PENSNode):
    """PENSNode con correzione del bug di valutazione in Fase 1"""
    
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: any 
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


class FixedSimulationReport(SimulationReport):
    """SimulationReport con logging corretto del numero di round"""
    
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: list):
        actual_round = round // self.delta
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((actual_round, ev))
        else:
            self._global_evaluations.append((actual_round, ev))


# ============================================================================
# CARICAMENTO E PREPROCESSING DATI
# ============================================================================

def load_and_preprocess_data1(csv_path='archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv'):
    """
    Carica e preprocessa il dataset data1.csv
    
    Returns:
    --------
    Tuple: (X_train, y_train, X_test, y_test, baseline_accuracy)
    """
    print(f"Caricamento dataset da {csv_path}...")
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"Shape originale: X={X.shape}, y={y.shape}")
    print("\nPreprocessing dei dati...")

    # Rimuovi colonne con troppi NaN
    nan_percentage = np.isnan(X).sum(axis=0) / X.shape[0]
    cols_to_keep = nan_percentage < 0.5
    print(f"Rimosse {(~cols_to_keep).sum()} colonne con >50% NaN")
    X = X[:, cols_to_keep]

    # Sostituisci infiniti con NaN
    X = np.where(np.isinf(X), np.nan, X)
    print(f"Valori infiniti sostituiti con NaN")

    # Imputa i valori mancanti
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    print(f"NaN imputati con la mediana")

    # Clip outliers
    percentile_low = np.percentile(X, 1, axis=0)
    percentile_high = np.percentile(X, 99, axis=0)
    X = np.clip(X, percentile_low, percentile_high)
    print(f"Valori clippati ai percentili 1-99")

    assert not np.any(np.isnan(X)), "Ancora presenti NaN dopo preprocessing!"
    assert not np.any(np.isinf(X)), "Ancora presenti infiniti dopo preprocessing!"

    print(f"Shape dopo preprocessing: X={X.shape}")

    # Encoding target
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"\nClassi trovate: {le.classes_}")
    print(f"Distribuzione classi: {pd.Series(y).value_counts().to_dict()}")

    # Normalizzazione
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Features normalizzate con StandardScaler")

    # Conversione a tensori
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(f"\nShape finale: X={X.shape}, y={y.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Calcolo baseline
    class_counts = Counter(y_test.numpy())
    baseline_accuracy = max(class_counts.values()) / len(y_test)
    print(f"\nBASELINE (always predict majority class): {baseline_accuracy:.4f}")

    return X_train, y_train, X_test, y_test, baseline_accuracy


# ============================================================================
# ESPERIMENTI CON PERDITA MESSAGGI
# ============================================================================

def run_experiment_with_message_loss(
    drop_rates: List[float],
    csv_path: str = 'archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv',
    n_rounds: int = 200,
    n_nodes: int = 5,
    n_sampled: int = 4,
    m_top: int = 2,
    step1_rounds: int = 100
) -> Dict:
    """
    Esegue esperimenti PENS con diversi tassi di perdita di messaggi
    """
    
    try:
        X_train, y_train, X_test, y_test, baseline_accuracy = load_and_preprocess_data1(csv_path)
    except FileNotFoundError:
        print(f"Errore: File '{csv_path}' non trovato!")
        return None
    except Exception as e:
        print(f"Errore nel caricamento dataset: {e}")
        return None
    
    input_size = X_train.shape[1]
    
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE ESPERIMENTI")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}")
    print(f"Input size: {input_size}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"\nParametri PENS:")
    print(f"  Nodi: {n_nodes}")
    print(f"  N_sampled: {n_sampled}")
    print(f"  M_top: {m_top}")
    print(f"  Step1_rounds: {step1_rounds}")
    print(f"  Total rounds: {n_rounds}")
    print(f"\nDrop rates da testare: {drop_rates}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for drop_rate in drop_rates:
        print(f"\n{'='*60}")
        print(f"ESPERIMENTO CON DROP_RATE = {drop_rate:.2f}")
        print(f"{'='*60}")
        
        data_handler = ClassificationDataHandler(
            X_train, y_train,
            X_test, y_test
        )
        
        data_dispatcher = CustomDataDispatcher(
            data_handler, 
            n=n_nodes, 
            eval_on_user=True,
            auto_assign=True
        )
        
        nodes = FixedPENSNode.generate(
            data_dispatcher=data_dispatcher,
            p2p_net=StaticP2PNetwork(n_nodes),
            model_proto=FixedTorchModelHandler(
                net=Data1Net(input_dim=input_size, hidden_dims=(128, 64, 32)),
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
            n_sampled=n_sampled,
            m_top=m_top,
            step1_rounds=step1_rounds
        )
        
        simulator = GossipSimulator(
            nodes=nodes,
            data_dispatcher=data_dispatcher,
            delta=100,
            protocol=AntiEntropyProtocol.PUSH,
            sampling_eval=0.2,
            drop_prob=drop_rate
        )
        
        report = FixedSimulationReport(delta=100)
        simulator.add_receiver(report)
        simulator.init_nodes(seed=42)
        
        print(f"Avvio simulazione PENS con drop_rate={drop_rate:.2f}...")
        simulator.start(n_rounds=n_rounds)
        
        # Calcola train accuracy media su tutti i nodi
        train_accs = []
        for node_id, node in nodes.items():
            X_train_node, y_train_node = node.data[0]
            train_eval = node.evaluate((X_train_node, y_train_node))
            train_accs.append(train_eval['accuracy'])
        
        avg_train_acc = np.mean(train_accs)
        
        evaluations = report.get_evaluation(False)
        if evaluations:
            final_metrics = evaluations[-1][1] if evaluations else None
            global_test_acc = final_metrics['accuracy'] if final_metrics else 0
            global_gap = avg_train_acc - global_test_acc
            
            results[drop_rate] = {
                'evaluations': evaluations,
                'final_metrics': final_metrics,
                'sent_messages': report._sent_messages,
                'failed_messages': report._failed_messages,
                'baseline_accuracy': baseline_accuracy,
                'global_train_acc': avg_train_acc,
                'global_test_acc': global_test_acc,
                'global_gap': global_gap
            }
            
            print(f"\nRisultati per drop_rate={drop_rate:.2f}:")
            print(f"  Messaggi inviati: {report._sent_messages}")
            print(f"  Messaggi persi: {report._failed_messages} ({report._failed_messages/max(1,report._sent_messages)*100:.1f}%)")
            if final_metrics:
                for metric, value in final_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            print(f"  Train acc (media nodi): {avg_train_acc:.4f}")
            print(f"  Test acc (globale): {global_test_acc:.4f}")
            print(f"  Gap (overfitting): {global_gap:.4f} ({global_gap*100:.2f}%)")
    
    return results


# ============================================================================
# VISUALIZZAZIONE RISULTATI
# ============================================================================

def plot_results(results: Dict):
    """
    Visualizza i risultati degli esperimenti
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impatto della perdita di messaggi sull\'addestramento PENS', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: Evoluzione accuracy nel tempo
    ax1 = axes[0, 0]
    for i, (drop_rate, data) in enumerate(results.items()):
        if data['evaluations']:
            rounds = [r for r, _ in data['evaluations']]
            accuracies = [metrics['accuracy'] for _, metrics in data['evaluations']]
            ax1.plot(rounds, accuracies, label=f'drop={drop_rate:.2f}', 
                    color=colors[i], linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Evoluzione Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy finale vs drop rate
    ax2 = axes[0, 1]
    drop_rates = list(results.keys())
    final_accuracies = [data['final_metrics']['accuracy'] if data['final_metrics'] else 0 
                       for data in results.values()]
    ax2.plot(drop_rates, final_accuracies, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Drop Rate')
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Accuracy Finale vs Tasso di Perdita')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Messaggi persi vs inviati
    ax3 = axes[1, 0]
    sent_messages = [data['sent_messages'] for data in results.values()]
    failed_messages = [data['failed_messages'] for data in results.values()]
    x = np.arange(len(drop_rates))
    width = 0.35
    ax3.bar(x - width/2, sent_messages, width, label='Inviati', alpha=0.8)
    ax3.bar(x + width/2, failed_messages, width, label='Persi', alpha=0.8)
    ax3.set_xlabel('Drop Rate')
    ax3.set_ylabel('Numero di Messaggi')
    ax3.set_title('Statistiche Messaggi')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{dr:.2f}' for dr in drop_rates])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: F1 Score finale vs drop rate
    ax4 = axes[1, 1]
    final_f1 = [data['final_metrics']['f1_score'] if data['final_metrics'] else 0 
                for data in results.values()]
    ax4.plot(drop_rates, final_f1, 's-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('Drop Rate')
    ax4.set_ylabel('Final F1 Score')
    ax4.set_title('F1 Score Finale vs Tasso di Perdita')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_summary_table(results: Dict):
    """
    Stampa una tabella riassuntiva dei risultati
    """
    print("\n" + "="*100)
    print("TABELLA RIASSUNTIVA RISULTATI")
    print("="*100)
    print(f"{'Drop Rate':<12} {'Accuracy':<12} {'F1 Score':<12} {'Train Acc':<12} "
          f"{'Gap':<12} {'Msg Lost':<15} {'vs Baseline':<15}")
    print("-"*100)
    
    for drop_rate in sorted(results.keys()):
        data = results[drop_rate]
        if data['final_metrics']:
            acc = data['final_metrics']['accuracy']
            f1 = data['final_metrics']['f1_score']
            train_acc = data['global_train_acc']
            gap = data['global_gap']
            baseline = data['baseline_accuracy']
            improvement = ((acc - baseline) / baseline * 100)
            
            msg_lost_pct = (data['failed_messages'] / max(1, data['sent_messages']) * 100)
            
            print(f"{drop_rate:<12.2f} {acc:<12.4f} {f1:<12.4f} {train_acc:<12.4f} "
                  f"{gap:<12.4f}      {msg_lost_pct:<13.1f}% {improvement:+12.2f}%")
    
    print("="*100 + "\n")


# ============================================================================
# ESECUZIONE PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    
    DROP_RATES = [0.1, 0.2, 0.3, 0.5, 0.7]
    CSV_PATH = 'archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv'
    N_ROUNDS = 200
    N_NODES = 5
    N_SAMPLED = 4
    M_TOP = 2
    STEP1_ROUNDS = 100
    
    print("="*60)
    print("STUDIO DELL'IMPATTO DELLA PERDITA DI MESSAGGI SU PENS")
    print("="*60)
    print("Versione con fix completi")
    print("="*60 + "\n")
    
    results = run_experiment_with_message_loss(
        drop_rates=DROP_RATES,
        csv_path=CSV_PATH,
        n_rounds=N_ROUNDS,
        n_nodes=N_NODES,
        n_sampled=N_SAMPLED,
        m_top=M_TOP,
        step1_rounds=STEP1_ROUNDS
    )
    
    if results:
        print_summary_table(results)
        plot_results(results)
        print("\nEsperimenti completati con successo!")
    else:
        print("\nErrore durante l'esecuzione degli esperimenti")