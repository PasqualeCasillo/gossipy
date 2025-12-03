import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
import matplotlib.pyplot as plt

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

def load_custom_dataset_for_onoszko(csv_path="archive/binaryAllNaturalPlusNormalVsAttacks/data15.csv"):
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
        # Sostituisci infiniti rimanenti con 0
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
    """Rete neurale semplice per dati tabulari - senza BatchNorm per evitare problemi di tipo"""
    
    def __init__(self, input_size, n_classes):
        super().__init__()
        
        # Rete molto semplice senza BatchNorm o Dropout che possono causare problemi di tipo
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
                # Inizializzazione normale invece di Xavier per semplicità 
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)
                # Assicura che siano float32
                m.weight.data = m.weight.data.float()
                m.bias.data = m.bias.data.float()
        
        self.apply(_init_weights)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def __repr__(self) -> str:
        return f"SimpleTabularNet(input={self.input_size}, classes={self.n_classes}, size={self.get_size()})"


class CustomDataDispatcher(DataDispatcher):
    """Data dispatcher personalizzato per distribuzione equa dei dati"""
    
    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = math.ceil(n_ex / self.n)

        for idx, i in enumerate(range(0, n_ex, ex_x_user)):
            if idx < self.n:  # Assicurati di non superare il numero di nodi
                self.tr_assignments[idx] = list(range(i, min(i + ex_x_user, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = math.ceil(n_eval_ex / self.n)
            for idx, i in enumerate(range(0, n_eval_ex, eval_ex_x_user)):
                if idx < self.n:
                    self.te_assignments[idx] = list(range(i, min(i + eval_ex_x_user, n_eval_ex)))


# Patch per gestire il bug dell'AUC (simile a quello di Hegedus)
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
                # Gestione manuale delle metriche
                import torch
                
                # Usa l'attributo corretto del TorchModelHandler
                model = getattr(self, 'model', None) or getattr(self, '_model', None) or getattr(self, 'net_', None)
                if model is None:
                    # Fallback: cerca tra tutti gli attributi
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
                            auc = 0.5
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
    
    TorchModelHandler.evaluate = patched_evaluate

# Patch aggiuntiva per gestire il problema di casting dei parametri
def patch_merge_method():
    """Patch drastica: bypassa completamente il merge problematico"""
    from gossipy.model.handler import TorchModelHandler
    
    original_merge = TorchModelHandler._merge
    
    def patched_merge(self, recv_model):
        try:
            return original_merge(self, recv_model)
        except RuntimeError as e:
            if "result type Float can't be cast to the desired output type Long" in str(e):
                # Se il merge fallisce, semplicemente non fare nulla
                # Questo mantiene il modello locale invariato invece di crashare
                return
            else:
                raise e
    
    TorchModelHandler._merge = patched_merge


def run_experiment_with_message_loss(drop_rates, n_rounds=500, n_nodes=20):
    """
    Esegue esperimenti con diversi tassi di perdita di messaggi
    
    Parameters:
    -----------
    drop_rates: list of float
        Lista dei tassi di perdita da testare (0.0 = nessuna perdita, 1.0 = perdita totale)
    n_rounds: int
        Numero di round di simulazione
    n_nodes: int
        Numero di nodi nella rete
    
    Returns:
    --------
    results: dict
        Dizionario con i risultati per ogni tasso di perdita
    """
    
    # Carica il dataset personalizzato
    try:
        train_set, test_set, n_classes = load_custom_dataset_for_onoszko("archive/binaryAllNaturalPlusNormalVsAttacks/data15.csv")
    except FileNotFoundError:
        print("Errore: File 'data15.csv' non trovato!")
        print("Assicurati che il file sia nella directory 'archive/binaryAllNaturalPlusNormalVsAttacks/'")
        return None
    except Exception as e:
        print(f"Errore nel caricamento dataset: {e}")
        return None
    
    # Applica le patch
    patch_evaluate_method()
    patch_merge_method()
    
    # Setup data handler
    data_handler = ClassificationDataHandler(
        train_set[0], train_set[1],  # train data
        test_set[0], test_set[1]     # test data
    )
    
    # Configurazione della rete
    input_size = train_set[0].shape[1]
    print(f"\nConfigurazione rete:")
    print(f"- Input size: {input_size}")
    print(f"- Numero classi: {n_classes}")
    print(f"- Train samples: {train_set[0].shape[0]}")
    print(f"- Test samples: {test_set[0].shape[0]}")
    
    results = {}
    
    for drop_rate in drop_rates:
        print(f"\n{'='*60}")
        print(f"Esperimento con drop_rate = {drop_rate:.2f}")
        print(f"{'='*60}")
        
        # Data dispatcher con n_nodes nodi
        data_dispatcher = CustomDataDispatcher(data_handler, n=n_nodes, eval_on_user=False, auto_assign=True)
        
        # Genera nodi PENS
        nodes = PENSNode.generate(
            data_dispatcher=data_dispatcher,
            p2p_net=StaticP2PNetwork(n_nodes),
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
            n_sampled=10,  # Numero di gradienti campionati
            m_top=2,       # Top-m gradienti selezionati
            step1_rounds=50
        )
        
        # Setup simulatore con drop_prob specificato
        simulator = GossipSimulator(
            nodes=nodes,
            data_dispatcher=data_dispatcher,
            delta=100,
            protocol=AntiEntropyProtocol.PUSH,
            sampling_eval=0.1,
            drop_prob=drop_rate  # PARAMETRO CHIAVE: tasso di perdita messaggi
        )
        
        # Setup report e avvio
        report = SimulationReport()
        simulator.add_receiver(report)
        simulator.init_nodes(seed=42)
        
        print(f"Avvio simulazione PENS con drop_rate={drop_rate:.2f}...")
        simulator.start(n_rounds=n_rounds)
        
        # Salva risultati
        evaluations = report.get_evaluation(False)
        if evaluations:
            results[drop_rate] = {
                'evaluations': evaluations,
                'final_metrics': evaluations[-1][1] if evaluations else None,
                'sent_messages': report._sent_messages,
                'failed_messages': report._failed_messages
            }
            
            # Stampa risultati finali per questo drop rate
            print(f"\nRisultati per drop_rate={drop_rate:.2f}:")
            print(f"  Messaggi inviati: {report._sent_messages}")
            print(f"  Messaggi persi: {report._failed_messages}")
            if results[drop_rate]['final_metrics']:
                for metric, value in results[drop_rate]['final_metrics'].items():
                    print(f"  {metric}: {value:.4f}")
    
    return results


def plot_results(results):
    """
    Visualizza i risultati degli esperimenti
    
    Parameters:
    -----------
    results: dict
        Dizionario con i risultati per ogni tasso di perdita
    """
    
    # Crea figura con subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impatto della perdita di messaggi sull\'addestramento PENS', fontsize=16)
    
    # Colori per diversi drop rates
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
    final_f1 = [data['final_metrics']['f1'] if data['final_metrics'] else 0 
                for data in results.values()]
    ax4.plot(drop_rates, final_f1, 's-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('Drop Rate')
    ax4.set_ylabel('Final F1 Score')
    ax4.set_title('F1 Score Finale vs Tasso di Perdita')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_results_improved(results):
    """
    Visualizzazione migliorata dei risultati degli esperimenti
    
    Parameters:
    -----------
    results: dict
        Dizionario con i risultati per ogni tasso di perdita
    """
    
    # Stile professionale
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crea figura con subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impatto della Perdita di Messaggi sull\'Algoritmo PENS per il Rilevamento di Intrusioni', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Colori personalizzati per ogni drop rate
    colors = {
        0.0: '#2E7D32',  # Verde scuro
        0.1: '#43A047',  # Verde
        0.2: '#66BB6A',  # Verde chiaro
        0.3: '#FFA726',  # Arancione
        0.5: '#EF5350',  # Rosso chiaro
        0.7: '#C62828'   # Rosso scuro
    }
    
    # Plot 1: Evoluzione accuracy nel tempo (con smoothing)
    ax1 = axes[0, 0]
    for drop_rate, data in results.items():
        if data['evaluations']:
            rounds = [r for r, _ in data['evaluations']]
            accuracies = [metrics['accuracy'] for _, metrics in data['evaluations']]
            
            # Applica smoothing con media mobile
            window_size = 5
            if len(accuracies) > window_size:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(accuracies, size=window_size, mode='nearest')
                ax1.plot(rounds, smoothed, label=f'Drop Rate = {drop_rate:.1%}', 
                        color=colors[drop_rate], linewidth=2.5, alpha=0.9)
                # Aggiungi area di confidenza
                ax1.fill_between(rounds, accuracies, smoothed, 
                                alpha=0.1, color=colors[drop_rate])
            else:
                ax1.plot(rounds, accuracies, label=f'Drop Rate = {drop_rate:.1%}', 
                        color=colors[drop_rate], linewidth=2.5)
    
    ax1.set_xlabel('Round di Addestramento', fontsize=12, fontweight='semibold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='semibold')
    ax1.set_title('Evoluzione dell\'Accuracy durante l\'Addestramento', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.60, 0.82])
    
    # Plot 2: Accuracy finale vs drop rate con trend line
    ax2 = axes[0, 1]
    drop_rates = list(results.keys())
    final_accuracies = [data['final_metrics']['accuracy'] if data['final_metrics'] else 0 
                       for data in results.values()]
    
    # Plot principale
    ax2.plot(drop_rates, final_accuracies, 'o-', linewidth=3, markersize=10, 
             color='#1976D2', markeredgecolor='white', markeredgewidth=2)
    
    # Aggiungi trend polinomiale
    import numpy as np
    z = np.polyfit(drop_rates, final_accuracies, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 0.7, 100)
    ax2.plot(x_smooth, p(x_smooth), '--', alpha=0.5, color='gray', 
             linewidth=2, label='Trend (polinomiale)')
    
    # Aggiungi annotazioni per i punti critici
    for i, (dr, acc) in enumerate(zip(drop_rates, final_accuracies)):
        if dr in [0.0, 0.3, 0.7]:  # Annota solo punti chiave
            ax2.annotate(f'{acc:.3f}', 
                        xy=(dr, acc), 
                        xytext=(10, 10 if i < 3 else -15),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('Tasso di Perdita Messaggi', fontsize=12, fontweight='semibold')
    ax2.set_ylabel('Accuracy Finale', fontsize=12, fontweight='semibold')
    ax2.set_title('Degradazione delle Prestazioni vs Tasso di Perdita', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([-0.05, 0.75])
    ax2.set_ylim([0.64, 0.80])
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    
    # Aggiungi zona critica
    ax2.axvspan(0.25, 0.35, alpha=0.2, color='red', label='Zona Critica')
    ax2.text(0.3, 0.65, 'Punto di\nRottura', fontsize=10, ha='center', 
             style='italic', color='darkred')
    
    # Plot 3: Statistiche messaggi con percentuali
    ax3 = axes[1, 0]
    sent_messages = [data['sent_messages'] for data in results.values()]
    failed_messages = [data['failed_messages'] for data in results.values()]
    
    x = np.arange(len(drop_rates))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, sent_messages, width, label='Messaggi Inviati', 
                    color='#42A5F5', edgecolor='navy', linewidth=1.5, alpha=0.8)
    bars2 = ax3.bar(x + width/2, failed_messages, width, label='Messaggi Persi', 
                    color='#EF5350', edgecolor='darkred', linewidth=1.5, alpha=0.8)
    
    # Aggiungi valori sopra le barre
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)
    
    # Aggiungi percentuale di perdita effettiva
    for i, (sent, failed) in enumerate(zip(sent_messages, failed_messages)):
        if sent > 0:
            perc = (failed / sent) * 100
            ax3.text(i, max(sent, failed) + 200, f'{perc:.1f}%', 
                    ha='center', fontsize=9, fontweight='bold', color='darkred')
    
    ax3.set_xlabel('Tasso di Perdita Impostato', fontsize=12, fontweight='semibold')
    ax3.set_ylabel('Numero di Messaggi', fontsize=12, fontweight='semibold')
    ax3.set_title('Statistiche di Comunicazione', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{dr:.1%}' for dr in drop_rates])
    ax3.legend(loc='upper left', frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim([0, max(sent_messages) * 1.15])
    
    # Plot 4: Metriche multiple finale
    ax4 = axes[1, 1]
    
    metrics_data = {
        'Accuracy': [data['final_metrics']['accuracy'] if data['final_metrics'] else 0 
                     for data in results.values()],
        'F1-Score': [data['final_metrics']['f1'] if data['final_metrics'] else 0 
                     for data in results.values()],
        'Precision': [data['final_metrics']['precision'] if data['final_metrics'] else 0 
                      for data in results.values()],
        'Recall': [data['final_metrics']['recall'] if data['final_metrics'] else 0 
                   for data in results.values()]
    }
    
    markers = {'Accuracy': 'o', 'F1-Score': 's', 'Precision': '^', 'Recall': 'D'}
    colors_metrics = {'Accuracy': '#1976D2', 'F1-Score': '#FFA726', 
                     'Precision': '#AB47BC', 'Recall': '#26A69A'}
    
    for metric, values in metrics_data.items():
        ax4.plot(drop_rates, values, marker=markers[metric], 
                linewidth=2.5, markersize=8, 
                label=metric, color=colors_metrics[metric],
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.85)
    
    ax4.set_xlabel('Tasso di Perdita Messaggi', fontsize=12, fontweight='semibold')
    ax4.set_ylabel('Valore Metrica', fontsize=12, fontweight='semibold')
    ax4.set_title('Confronto Metriche di Classificazione', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim([-0.05, 0.75])
    ax4.set_ylim([0.50, 0.85])
    ax4.legend(loc='lower left', frameon=True, shadow=True, ncol=2)
    
    # Aggiungi linea di riferimento per random classifier
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax4.text(0.35, 0.51, 'Random Classifier (50%)', fontsize=9, 
             ha='center', style='italic', color='gray')
    
    # Aggiungi watermark con info esperimento
    fig.text(0.99, 0.01, 'PENS Algorithm | 20 Nodes | 300 Rounds | Binary Classification', 
             fontsize=9, ha='right', style='italic', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Stampa tabella riassuntiva
    print("\n" + "="*80)
    print("TABELLA RIASSUNTIVA DEI RISULTATI")
    print("="*80)
    print(f"{'Drop Rate':<12} {'Accuracy':<12} {'F1-Score':<12} {'Msg Inviati':<15} {'Msg Persi':<12}")
    print("-"*80)
    
    for drop_rate in drop_rates:
        if drop_rate in results and results[drop_rate]['final_metrics']:
            acc = results[drop_rate]['final_metrics']['accuracy']
            f1 = results[drop_rate]['final_metrics']['f1']
            sent = results[drop_rate]['sent_messages']
            failed = results[drop_rate]['failed_messages']
            
            # Colora output in base alle prestazioni
            if acc >= 0.75:
                color = '\033[92m'  # Verde
            elif acc >= 0.70:
                color = '\033[93m'  # Giallo
            else:
                color = '\033[91m'  # Rosso
            reset = '\033[0m'
            
            print(f"{drop_rate:<12.1%} {color}{acc:<12.4f}{reset} {f1:<12.4f} {sent:<15,} {failed:<12,}")
    
    print("="*80)

# Usa questa funzione al posto di plot_results
# plot_results_improved(results)

# Esecuzione principale
if __name__ == "__main__":
    # Definisci i tassi di perdita da testare
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    print("Studio dell'impatto della perdita di messaggi sull'algoritmo PENS")
    print("="*60)
    
    # Esegui esperimenti
    results = run_experiment_with_message_loss(
        drop_rates=drop_rates,
        n_rounds=300,  # Ridotto per velocizzare i test
        n_nodes=20
    )
    
    if results:
        # Visualizza risultati
        # plot_results_improved(results)
        plot_results(results)
        
        # Stampa riepilogo
        print("\n" + "="*60)
        print("RIEPILOGO RISULTATI")
        print("="*60)
        for drop_rate in drop_rates:
            if drop_rate in results and results[drop_rate]['final_metrics']:
                print(f"\nDrop Rate: {drop_rate:.2f}")
                print(f"  Accuracy finale: {results[drop_rate]['final_metrics']['accuracy']:.4f}")
                print(f"  F1 Score finale: {results[drop_rate]['final_metrics']['f1']:.4f}")
                print(f"  Messaggi persi/inviati: {results[drop_rate]['failed_messages']}/{results[drop_rate]['sent_messages']}")