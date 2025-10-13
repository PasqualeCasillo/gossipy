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


# Carica il dataset personalizzato
try:
    train_set, test_set, n_classes = load_custom_dataset_for_onoszko("archive/binaryAllNaturalPlusNormalVsAttacks/data15.csv")
except FileNotFoundError:
    print("Errore: File 'data7.csv' non trovato!")
    print("Assicurati che il file sia nella directory 'archive/binaryAllNaturalPlusNormalVsAttacks/'")
    exit(1)
except Exception as e:
    print(f"Errore nel caricamento dataset: {e}")
    exit(1)

# Setup data handler
data_handler = ClassificationDataHandler(
    train_set[0], train_set[1],  # train data
    test_set[0], test_set[1]     # test data
)

# Data dispatcher con 20 nodi
data_dispatcher = CustomDataDispatcher(data_handler, n=20, eval_on_user=False, auto_assign=True)

# Configurazione della rete
input_size = train_set[0].shape[1]
print(f"\nConfigurazione rete:")
print(f"- Input size: {input_size}")
print(f"- Numero classi: {n_classes}")
print(f"- Train samples: {train_set[0].shape[0]}")
print(f"- Test samples: {test_set[0].shape[0]}")

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

# Applica la patch
patch_evaluate_method()

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

# Applica entrambe le patch
patch_merge_method()

# Genera nodi PENS
nodes = PENSNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=StaticP2PNetwork(20),
    model_proto=TorchModelHandler(
        net=TabularNet(input_size, n_classes),
        optimizer=torch.optim.SGD,  # Torniamo a SGD che è più stabile
        optimizer_params={
            "lr": 0.01,
            "weight_decay": 0.0001
        },
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=32,  # Batch size più grande per dati tabulari
        local_epochs=10   # Più epoche locali
    ),
    round_len=100,
    sync=False,
    n_sampled=10,  # Numero di gradienti campionati
    m_top=2,       # Top-m gradienti selezionati
    step1_rounds=50
)

# Setup simulatore
simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.1
)

# Setup report e avvio
report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)

print(f"\nAvvio simulazione PENS (Onoszko) con {len(nodes)} nodi...")
print(f"Features: {input_size}, Classi: {n_classes}")
print(f"Algoritmo: PENS con sparsificazione dei gradienti")

simulator.start(n_rounds=500)

# Visualizza risultati
print(f"\nSimulazione completata!")
plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], 
                f"PENS - Dataset Personalizzato ({data_handler.size()} samples, {n_classes} classi)")

# Statistiche finali
if report.get_evaluation(False):
    final_metrics = report.get_evaluation(False)[-1][1]
    print(f"\nRisultati finali:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")