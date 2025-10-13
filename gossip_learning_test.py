from gossip_learning import GossipLearningNetwork, load_and_preprocess_data
from sklearn.model_selection import train_test_split

# Carica e preprocessa i dati
print("Caricamento dati...")
X, y = load_and_preprocess_data('archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Crea e allena la rete
print("Creazione rete...")
network = GossipLearningNetwork(
    n_nodes=500,              # Più nodi
    features=X_train.shape[1],
    learning_rate=0.02,      # LR più alto
    gossip_probability=0.3, # Più gossip
    topology='random'      # Topologia random
)

print("Inizio training...")
final_accuracy, final_loss = network.train(
    X_train, y_train, X_test, y_test, 
    epochs=200,  # Più training
    verbose=True
)
print(f"Risultati finali:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Loss: {final_loss:.4f}")

# Visualizza i risultati
network.plot_training_metrics()