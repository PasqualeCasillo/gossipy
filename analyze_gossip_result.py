"""
Analisi focalizzata solo sulle heatmap delle divergenze, senza emoji.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from divergence_metrics import jensen_renyi_divergence, kl_from_raw_weights, js_from_raw_weights
import seaborn as sns

def load_gossip_data(filename="gossip_divergence_data.pkl"):
    """Carica i dati salvati dal Gossip Learning"""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} non trovato!")
        return None

def calculate_all_divergences(node_weights_history):
    """Calcola tutte le metriche di divergenza"""
    
    # Prendi gli ultimi pesi di ogni nodo
    final_weights = {}
    for node_id, weights_list in node_weights_history.items():
        if weights_list:
            final_weights[node_id] = weights_list[-1]
    
    node_ids = list(final_weights.keys())
    n_nodes = len(node_ids)
    
    # Matrici per tutte le divergenze
    jr_matrix = np.zeros((n_nodes, n_nodes))
    kl_matrix = np.zeros((n_nodes, n_nodes))
    js_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                try:
                    w1 = final_weights[node_ids[i]]
                    w2 = final_weights[node_ids[j]]
                    
                    jr_matrix[i, j] = jensen_renyi_divergence(w1, w2, alpha=2)
                    kl_matrix[i, j] = kl_from_raw_weights(w1, w2)
                    js_matrix[i, j] = js_from_raw_weights(w1, w2)
                    
                except Exception as e:
                    print(f"Errore divergenza nodi {i}-{j}: {e}")
                    jr_matrix[i, j] = np.nan
                    kl_matrix[i, j] = np.nan
                    js_matrix[i, j] = np.nan
    
    return jr_matrix, kl_matrix, js_matrix, node_ids

def create_clean_heatmaps(jr_matrix, kl_matrix, js_matrix, node_ids):
    """Crea heatmap pulite e professionali"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Configurazione comune
    mask = np.eye(len(node_ids), dtype=bool)
    
    # Parametri per le heatmap
    heatmap_configs = [
        {'matrix': jr_matrix, 'title': 'Jensen-Renyi Divergence', 'cmap': 'RdYlBu_r'},
        {'matrix': kl_matrix, 'title': 'Kullback-Leibler Divergence', 'cmap': 'plasma'},
        {'matrix': js_matrix, 'title': 'Jensen-Shannon Divergence', 'cmap': 'viridis'}
    ]
    
    for idx, config in enumerate(heatmap_configs):
        ax = axes[idx]
        matrix = config['matrix']
        
        # Calcola range valori per colorbar consistente
        valid_values = matrix[~mask & ~np.isnan(matrix)]
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        
        # Crea heatmap
        im = sns.heatmap(
            matrix,
            mask=mask,
            annot=False,
            cmap=config['cmap'],
            square=True,
            linewidths=0.3,
            cbar_kws={"shrink": 0.8, "format": "%.1e"},
            xticklabels=node_ids,
            yticklabels=node_ids,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )
        
        # Titolo e etichette
        ax.set_title(config['title'], fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Node ID', fontsize=12)
        ax.set_ylabel('Node ID', fontsize=12)
        
        # Statistiche nell'angolo
        stats_text = f'Mean: {np.mean(valid_values):.2e}\nStd: {np.std(valid_values):.2e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                         edgecolor='black', linewidth=1))
    
    plt.suptitle('Divergence Analysis - Gossip Learning Final State', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Salva con alta qualità
    plt.savefig('gossip_heatmaps_clean.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Heatmaps salvate: gossip_heatmaps_clean.png")
    
    plt.show()

def create_individual_heatmaps(jr_matrix, kl_matrix, js_matrix, node_ids):
    """Crea heatmap individuali più grandi"""
    
    matrices = [
        (jr_matrix, 'Jensen-Renyi Divergence', 'RdYlBu_r'),
        (kl_matrix, 'Kullback-Leibler Divergence', 'plasma'),
        (js_matrix, 'Jensen-Shannon Divergence', 'viridis')
    ]
    
    mask = np.eye(len(node_ids), dtype=bool)
    
    for i, (matrix, title, cmap) in enumerate(matrices):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcola statistiche
        valid_values = matrix[~mask & ~np.isnan(matrix)]
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        # Heatmap
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt='.2e',
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "format": "%.2e"},
            xticklabels=node_ids,
            yticklabels=node_ids,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            annot_kws={'fontsize': 6}
        )
        
        ax.set_title(f'{title}\nMean: {mean_val:.2e}, Std: {std_val:.2e}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Node ID', fontsize=12)
        ax.set_ylabel('Node ID', fontsize=12)
        
        plt.tight_layout()
        filename = f'gossip_{title.lower().replace(" ", "_").replace("-", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Salvata: {filename}")
        plt.show()

def print_detailed_statistics(jr_matrix, kl_matrix, js_matrix):
    """Stampa statistiche dettagliate senza emoji"""
    
    print("\n" + "="*70)
    print("                    STATISTICAL ANALYSIS")
    print("="*70)
    
    mask = ~np.eye(jr_matrix.shape[0], dtype=bool)
    
    metrics_data = {
        'Jensen-Renyi': jr_matrix[mask & ~np.isnan(jr_matrix)],
        'Kullback-Leibler': kl_matrix[mask & ~np.isnan(kl_matrix)],
        'Jensen-Shannon': js_matrix[mask & ~np.isnan(js_matrix)]
    }
    
    for name, values in metrics_data.items():
        print(f"\n{name} Divergence:")
        print(f"  Sample size:     {len(values)}")
        print(f"  Mean:           {np.mean(values):.8f}")
        print(f"  Median:         {np.median(values):.8f}")
        print(f"  Standard Dev:   {np.std(values):.8f}")
        print(f"  Min:            {np.min(values):.8f}")
        print(f"  Max:            {np.max(values):.8f}")
        print(f"  Range:          {np.max(values) - np.min(values):.8f}")
        print(f"  Coeff. Var:     {np.std(values)/np.mean(values):.4f}")
    
    print("\n" + "="*70)
    print("CONVERGENCE QUALITY ASSESSMENT:")
    
    jr_mean = np.mean(metrics_data['Jensen-Renyi'])
    if jr_mean < 1e-5:
        quality = "EXCELLENT"
        description = "Perfect consensus achieved"
    elif jr_mean < 1e-3:
        quality = "GOOD"
        description = "Strong consensus achieved"
    elif jr_mean < 1e-1:
        quality = "MODERATE"
        description = "Partial consensus achieved"
    else:
        quality = "POOR"
        description = "Limited consensus"
    
    print(f"  Overall Quality: {quality}")
    print(f"  Status:         {description}")
    print(f"  Threshold:      < 1e-5 (Excellent), < 1e-3 (Good), < 1e-1 (Moderate)")
    print("="*70)

def main():
    """Funzione principale per analisi focalizzata sulle heatmap"""
    
    print("Analisi divergenze Gossip Learning - Focus Heatmaps")
    print("-" * 55)
    
    # Carica dati
    data = load_gossip_data()
    if data is None:
        return
    
    node_weights_history = data['node_weights_history']
    config = data['config']
    
    print(f"Configurazione: {config['n_nodes']} nodi, {config['rounds']} round")
    
    # Calcola divergenze
    print("Calcolo delle divergenze...")
    jr_matrix, kl_matrix, js_matrix, node_ids = calculate_all_divergences(node_weights_history)
    
    # Statistiche dettagliate
    print_detailed_statistics(jr_matrix, kl_matrix, js_matrix)
    
    # Heatmaps combinate
    print("\nGenerazione heatmaps combinate...")
    create_clean_heatmaps(jr_matrix, kl_matrix, js_matrix, node_ids)
    
    # Heatmaps individuali (opzionale)
    response = input("\nVuoi generare anche le heatmaps individuali? (y/n): ")
    if response.lower() == 'y':
        print("Generazione heatmaps individuali...")
        create_individual_heatmaps(jr_matrix, kl_matrix, js_matrix, node_ids)
    
    print("\nAnalisi completata.")
    print("File generati:")
    print("- gossip_heatmaps_clean.png (combinato)")
    if response.lower() == 'y':
        print("- gossip_jensen_renyi_divergence.png")
        print("- gossip_kullback_leibler_divergence.png") 
        print("- gossip_jensen_shannon_divergence.png")

if __name__ == "__main__":
    main()