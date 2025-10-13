"""
Modulo per il calcolo di misure di divergenza tra modelli di machine learning.
Implementa Jensen-Rényi, KL e Jensen-Shannon divergence.
"""

import numpy as np
import torch
import torch.nn as nn


def extract_model_weights(model):
    """
    Estrae e concatena tutti i pesi del modello in un singolo vettore.
    
    Args:
        model: Modello PyTorch o parametri del modello
        
    Returns:
        np.ndarray: Vettore concatenato dei pesi del modello
    """
    if isinstance(model, nn.Module):
        return np.concatenate([param.detach().cpu().numpy().flatten() 
                              for param in model.parameters()])
    elif isinstance(model, list):
        # Per il caso di pesi già estratti (come in FL)
        return np.concatenate([w.flatten() for w in model])
    else:
        return np.array(model).flatten()


def softmax(x):
    """
    Applica la funzione softmax per normalizzare in distribuzione di probabilità.
    
    Args:
        x: array-like, valori di input
        
    Returns:
        np.ndarray: distribuzione di probabilità normalizzata
    """
    x = np.asarray(x)
    # Stabilità numerica
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def renyi_entropy(p, alpha=2):
    """
    Calcola l'entropia di Rényi di ordine alpha.
    
    Args:
        p: array-like, distribuzione di probabilità
        alpha: float, ordine dell'entropia (alpha > 0, alpha != 1)
        
    Returns:
        float: entropia di Rényi
    """
    p = np.asarray(p)
    if alpha <= 0 or np.isclose(alpha, 1.0):
        raise ValueError("Alpha deve essere > 0 e != 1.")
    
    # Evita problemi numerici
    p = np.clip(p, 1e-12, 1)
    return (1.0 / (1 - alpha)) * np.log(np.sum(p ** alpha))


def jensen_renyi_divergence(w1, w2, alpha=2):
    """
    Calcola la divergenza Jensen-Rényi tra due vettori di pesi.
    
    Args:
        w1, w2: array-like, vettori di pesi dei modelli
        alpha: float, ordine dell'entropia di Rényi
        
    Returns:
        float: divergenza Jensen-Rényi
    """
    # Normalizza i pesi in distribuzioni di probabilità
    p = softmax(w1)
    q = softmax(w2)
    m = 0.5 * (p + q)
    
    h_m = renyi_entropy(m, alpha)
    h_p = renyi_entropy(p, alpha)
    h_q = renyi_entropy(q, alpha)
    
    jr = h_m - 0.5 * (h_p + h_q)
    return jr


def kl_divergence(p, q):
    """
    Calcola la divergenza di Kullback-Leibler D_KL(P || Q).
    
    Args:
        p, q: array-like, distribuzioni di probabilità
        
    Returns:
        float: divergenza KL
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Evita divisione per zero
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    
    return np.sum(p * np.log(p / q))


def jensen_shannon_divergence(p, q):
    """
    Calcola la divergenza Jensen-Shannon tra due distribuzioni.
    
    Args:
        p, q: array-like, distribuzioni di probabilità
        
    Returns:
        float: divergenza JS
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_from_raw_weights(w1, w2):
    """
    Calcola KL divergence tra vettori di pesi usando normalizzazione softmax.
    
    Args:
        w1, w2: array-like, vettori di pesi
        
    Returns:
        float: divergenza KL
    """
    p = softmax(w1)
    q = softmax(w2)
    return kl_divergence(p, q)


def js_from_raw_weights(w1, w2):
    """
    Calcola JS divergence tra vettori di pesi usando normalizzazione softmax.
    
    Args:
        w1, w2: array-like, vettori di pesi
        
    Returns:
        float: divergenza JS
    """
    p = softmax(w1)
    q = softmax(w2)
    return jensen_shannon_divergence(p, q)


class DivergenceTracker:
    """
    Classe per tracciare le divergenze durante l'addestramento.
    """
    
    def __init__(self):
        self.divergences = {
            'jensen_renyi': [],
            'kl': [],
            'js': [],
            'rounds': []
        }
    
    def add_measurement(self, round_num, w1, w2):
        """
        Aggiunge una misurazione di divergenza.
        
        Args:
            round_num: int, numero del round
            w1, w2: array-like, vettori di pesi da confrontare
        """
        try:
            jr = jensen_renyi_divergence(w1, w2)
            kl = kl_from_raw_weights(w1, w2)
            js = js_from_raw_weights(w1, w2)
            
            self.divergences['jensen_renyi'].append(jr)
            self.divergences['kl'].append(kl)
            self.divergences['js'].append(js)
            self.divergences['rounds'].append(round_num)
            
        except Exception as e:
            print(f"Errore nel calcolo divergenze round {round_num}: {e}")
    
    def get_latest_divergences(self):
        """Restituisce le ultime divergenze calcolate."""
        if not self.divergences['rounds']:
            return None
        return {
            'jensen_renyi': self.divergences['jensen_renyi'][-1],
            'kl': self.divergences['kl'][-1],
            'js': self.divergences['js'][-1],
            'round': self.divergences['rounds'][-1]
        }
    
    def get_all_divergences(self):
        """Restituisce tutte le divergenze tracciate."""
        return self.divergences.copy()