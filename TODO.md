# 📋 GUIDE - Projet IIC (Invariant Information Clustering)

> **Paper:** Invariant Information Clustering for Unsupervised Image Classification and Segmentation  
> **Objectif:** Implémenter IIC pour le clustering non supervisé d'images  
> **Dataset principal:** MNIST (pour commencer, plus simple)

---

# 🗒️ PARTIE 1: NOTEBOOK

---

## 1.1 Setup & Configuration

### Imports nécessaires
```python
# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Vision
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

# Utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
```

### Configuration à définir
- [ ] `device`: GPU ou CPU
- [ ] `seed`: Pour la reproductibilité (ex: 42)
- [ ] `batch_size`: 256 (comme dans le paper)
- [ ] `epochs`: 100-200
- [ ] `learning_rate`: 1e-4
- [ ] `num_clusters`: 10 (= nombre de classes MNIST)
- [ ] `num_overclusters`: 50 (optionnel, pour l'over-clustering)

---

## 1.2 Dataset & Transformations

### Concept clé: Les PAIRES d'images
> IIC utilise des **paires** (image originale, image transformée) pour apprendre.
> L'idée: une image et sa version transformée doivent avoir le MÊME cluster.

### Transformations à implémenter

#### Pour MNIST:
```python
# Transformation de base (normalisation)
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Transformation avec augmentation (pour créer la paire)
transform_aug = transforms.Compose([
    transforms.RandomAffine(
        degrees=(-25, 25),      # Rotation
        translate=(0.2, 0.2),   # Translation
        scale=(0.8, 1.2)        # Scale
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Dataset personnalisé à créer
- [ ] Créer une classe `IICDataset` qui hérite de `Dataset`
- [ ] La méthode `__getitem__` doit retourner:
  - `img`: image avec transform de base
  - `img_aug`: MÊME image avec transform augmenté
  - `label`: label réel (pour l'évaluation seulement, pas pour le training!)

```python
class IICDataset(Dataset):
    def __init__(self, dataset, transform_base, transform_aug):
        self.dataset = dataset
        self.transform_base = transform_base
        self.transform_aug = transform_aug
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Appliquer les deux transformations à la MÊME image PIL
        img_base = self.transform_base(img)
        img_aug = self.transform_aug(img)
        return img_base, img_aug, label
```

### DataLoaders
- [ ] `train_loader`: shuffle=True, drop_last=True
- [ ] `test_loader`: shuffle=False (pour évaluation)

---

## 1.3 Architecture du Modèle

### Structure du réseau IIC

```
Input Image (1x28x28 pour MNIST)
        │
        ▼
┌───────────────────┐
│   ENCODER (CNN)   │
│  - Conv layers    │
│  - BatchNorm      │
│  - ReLU           │
│  - MaxPool        │
└───────────────────┘
        │
        ▼
   Feature Vector
        │
        ▼
┌───────────────────┐
│  CLUSTERING HEAD  │
│  - Linear layers  │
│  - Softmax        │
└───────────────────┘
        │
        ▼
   Cluster Probabilities
   (taille: num_clusters)
```

### Architecture CNN pour MNIST
- [ ] **Conv Block 1**: Conv2d(1, 64, 5) → BatchNorm → ReLU → MaxPool(2)
- [ ] **Conv Block 2**: Conv2d(64, 128, 5) → BatchNorm → ReLU → MaxPool(2)
- [ ] **Conv Block 3**: Conv2d(128, 256, 5) → BatchNorm → ReLU
- [ ] **Flatten**
- [ ] **Clustering Head**: Linear → ReLU → Linear(num_clusters) → Softmax

### Implémentation
```python
class IICNet(nn.Module):
    def __init__(self, num_clusters=10):
        super().__init__()
        
        # Encoder CNN
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_clusters),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        clusters = self.cluster_head(features)
        return clusters
```

---

## 1.4 Fonction de Perte IIC ⭐ (LE CŒUR DE L'ALGORITHME)

### Concept: Information Mutuelle

> **But:** Maximiser l'information mutuelle I(z, z') entre:
> - z = cluster assignment de l'image originale
> - z' = cluster assignment de l'image transformée

### Formule mathématique

$$I(z, z') = \sum_{c} \sum_{c'} P(z=c, z'=c') \log \frac{P(z=c, z'=c')}{P(z=c) \cdot P(z'=c')}$$

### Étapes de calcul

1. **Calculer la matrice de probabilité jointe P(z, z')**
   ```python
   # z et z_prime sont de shape [batch_size, num_clusters]
   # P_joint = z.T @ z_prime / batch_size
   P = torch.mm(z.t(), z_prime) / batch_size  # [num_clusters, num_clusters]
   ```

2. **Symétriser la matrice** (pour stabilité)
   ```python
   P = (P + P.t()) / 2
   ```

3. **Calculer les marginales**
   ```python
   P_i = P.sum(dim=1).view(-1, 1)  # P(z)
   P_j = P.sum(dim=0).view(1, -1)  # P(z')
   ```

4. **Calculer l'information mutuelle**
   ```python
   # I(z, z') = sum(P * log(P / (P_i * P_j)))
   # Avec stabilité numérique (epsilon)
   eps = 1e-10
   MI = P * (torch.log(P + eps) - torch.log(P_i + eps) - torch.log(P_j + eps))
   loss = -MI.sum()  # Négatif car on veut MAXIMISER
   ```

### Implémentation complète
```python
def iic_loss(z, z_prime, eps=1e-10):
    """
    Calcule la perte IIC (negative mutual information).
    
    Args:
        z: [batch_size, num_clusters] - probabilités cluster image originale
        z_prime: [batch_size, num_clusters] - probabilités cluster image transformée
        eps: petit nombre pour stabilité numérique
    
    Returns:
        loss: scalar - negative mutual information (à minimiser)
    """
    batch_size = z.size(0)
    num_clusters = z.size(1)
    
    # Matrice de probabilité jointe [num_clusters, num_clusters]
    P = torch.mm(z.t(), z_prime) / batch_size
    
    # Symétrisation
    P = (P + P.t()) / 2
    
    # Clamp pour éviter les zéros
    P = torch.clamp(P, min=eps)
    
    # Marginales
    P_i = P.sum(dim=1, keepdim=True)  # [num_clusters, 1]
    P_j = P.sum(dim=0, keepdim=True)  # [1, num_clusters]
    
    # Information mutuelle
    MI = P * (torch.log(P) - torch.log(P_i) - torch.log(P_j))
    
    # Retourne -MI (on veut maximiser MI, donc minimiser -MI)
    return -MI.sum()
```

---

## 1.5 Boucle d'Entraînement

### Train epoch
```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for img, img_aug, _ in tqdm(dataloader, desc="Training"):
        img = img.to(device)
        img_aug = img_aug.to(device)
        
        # Forward pass
        z = model(img)
        z_prime = model(img_aug)
        
        # Compute loss
        loss = iic_loss(z, z_prime)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Boucle complète
```python
def train(model, train_loader, test_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': [], 'nmi': []}
    
    for epoch in range(epochs):
        # Train
        loss = train_epoch(model, train_loader, optimizer, device)
        history['loss'].append(loss)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            acc, nmi = evaluate(model, test_loader, device)
            history['accuracy'].append(acc)
            history['nmi'].append(nmi)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f} - NMI: {nmi:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    return history
```

---

## 1.6 Évaluation

### Problème: Les clusters n'ont pas de labels!

> Le modèle assigne des numéros de clusters (0-9), mais ils ne correspondent
> pas forcément aux vrais labels (0-9). Il faut faire un **matching**.

### Solution: Hungarian Algorithm

```python
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred, num_clusters):
    """
    Calcule l'accuracy après matching optimal cluster -> label.
    """
    # Matrice de confusion
    confusion = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    # Hungarian algorithm (maximiser = minimiser le négatif)
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # Accuracy
    correct = confusion[row_ind, col_ind].sum()
    accuracy = correct / len(y_true)
    
    return accuracy, dict(zip(col_ind, row_ind))  # mapping cluster -> label
```

### Fonction d'évaluation complète
```python
def evaluate(model, dataloader, device, num_clusters=10):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img, _, label in dataloader:
            img = img.to(device)
            z = model(img)
            pred = z.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(pred)
            all_labels.extend(label.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy avec Hungarian matching
    acc, mapping = cluster_accuracy(all_labels, all_preds, num_clusters)
    
    # NMI (Normalized Mutual Information)
    nmi = normalized_mutual_info_score(all_labels, all_preds)
    
    return acc, nmi
```

---

## 1.7 Visualisations à faire

- [ ] **Training curves**: Loss, Accuracy, NMI au fil des epochs
- [ ] **Exemples par cluster**: Montrer des images de chaque cluster
- [ ] **Matrice de confusion**: Clusters vs Labels réels
- [ ] **t-SNE/UMAP**: Visualiser les features dans un espace 2D
- [ ] **Exemples de paires**: Image originale vs transformée

---

## 1.8 Résultats attendus

| Métrique | MNIST (paper) | Notre objectif |
|----------|---------------|----------------|
| Accuracy | ~99%          | >90%           |
| NMI      | ~0.98         | >0.85          |

---

# 🎯 PARTIE 2: STREAMLIT

---

## 2.1 Structure de l'App

```
streamlit_app.py
│
├── Page 1: 📊 EDA (Exploratory Data Analysis)
│   ├── Afficher quelques images du dataset
│   ├── Distribution des classes
│   └── Exemples de paires (original + transformé)
│
├── Page 2: 🔬 Modèle
│   ├── Architecture du réseau (schéma)
│   ├── Explication de la loss IIC
│   └── Hyperparamètres utilisés
│
├── Page 3: 📈 Résultats
│   ├── Courbes de training (loss, accuracy)
│   ├── Matrice de confusion
│   ├── Comparaison avec le paper
│   └── Visualisation t-SNE
│
├── Page 4: 🎯 Prédictions Live
│   ├── Upload d'image OU dessin à la main
│   ├── Prédiction du cluster
│   └── Afficher les images similaires du même cluster
│
└── Page 5: 📝 Conclusions
    ├── Ce qui a bien marché
    ├── Difficultés rencontrées
    └── Améliorations possibles
```

---

## 2.2 Fonctionnalités à implémenter

### Page EDA
- [ ] `st.image()` pour afficher des exemples
- [ ] `st.bar_chart()` pour la distribution
- [ ] Slider pour choisir combien d'images afficher

### Page Modèle
- [ ] Afficher l'architecture avec un schéma ou du texte
- [ ] Expliquer la formule de la loss avec LaTeX: `st.latex()`
- [ ] Afficher les hyperparamètres dans un tableau

### Page Résultats
- [ ] `st.line_chart()` pour les courbes de training
- [ ] `st.pyplot()` pour la matrice de confusion
- [ ] Métriques dans des `st.metric()`

### Page Prédictions Live
- [ ] `st.file_uploader()` pour upload image
- [ ] OU `st_drawable_canvas` pour dessiner (package: streamlit-drawable-canvas)
- [ ] Charger le modèle sauvegardé
- [ ] Faire la prédiction et afficher le résultat

### Page Conclusions
- [ ] Texte avec `st.markdown()`
- [ ] Bullet points des difficultés/succès

---

## 2.3 Fichiers nécessaires pour Streamlit

- [ ] `models/iic_model.pth` - Modèle entraîné sauvegardé
- [ ] `data/training_history.json` - Historique du training (loss, acc, nmi)
- [ ] `data/confusion_matrix.npy` - Matrice de confusion
- [ ] `data/cluster_examples/` - Quelques images par cluster

---

## 2.4 Commande pour lancer

```bash
cd app/
streamlit run streamlit_app.py
```

---

# ✅ CHECKLIST FINALE

## Notebook
- [ ] Le notebook tourne de A à Z sans erreur
- [ ] Le code est commenté et expliqué
- [ ] Les résultats sont affichés clairement
- [ ] Le modèle est sauvegardé à la fin

## Streamlit
- [ ] L'app se lance sans erreur
- [ ] Toutes les pages fonctionnent
- [ ] La prédiction live marche
- [ ] L'interface est propre et intuitive

## Fichiers à rendre
- [ ] `notebooks/IIC_main.ipynb` - Notebook complet
- [ ] `app/streamlit_app.py` - Application Streamlit
- [ ] `models/iic_model.pth` - Modèle entraîné
- [ ] `requirements.txt` - Dépendances

---

# 📚 RESSOURCES

- **Paper IIC**: https://arxiv.org/abs/1807.06653
- **Code officiel**: https://github.com/xu-ji/IIC
- **PyTorch Docs**: https://pytorch.org/docs/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Hungarian Algorithm**: `scipy.optimize.linear_sum_assignment`

