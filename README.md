# 🧠 Invariant Information Clustering (IIC)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deeplearning-project-alaa.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 Réplication du Paper

Ce projet est une **réplication simplifiée** de l'article :

> **"Invariant Information Clustering for Unsupervised Image Classification and Segmentation"**
> 
> Xu Ji, João F. Henriques, Andrea Vedaldi — ICCV 2019
> 
> 📎 [arXiv:1807.06653](https://arxiv.org/abs/1807.06653)

---

## 🎯 Qu'est-ce que IIC ?

**IIC** (Invariant Information Clustering) est une méthode de **deep learning non supervisé** qui permet de classifier des images **sans aucun label** en maximisant l'**information mutuelle** entre deux vues augmentées de la même image.

### 💡 Idée Clé

```
Image originale
      │
 ┌────┴────┐
 ▼         ▼
Aug 1    Aug 2      (Augmentations différentes)
 │         │
 ▼         ▼
CNN ◄────► CNN      (Poids partagés)
 │         │
 ▼         ▼
 z        z'        (Probabilités sur K clusters)
 └────┬────┘
      ▼
 I(z, z') → MAXIMISER
```

**Intuition :** Si deux images sont des versions augmentées de la même image originale, elles doivent appartenir au **même cluster**.

---

## 🚀 Démo en Ligne

### ▶️ [Lancer l'Application Streamlit](https://deeplearning-project-alaa.streamlit.app/)

L'application permet de :
- 📤 **Uploader une image** et voir la prédiction du modèle
- 📊 **Visualiser** les données CIFAR-10
- 📈 **Analyser** les résultats et métriques
- 🔬 **Comprendre** la méthode IIC

---

## 📊 Résultats

| Métrique | Notre Réplication | Paper Original |
|----------|-------------------|----------------|
| **Dataset** | CIFAR-10 | CIFAR-10 |
| **Architecture** | CNN 4 blocs | ResNet |
| **Accuracy** | **31.9%** | ~61% |
| **NMI** | **0.129** | ~0.51 |
| **Epochs** | 30 | 2000+ |
| **Pré-entraînement** | Aucun | ImageNet |

> ⚠️ Notre accuracy est inférieure car nous utilisons un CNN simple entraîné from scratch avec moins d'epochs. Le paper utilise ResNet pré-entraîné sur ImageNet.

---

## 📁 Structure du Projet

```
invariant-information-clustering/
├── 📄 README.md                 # Ce fichier
├── 📋 requirements.txt          # Dépendances Python
│
├── 📓 notebooks/
│   └── IIC_main.ipynb           # Notebook principal d'entraînement
│
├── 🐍 src/
│   ├── __init__.py
│   ├── model.py                 # Architecture du réseau
│   ├── loss.py                  # Fonction de perte IIC
│   ├── dataset.py               # Dataset et transformations
│   └── utils.py                 # Fonctions utilitaires
│
├── 🌐 app/
│   ├── streamlit_app.py         # Application Streamlit
│   └── requirements.txt         # Dépendances Streamlit
│
├── 🗂️ models/
│   └── iic_cifar10_model.pth    # Modèle entraîné (~38 MB)
│
├── 📊 data/                      # Données CIFAR-10 (auto-téléchargé)
│
└── 📚 docs/                      # Documents du projet
```

---

## 🛠️ Installation Locale

### Prérequis
- Python 3.10+
- pip

### Installation

```bash
# 1. Cloner le repo
git clone https://github.com/alaabelhadj/invariant-information-clustering.git
cd invariant-information-clustering

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 📓 Utilisation

### Option 1 : Notebook (Entraînement)
```bash
jupyter notebook notebooks/IIC_main.ipynb
```

### Option 2 : Application Streamlit (Démo)
```bash
cd app
streamlit run streamlit_app.py
```

### Option 3 : Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alaabelhadj/invariant-information-clustering/blob/main/notebooks/IIC_main.ipynb)

---

## 🧮 Formule Mathématique

### Information Mutuelle

$$I(z, z') = \sum_{c=1}^{K} \sum_{c'=1}^{K} P(z=c, z'=c') \cdot \log \frac{P(z=c, z'=c')}{P(z=c) \cdot P(z'=c')}$$

### Loss IIC

$$\mathcal{L}_{IIC} = -I(z, z')$$

**Pourquoi ça marche ?**
- Maximiser l'information mutuelle force les clusters à être **consistants** entre les deux vues
- La normalisation par les marginales **pénalise** les solutions triviales (tout dans 1 cluster)

---

## 🔧 Configuration du Modèle

| Hyperparamètre | Valeur |
|----------------|--------|
| Dataset | CIFAR-10 |
| Clusters K | 10 |
| Batch Size | 256 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Epochs | 30 |
| Architecture | CNN (64→128→256→256) |

---

## 📈 Métriques

### Accuracy (avec mapping)
L'accuracy est calculée après avoir trouvé le **mapping optimal** entre clusters et vraies classes (car IIC est non supervisé).

### NMI (Normalized Mutual Information)
$$NMI(Y, C) = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}$$

- Mesure la correspondance entre clusters et classes
- Valeur entre 0 (aléatoire) et 1 (parfait)
- **Avantage :** Invariant au mapping et au nombre de clusters

---

## 🎓 Leçons Apprises

1. **L'over-clustering aide** : K > nb de classes peut améliorer les résultats
2. **Les augmentations sont cruciales** : Plus elles sont agressives, mieux c'est
3. **Pas de convergence garantie** : L'entraînement non supervisé est instable
4. **Le mapping est essentiel** : Sans mapping optimal, l'accuracy est sous-estimée

---

## 📚 Références

- 📄 **Paper original :** [arXiv:1807.06653](https://arxiv.org/abs/1807.06653)
- 💻 **Code officiel :** [github.com/xu-ji/IIC](https://github.com/xu-ji/IIC)
- 🎥 **Vidéo explicative :** [YouTube - IIC Explained](https://www.youtube.com/results?search_query=invariant+information+clustering)

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours de **Machine Learning / Deep Learning**.

---

## 📜 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
