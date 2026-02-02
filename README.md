# 🧠 Invariant Information Clustering (IIC)

## Projet ML - Réplication du paper

**Paper:** [Invariant Information Clustering for Unsupervised Image Classification and Segmentation](https://arxiv.org/abs/1807.06653)

**Auteurs:** Xu Ji, João F. Henriques, Andrea Vedaldi

---

## 📁 Structure du Projet

```
invariant-information-clustering/
├── README.md                    # Ce fichier
├── TODO.md                      # Liste des tâches à faire
├── requirements.txt             # Dépendances Python
│
├── notebooks/
│   └── IIC_main.ipynb          # Notebook principal
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Architecture du réseau
│   ├── loss.py                 # Fonction de perte IIC
│   ├── dataset.py              # Dataset et transformations
│   └── utils.py                # Fonctions utilitaires
│
├── app/
│   └── streamlit_app.py        # Application Streamlit pour la démo
│
├── data/                       # Données (gitignore)
│   └── ...
│
├── models/                     # Modèles sauvegardés (gitignore)
│   └── ...
│
└── docs/                       # Documents du projet
    ├── article machine learning.pdf
    ├── project_iic.pdf
    └── guidelines.pdf
```

---

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/[username]/invariant-information-clustering.git
cd invariant-information-clustering

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📓 Utilisation

### Notebook
```bash
jupyter notebook notebooks/IIC_main.ipynb
```

### Démo Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Résultats

| Dataset | Accuracy | NMI | Paper Accuracy |
|---------|----------|-----|----------------|
| MNIST   | -        | -   | 99.3%          |
| CIFAR-10| -        | -   | -              |

---

## 📚 Références

- [Paper original (arXiv)](https://arxiv.org/abs/1807.06653)
- [Code officiel (GitHub)](https://github.com/xu-ji/IIC)
