# 📋 TODO - Projet IIC (Invariant Information Clustering)

> **Paper:** Invariant Information Clustering for Unsupervised Image Classification and Segmentation  
> **Deadline présentation:** À confirmer  
> **Date de création:** 2 février 2026

---

## 📌 Phase 1: Compréhension du Paper (Semaine 1)

- [ ] Lire le paper en entier une première fois
- [ ] Identifier les concepts clés:
  - [ ] Mutual Information (Information Mutuelle)
  - [ ] Clustering non supervisé
  - [ ] Invariance aux transformations
  - [ ] Architecture des réseaux utilisés
- [ ] Prendre des notes sur la méthodologie
- [ ] Identifier les datasets utilisés (MNIST, STL-10, CIFAR, etc.)
- [ ] Comprendre la fonction de perte IIC

---

## 📌 Phase 2: Préparation de la Présentation

### 2.1 Context
- [ ] Quel est le contexte du paper ?
- [ ] Quel est l'objectif de l'article ?
- [ ] Pourquoi le clustering non supervisé est important ?

### 2.2 Research Gap
- [ ] Qu'est-ce qui manque dans la littérature ?
- [ ] Pourquoi les auteurs posent cette question ?
- [ ] Quelles sont les limitations des méthodes existantes ?

### 2.3 Methodology
- [ ] Data acquisition: quels datasets ?
- [ ] Data pipeline: preprocessing, augmentations
- [ ] Feature engineering: quelles features ?
- [ ] Models: architecture du réseau
- [ ] Metrics: accuracy, NMI (Normalized Mutual Information)
- [ ] Evaluation setup: train/test split, hyperparamètres

### 2.4 Replication
- [ ] Documenter ce qui était facile vs difficile
- [ ] Noter les surprises/éléments intéressants
- [ ] Comparer nos résultats avec ceux du paper

---

## 📌 Phase 3: Implémentation (Notebook)

### 3.1 Setup & Imports
- [ ] Importer les librairies (PyTorch, NumPy, etc.)
- [ ] Configurer le device (GPU/CPU)
- [ ] Fixer les seeds pour reproductibilité

### 3.2 Data Loading
- [ ] Charger le dataset (commencer par MNIST - plus simple)
- [ ] Implémenter les transformations/augmentations
- [ ] Créer les DataLoaders

### 3.3 Modèle
- [ ] Implémenter l'architecture du réseau (encoder)
- [ ] Implémenter la tête de clustering
- [ ] Implémenter la fonction de perte IIC

### 3.4 Training
- [ ] Boucle d'entraînement
- [ ] Logging des métriques
- [ ] Sauvegarde des checkpoints

### 3.5 Évaluation
- [ ] Calculer l'accuracy de clustering
- [ ] Visualiser les clusters
- [ ] Comparer avec les résultats du paper

---

## 📌 Phase 4: Démo Streamlit

- [ ] Créer l'application Streamlit de base
- [ ] Page 1: EDA (Exploratory Data Analysis)
  - [ ] Visualisation du dataset
  - [ ] Distribution des classes
- [ ] Page 2: Modèle & Résultats
  - [ ] Afficher les caractéristiques du modèle
  - [ ] Montrer les résultats obtenus
- [ ] Page 3: Prédictions Live
  - [ ] Upload d'image
  - [ ] Prédiction du cluster
  - [ ] Visualisation du résultat
- [ ] Page 4: Limitations & Conclusions

---

## 📌 Phase 5: Finalisation

- [ ] Nettoyer le code du notebook
- [ ] Ajouter des commentaires explicatifs
- [ ] Vérifier que le notebook est exécutable de A à Z
- [ ] Préparer les slides de présentation
- [ ] Répéter la présentation
- [ ] Tester la démo Streamlit

---

## 📚 Ressources Utiles

- **Paper original:** [IIC Paper](https://arxiv.org/abs/1807.06653)
- **Code officiel:** [GitHub xu-ji/IIC](https://github.com/xu-ji/IIC)
- **PyTorch:** https://pytorch.org/docs/
- **Streamlit:** https://docs.streamlit.io/

---

## 📝 Notes

_Ajoute ici tes notes au fur et à mesure du projet..._

