# TODO: Application Streamlit pour la démo
# Lancer avec: streamlit run streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="IIC - Invariant Information Clustering",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Invariant Information Clustering (IIC)")
st.subheader("Unsupervised Image Classification Demo")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à:", [
    "📊 EDA",
    "🔬 Modèle & Résultats",
    "🎯 Prédictions Live",
    "📝 Conclusions"
])

if page == "📊 EDA":
    st.header("Exploratory Data Analysis")
    st.write("TODO: Ajouter les visualisations du dataset")

elif page == "🔬 Modèle & Résultats":
    st.header("Modèle & Résultats")
    st.write("TODO: Afficher l'architecture et les résultats")

elif page == "🎯 Prédictions Live":
    st.header("Prédictions Live")
    st.write("TODO: Permettre l'upload d'images et prédire le cluster")
    
    uploaded_file = st.file_uploader("Choisir une image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image uploadée", width=200)
        st.write("TODO: Ajouter la prédiction ici")

elif page == "📝 Conclusions":
    st.header("Conclusions & Limitations")
    st.write("TODO: Résumer les conclusions du projet")
