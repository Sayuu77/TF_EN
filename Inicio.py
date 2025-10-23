import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(
    page_title="TF-IDF Analysis",
    page_icon="🔍",
    layout="centered"
)

# Tema minimalista con morado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    .main-title {
        font-size: 2.2rem;
        text-align: center;
        background: linear-gradient(45deg, #a855f7, #ec4899, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton button {
        background: linear-gradient(45deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #a855f7;
    }
    .similarity-high { color: #10b981; font-weight: 700; }
    .similarity-medium { color: #f59e0b; font-weight: 700; }
    .similarity-low { color: #ef4444; font-weight: 700; }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🔍 TF-IDF Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis de similitud semántica usando TF-IDF y Cosine Similarity</div>', unsafe_allow_html=True)

# Sección de entrada
with st.container():
    
    st.markdown("**📄 Documentos** (uno por línea)")
    text_input = st.text_area(
        "",
        "My mom shouts.\nThat cat is so chunky.\nI wanna play videogames.",
        height=120,
        label_visibility="collapsed"
    )
    
    st.markdown("**❓ Pregunta**")
    question = st.text_input("", "Who is chunky?", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Botón de cálculo centrado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Calcular Similitud", use_container_width=True):
        documents = [d.strip() for d in text_input.split("\n") if d.strip()]
        
        if len(documents) < 1:
            st.warning("⚠️ Ingresa al menos un documento.")
        else:
            with st.spinner("Analizando similitudes..."):
                # Vectorizador con stemming
                vectorizer = TfidfVectorizer(
                    tokenizer=tokenize_and_stem,
                    stop_words="english",
                    token_pattern=None
                )

                # Ajustar con documentos
                X = vectorizer.fit_transform(documents)

                # Vector de la pregunta
                question_vec = vectorizer.transform([question])

                # Similitud coseno
                similarities = cosine_similarity(question_vec, X).flatten()

                # Documento más parecido
                best_idx = similarities.argmax()
                best_doc = documents[best_idx]
                best_score = similarities[best_idx]

                # RESULTADO PRINCIPAL
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                # Header del resultado
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("### 🎯 Mejor Coincidencia")
                    st.markdown(f"**Documento {best_idx + 1}**")
                with col2:
                    if best_score > 0.5:
                        sim_class = "similarity-high"
                    elif best_score > 0.2:
                        sim_class = "similarity-medium"
                    else:
                        sim_class = "similarity-low"
                    st.markdown(f'<div class="{sim_class}" style="font-size: 1.5rem; text-align: center;">{best_score:.3f}</div>', unsafe_allow_html=True)
                
                # Documento y pregunta
                st.markdown("**Pregunta:**")
                st.info(f"\"{question}\"")
                
                st.markdown("**Documento encontrado:**")
                st.success(f"\"{best_doc}\"")
                
                st.markdown('</div>', unsafe_allow_html=True)

                # MATRIZ TF-IDF
                with st.expander("Matriz TF-IDF", expanded=False):
                    df_tfidf = pd.DataFrame(
                        X.toarray(),
                        columns=vectorizer.get_feature_names_out(),
                        index=[f"Doc {i+1}" for i in range(len(documents))]
                    )
                    st.dataframe(df_tfidf.round(3), use_container_width=True)

                # TODAS LAS SIMILITUDES
                with st.expander("Todas las Similitudes", expanded=True):
                    sim_df = pd.DataFrame({
                        "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                        "Similitud": similarities,
                        "Texto": documents
                    })
                    
                    # Ordenar y mostrar
                    sim_df_sorted = sim_df.sort_values("Similitud", ascending=False)
                    
                    for _, row in sim_df_sorted.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col1:
                                st.markdown(f"**{row['Documento']}**")
                            with col2:
                                st.caption(row['Texto'][:60] + "..." if len(row['Texto']) > 60 else row['Texto'])
                            with col3:
                                score = row['Similitud']
                                if score > 0.5:
                                    sim_class = "similarity-high"
                                elif score > 0.2:
                                    sim_class = "similarity-medium"
                                else:
                                    sim_class = "similarity-low"
                                st.markdown(f'<div class="{sim_class}" style="text-align: right;">{score:.3f}</div>', unsafe_allow_html=True)
                            st.divider()

                # STEMS COINCIDENTES
                with st.expander("Stems Coincidentes", expanded=False):
                    vocab = vectorizer.get_feature_names_out()
                    q_stems = tokenize_and_stem(question)
                    matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
                    
                    if matched:
                        st.markdown("**Stems encontrados:**")
                        cols = st.columns(4)
                        for i, stem in enumerate(matched):
                            with cols[i % 4]:
                                st.markdown(f'<div style="background: rgba(168, 85, 247, 0.2); padding: 0.5rem; border-radius: 6px; text-align: center; margin: 0.2rem;">{stem}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No se encontraron stems coincidentes.")

# Información en acordeón
with st.expander("ℹ️ Acerca del análisis", expanded=False):
    st.markdown("""
    **🔍 TF-IDF (Term Frequency-Inverse Document Frequency)**
    
    Mide la importancia de palabras en documentos considerando:
    - **Frecuencia en el documento** (TF)
    - **Frecuencia inversa en el corpus** (IDF)
    
    **🎯 Similitud Coseno**
    - Compara vectores TF-IDF
    - Rango: 0 (sin similitud) a 1 (idéntico)
    
    **Interpretación:**
    - 🟢 > 0.5: Alta similitud
    - 🟠 0.2 - 0.5: Similitud media  
    - 🔴 < 0.2: Baja similitud
    """)
