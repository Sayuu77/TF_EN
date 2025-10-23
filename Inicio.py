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
    layout="wide"
)

# Tema minimalista con colores frescos
st.markdown("""
<style>
    .stApp {
        background: #0a0a0a;
        color: #e0e0e0;
    }
    .main-title {
        font-size: 2.5rem;
        text-align: center;
        background: linear-gradient(45deg, #00d4ff, #0099ff, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.3rem;
        margin: 2rem 0 1rem 0;
        color: #00d4ff;
        border-left: 3px solid #00d4ff;
        padding-left: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00d4ff;
        margin: 0.5rem 0;
    }
    .stButton button {
        background: linear-gradient(45deg, #0066ff, #0099ff);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
    }
    .result-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    .similarity-high {
        color: #00ff88;
        font-weight: 600;
    }
    .similarity-medium {
        color: #ffaa00;
        font-weight: 600;
    }
    .similarity-low {
        color: #ff4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🔍 TF-IDF Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #a0a0a0; margin-bottom: 2rem;">
Cada línea se trata como un <strong>documento</strong>.  
Los documentos y preguntas deben estar en <strong>inglés</strong> para un análisis óptimo.
</div>
""", unsafe_allow_html=True)

# Ejemplo inicial en inglés
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "📄 Documentos (uno por línea):",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
        height=150
    )

with col2:
    question = st.text_input("❓ Pregunta:", "Who is playing?")

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🚀 Calcular Similitud"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        with st.spinner("Analizando documentos..."):
            # Vectorizador con stemming
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)

            # Mostrar matriz TF-IDF
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(documents))]
            )

            st.markdown('<div class="section-title">📊 Matriz TF-IDF</div>', unsafe_allow_html=True)
            st.dataframe(df_tfidf.round(3), use_container_width=True)

            # Vector de la pregunta
            question_vec = vectorizer.transform([question])

            # Similitud coseno
            similarities = cosine_similarity(question_vec, X).flatten()

            # Documento más parecido
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            # Resultado principal
            st.markdown('<div class="section-title">🎯 Resultado Principal</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("**Pregunta:**")
                st.info(question)
            
            with col2:
                st.markdown("**Documento más relevante:**")
                st.success(f"Doc {best_idx + 1}")
            
            with col3:
                st.markdown("**Similitud:**")
                if best_score > 0.5:
                    st.markdown(f'<div class="similarity-high">{best_score:.3f}</div>', unsafe_allow_html=True)
                elif best_score > 0.2:
                    st.markdown(f'<div class="similarity-medium">{best_score:.3f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="similarity-low">{best_score:.3f}</div>', unsafe_allow_html=True)
            
            st.markdown("**Texto del documento:**")
            st.markdown(f'<div class="result-card">{best_doc}</div>', unsafe_allow_html=True)

            # Todas las similitudes
            st.markdown('<div class="section-title">📈 Todas las Similitudes</div>', unsafe_allow_html=True)
            
            sim_df = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities
            })
            
            # Aplicar formato de color a las similitudes
            def color_similarity(val):
                if val > 0.5:
                    color = '#00ff88'
                elif val > 0.2:
                    color = '#ffaa00'
                else:
                    color = '#ff4444'
                return f'color: {color}; font-weight: 600'
            
            styled_df = sim_df.sort_values("Similitud", ascending=False).style.format({
                'Similitud': '{:.3f}'
            }).applymap(lambda x: color_similarity(x) if isinstance(x, (int, float)) else '', 
                       subset=['Similitud'])
            
            st.dataframe(styled_df, use_container_width=True)

            # Stems coincidentes
            st.markdown('<div class="section-title">🔤 Stems Coincidentes</div>', unsafe_allow_html=True)
            
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
            
            if matched:
                st.write("Stems de la pregunta encontrados en el documento:")
                for stem in matched:
                    st.code(stem, language='text')
            else:
                st.info("No se encontraron stems coincidentes.")

# Información adicional minimalista
with st.expander("ℹ️ Acerca del análisis"):
    st.markdown("""
    **TF-IDF** (Term Frequency-Inverse Document Frequency) mide la importancia de las palabras en los documentos.
    
    **Proceso:**
    1. **Tokenización y stemming** - Normalización de palabras
    2. **TF-IDF** - Cálculo de relevancia
    3. **Similitud coseno** - Comparación con la pregunta
    
    **Interpretación de similitudes:**
    - 🔴 < 0.2: Baja similitud
    - 🟠 0.2 - 0.5: Similitud media  
    - 🟢 > 0.5: Alta similitud
    """)
