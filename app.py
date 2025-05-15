# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import umap.umap_ as umap
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔭 CubeSat Concept Clusters in Latin America")

# ===== BARRA DE BÚSQUEDA PARA KEYWORDS =====
st.sidebar.title("🔎 Buscar temas")
user_keywords = st.sidebar.text_input("Palabras clave para buscar en OpenAlex:", 
                                      value="CubeSat OR CubeSats OR Nanosatellite")

# ===============================
# 🔄 Obtener datos desde OpenAlex
# ===============================
@st.cache_data(show_spinner=True)
def fetch_data(keywords):
    latam_countries = ['HN', 'MX', 'GT', 'SV', 'NI', 'CR', 'PA', 'CU', 'CO', 
                       'VE', 'EC', 'PE', 'DO', 'PR', 'BO', 'PY', 'BR', 'AR', 'UY']
    base_url = 'https://api.openalex.org/works'
    per_page = 200
    max_pages = 5
    all_results = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?search={keywords}&per-page={per_page}&page={page}&filter=authorships.institutions.country_code:{'|'.join(latam_countries)}"
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"❌ Error en la página {page} al obtener datos desde OpenAlex.")
            break
        data = response.json()
        if 'results' not in data or not data['results']:
            break
        all_results.extend(data['results'])

    return all_results

# ===== OBTENER DATOS SEGÚN KEYWORDS =====
data = fetch_data(user_keywords)

# Si no hay datos, terminar ejecución
if not data:
    st.warning("No se encontraron resultados con los términos buscados.")
    st.stop()

# ===============================
# 🧠 Procesamiento de conceptos
# ===============================
concept_lists = [[c['display_name'] for c in work.get('concepts', [])] for work in data]
all_concepts = sum(concept_lists, [])
top_concepts_counts = Counter(all_concepts).most_common(100)
top_concepts = [c for c, _ in top_concepts_counts]

# ===============================
# 📊 Vectorización y reducción
# ===============================
vectorizer = CountVectorizer(vocabulary=top_concepts, binary=True)
X = vectorizer.fit_transform(['; '.join(concepts) for concepts in concept_lists])

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
embedding = reducer.fit_transform(X.T)

# ===============================
# 🔍 Clustering HDBSCAN
# ===============================
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
clusters = clusterer.fit_predict(embedding)

# ===============================
# 📋 Preparar DataFrame
# ===============================
min_len = min(len(top_concepts), len(clusters), len(top_concepts_counts))
df = pd.DataFrame({
    'Concept': top_concepts[:min_len],
    'X': embedding[:min_len, 0],
    'Y': embedding[:min_len, 1],
    'Cluster': clusters[:min_len],
    'Frequency': [count for _, count in top_concepts_counts[:min_len]]
})

df_filtered = df[df['Cluster'] >= 0]
if df_filtered.empty:
    st.error("❌ No se encontraron clusters válidos. Ajusta los parámetros.")
    st.stop()

# ===============================
# 🎛️ Sidebar: selección de cluster
# ===============================
cluster_ids = sorted(df_filtered['Cluster'].unique())
selected_cluster = st.sidebar.selectbox("Selecciona un clúster para explorar:", cluster_ids)
df_selected = df_filtered[df_filtered['Cluster'] == selected_cluster]

# ===============================
# 🌐 Red de conceptos
# ===============================
st.subheader("🌐 Cluster Network")

def build_graph(df_filtered, concept_lists):
    G = nx.Graph()
    concept_set = set(df_filtered['Concept'])

    for _, row in df_filtered.iterrows():
        G.add_node(row['Concept'], size=row['Frequency'], cluster=row['Cluster'], pos=(row['X'], row['Y']))

    for concepts in concept_lists:
        filtered = [c for c in concepts if c in concept_set]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                c1, c2 = filtered[i], filtered[j]
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
    return G

G = build_graph(df_filtered, concept_lists)

if not G.nodes:
    st.warning("⚠️ El grafo está vacío.")
else:
    pos = nx.get_node_attributes(G, 'pos')
    colors = px.colors.qualitative.Set2
    node_colors = [colors[G.nodes[n]['cluster'] % len(colors)] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] * 5 for n in G.nodes()]
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=list(zip(*pos.values()))[0], y=list(zip(*pos.values()))[1],
                            mode='markers+text', text=list(G.nodes),
                            marker=dict(color=node_colors, size=node_sizes, line_width=1))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='CubeSat Interactive Cluster Network',
        showlegend=False, margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 🍯 Honeycomb Plot
# ===============================
st.subheader("🍯 Honeycomb Cluster Plot")
fig2, ax = plt.subplots(figsize=(12, 8))
hb = ax.hexbin(df_filtered['X'], df_filtered['Y'], gridsize=25, cmap='YlGnBu', mincnt=1)

# Agregar la barra de color (colorbar)
cb = fig2.colorbar(hb, ax=ax)
cb.set_label('Density of Concepts')

for _, row in df_filtered.iterrows():
    ax.text(row['X'], row['Y'], row['Concept'], fontsize=8, ha='center', va='center')

ax.axis('off')
ax.set_title("Honeycomb (Hexbin) Plot of CubeSat Concept Clusters")

# Mostrarlo
st.pyplot(fig2)


# ===============================
# 🔍 Detalles del Clúster Seleccionado
# ===============================
st.subheader(f"🔍 Exploración del Clúster {selected_cluster}")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("**📊 Frecuencias de Conceptos:**")
    fig_bar = px.bar(df_selected.sort_values(by="Frequency", ascending=False),
                     x="Frequency", y="Concept", orientation="h",
                     color="Frequency", color_continuous_scale="Viridis")
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("**📋 Tabla de Conceptos:**")
    st.dataframe(df_selected.sort_values(by="Frequency", ascending=False), use_container_width=True)

# ===============================
# 🌞 Árbol de Conocimiento Real
# ===============================

st.subheader("🌞 Árbol de Conocimiento LACCEI – Panorama General")

# Opciones de filtrado
years_available = list(range(2015, 2025))
selected_year = st.selectbox("Selecciona el año:", years_available[::-1])

# Refiltrar data por año
filtered_data = []
for work in data:
    if 'publication_year' in work and work['publication_year'] == selected_year:
        filtered_data.append(work)

if not filtered_data:
    st.warning("⚠️ No hay publicaciones para el año seleccionado.")
    st.stop()

# Contar publicaciones por institución
institution_counts = Counter()
concept_counts = Counter()
institution_concepts = {}

for work in filtered_data:
    main_concept = work['concepts'][0]['display_name'] if work.get('concepts') else "Unknown"
    concept_counts[main_concept] += 1

    for auth in work.get('authorships', []):
        for inst in auth.get('institutions', []):
            name = inst.get('display_name', 'Unknown')
            institution_counts[name] += 1

            if name not in institution_concepts:
                institution_concepts[name] = Counter()
            institution_concepts[name][main_concept] += 1
# Ordenamiento
sort_option = st.selectbox("Ordenar por:", ["Número de publicaciones", "Alfabético"])
if sort_option == "Número de publicaciones":
    sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)
else:
    sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[0])

# Preparar DataFrame para tabla
# Diccionario de códigos de país a nombres completos
country_names = {
    'AR': 'Argentina', 'BO': 'Bolivia', 'BR': 'Brasil', 'CL': 'Chile', 'CO': 'Colombia',
    'CR': 'Costa Rica', 'CU': 'Cuba', 'DO': 'República Dominicana', 'EC': 'Ecuador',
    'GT': 'Guatemala', 'HN': 'Honduras', 'MX': 'México', 'NI': 'Nicaragua',
    'PA': 'Panamá', 'PE': 'Perú', 'PR': 'Puerto Rico', 'PY': 'Paraguay',
    'SV': 'El Salvador', 'UY': 'Uruguay', 'VE': 'Venezuela',
    'DE': 'Alemania', 'US': 'Estados Unidos', 'GB': 'Reino Unido', 'FR': 'Francia',
    'IT': 'Italia', 'ES': 'España', 'CN': 'China', 'JP': 'Japón', 'KR': 'Corea del Sur',
    'IN': 'India', 'CA': 'Canadá', 'AU': 'Australia', 'RU': 'Rusia', 'ZA': 'Sudáfrica'
}
institution_country_map = {
    'University of Tübingen': 'Alemania',
    'Université de Paris': 'Francia',
    'University of Oxford': 'Reino Unido',
    'Massachusetts Institute of Technology': 'Estados Unidos',
    # Agregar mas al mapa por la logica y por que no reconoce algunos
}

# Preparar DataFrame para tabla con país
institutions_table = []

for inst, pubs in sorted_institutions:
    top_concept = institution_concepts.get(inst, Counter()).most_common(1)[0][0] if institution_concepts.get(inst) else "Desconocido"
    country_code = None

    # Buscar país en la metadata de los trabajos
    for work in filtered_data:
        for auth in work.get('authorships', []):
            for i in auth.get('institutions', []):
                if i.get('display_name') == inst:
                    raw_code = i.get('country_code')
                    country_code = raw_code.upper() if raw_code else None
                    break
            if country_code:
                break
        if country_code:
            break

    # Determinar el nombre del país
    if country_code:
        country_name = country_names.get(country_code, "Desconocido")
    else:
        # Intentar inferir el país por el nombre de la institución
        country_name = institution_country_map.get(inst, "Desconocido")

    institutions_table.append({
        "Institución": inst,
        "País": country_name,
        "Publicaciones": pubs,
        "Conceptos Top": top_concept
    })


df_institutions = pd.DataFrame(institutions_table)



# Sunburst Data
sunburst_df = pd.DataFrame({
    "Parent": ["LACCEI"] * len(concept_counts),
    "Child": list(concept_counts.keys()),
    "Value": list(concept_counts.values())
})

fig_sunburst = px.sunburst(
    sunburst_df,
    names="Child",
    parents="Parent",
    values="Value",
    title="Árbol de Conocimiento LACCEI – Panorama General",
    color="Value",
    color_continuous_scale="Blues"
)

fig_sunburst.update_layout(margin=dict(t=40, l=0, r=0, b=0))
st.plotly_chart(fig_sunburst, use_container_width=True)

# Mostrar tabla
st.subheader("🏛️ Instituciones Publicando en el Año Seleccionado")
st.dataframe(df_institutions, use_container_width=True)

