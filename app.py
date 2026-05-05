import streamlit as st
import os, json, json5, re, io
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image

# ==========================================
# 1. SETUP & SECRETS CONFIGURATION
# ==========================================
st.set_page_config(page_title="Multimodal Forensic Model", layout="wide")
st.title("Multimodal Forensic Analysis Model")

# Pull API Key from .streamlit/secrets.toml
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    st.error("Missing API Key! Ensure GEMINI_API_KEY is in .streamlit/secrets.toml")
    st.stop()

MODEL_NAME = "models/gemini-2.5-pro"

# ==========================================
# 2. STAGE 1: PROFESSIONAL NETWORK PLOTTING
# ==========================================
def process_stage_1(chat_data):
    """
    Exact replica of the graphing logic from Stage 1.ipynb.
    Uses the notebook's specific edge weight calculation and design.
    """
    G = nx.DiGraph()
    
    # Use the specific weight calculation from Stage 1 notebook
    for msg in chat_data:
        u, v = msg.get("sender"), msg.get("receiver")
        weight = msg.get("weight", 1.0) # Ensure it pulls the notebook's weight value
        if u and v:
            if G.has_edge(u, v):
                G[u][v]['weight'] += weight
            else:
                G.add_edge(u, v, weight=weight)

    # Table Scores (HITS Algorithm)[cite: 2]
    hubs, authorities = nx.hits(G, max_iter=100)
    
    # Forensic Role Assignment
    top_hub = max(hubs, key=hubs.get)
    sorted_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
    middlemen = [n for n, s in sorted_auth[:2] if n != top_hub]

    node_colors = []
    for node in G.nodes():
        if node == top_hub: node_colors.append("red")
        elif node in middlemen: node_colors.append("orange")
        else: node_colors.append("skyblue")

    # Exact Notebook Drawing Parameters
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.0, iterations=50) # Matching notebook layout[cite: 3]
    
    # Nodes with black outlines
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Edges with thickness matching image 1[cite: 3]
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'] * 2, 
                               edge_color='grey', alpha=0.7, arrowsize=25, ax=ax)
    
    # Edge Weight Labels - Exact same formatting as image 1[cite: 3]
    edge_labels = { (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True) }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', 
                                 font_size=9, label_pos=0.5, font_weight='bold')

    # Legend exactly as shown in notebook[cite: 3]
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Mastermind (Top Hub)', markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Middleman (High Auth)', markerfacecolor='orange', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Follower', markerfacecolor='skyblue', markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Roles")
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=False)
    buf.seek(0)
    plt.close(fig)
    
    # DataFrame for Stage 2[cite: 2]
    df = pd.DataFrame({
        "Actor": list(hubs.keys()),
        "Hub_Score": [round(v, 4) for v in hubs.values()],
        "Auth_Score": [round(v, 4) for v in authorities.values()]
    }).sort_values("Hub_Score", ascending=False)
    
    return buf, df
# ==========================================
# 3. STAGE 2: MULTIMODAL FORENSIC REPORT
# ==========================================
def process_stage_2(chat_data, graph_bytes, centrality_df, audio_files, image_files):
    """Combines all input types for LLM processing[cite: 2]."""
    
    participants = sorted(set(m["sender"] for m in chat_data if m.get("sender")) | 
                          set(m["receiver"] for m in chat_data if m.get("receiver")))
    
    prompt = f"""
    You are a professional Forensic Analyst AI. 
    Review the evidence involving participants: {', '.join(participants)}
    
    Required Analysis:
    1. Study the Network Graph and Centrality Scores.
    2. Review the Chat Transcripts and multimodal attachments.
    3. Detect criminal intent and classify the activity.

    RETURN ONLY VALID JSON:
    {{
      "summary": "Detailed narrative of criminal operation",
      "roles": {{ "name": "detailed forensic role" }},
      "intent_classification": "crime category",
      "confidence_score": 0.0
    }}
    """
    
    # Feed Stage 1 output image directly into Stage 2 prompt[cite: 2]
    parts = [
        prompt, 
        {"mime_type": "image/png", "data": graph_bytes.getvalue()}, 
        f"CENTRALITY DATA:\n{centrality_df.to_string(index=False)}"
    ]

    # Modality: Text
    chat_text = "\n".join([f"{m['sender']} -> {m['receiver']}: {m['message_text']}" 
                           for m in chat_data if m.get("message_text")])
    parts.append(f"CHAT LOGS:\n{chat_text}")

    # Modality: Audio/Image
    if audio_files:
        for audio in audio_files:
            parts.append({"mime_type": "audio/mpeg", "data": audio.read()})
    if image_files:
        for img in image_files:
            parts.append({"mime_type": "image/jpeg", "data": img.read()})

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(parts)
    
    clean_text = re.sub(r"^```json\n|\n```$", "", response.text.strip())
    return json5.loads(clean_text)

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    json_up = st.file_uploader("Upload Chat JSON", type="json")
with cols[1]:
    audio_up = st.file_uploader("Audio Evidence", type=["mp3", "wav"], accept_multiple_files=True)
with cols[2]:
    img_up = st.file_uploader("Image Evidence", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if json_up:
    chat_history = json.load(json_up)
    
    if st.button("🚀 Run Forensic Analysis"):
        # RUN STAGE 1
        with st.spinner("Plotting Network..."):
            graph_buf, stats_df = process_stage_1(chat_history)
            
        st.subheader("📍 Stage 1 Output: Network Analysis")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(graph_buf, caption="Criminal Network Graph (Passed to Stage 2)")
        with c2:
            st.write("**Node Importance**")
            st.dataframe(stats_df, hide_index=True)
            
        # RUN STAGE 2
        with st.spinner("Generating Forensic Report..."):
            report = process_stage_2(chat_history, graph_buf, stats_df, audio_up, img_up)
            
        st.divider()
        st.subheader("📋 Stage 2 Output: Forensic Report")
        
        rep_col1, rep_col2 = st.columns([2, 1])
        with rep_col1:
            st.success(f"**Intent:** {report.get('intent_classification')}")
            st.write(f"**Summary:** {report.get('summary')}")
        with rep_col2:
            st.metric("Confidence", f"{int(report.get('confidence_score', 0) * 100)}%")
            st.write("**Identified Roles:**")
            st.json(report.get("roles"))