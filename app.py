import streamlit as st
import os, json, json5, re, io
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

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
# 2. LOAD ML MODELS (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_scoring_models():
    """Loads the Hugging Face models used in the Stage 1 notebook."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return classifier, embedding_model

classifier, embedding_model = load_scoring_models()

# Define Knowledge Base exactly as in notebook
candidate_labels = ["criminal conspiracy", "issuing a command", "normal conversation"]
prototypes_by_category = {
    "Threats & Enforcement": ["Make him understand the consequences.", "Send a clear message.", "Neutralize the threat."],
    "Financial Transactions": ["Confirm the payment.", "Launder the money.", "Settle the accounts.", "Look at this cash."],
    "Commands & Directives": ["Get the job done now.", "Proceed with the plan.", "Green-light the operation."],
    "Secrecy & Evasion": ["Use the secure line.", "Delete this after reading.", "Use the burner phone."],
    "Acquiring Resources": ["Acquire the equipment.", "Source the vehicle.", "Here are the weapons."],
    "Logistics & Transport": ["Coordinate the pickup.", "The package is delivered.", "Move the merchandise."],
    "Reporting & Status Updates": ["Report your status.", "Target is under surveillance.", "Operation complete."]
}
category_weights = {
    "Commands & Directives": 1.8, "Financial Transactions": 1.6, "Threats & Enforcement": 1.5,
    "Secrecy & Evasion": 1.2, "Logistics & Transport": 0.8, "Acquiring Resources": 0.7, "Reporting & Status Updates": 0.5
}
# Pre-compute prototype embeddings
prototype_embeddings_by_category = {
    cat: embedding_model.encode(sents, convert_to_tensor=True) 
    for cat, sents in prototypes_by_category.items()
}

# ==========================================
# 3. STAGE 1: PROFESSIONAL NETWORK PLOTTING
# ==========================================
def process_stage_1(chat_data):
    """Exact replica of Stage 1 ML logic to calculate proper weights."""
    directed_edge_weights = {}
    
    # EXACT Scoring Logic from your Colab Notebook
    for msg in chat_data:
        sender = msg.get('sender')
        receiver = msg.get('receiver')
        text = msg.get('message_text')
        
        if not text or not sender or not receiver:
            continue
            
        try:
            # Zero-shot
            result = classifier(text, candidate_labels, multi_label=False)
            score_map = {label: score for label, score in zip(result['labels'], result['scores'])}
            c_score = (score_map.get("criminal conspiracy", 0) * 1.5) + (score_map.get("issuing a command", 0) * 1.0)
            
            # Semantic Similarity
            msg_emb = embedding_model.encode(text, convert_to_tensor=True)
            sem_score = 0
            for cat, proto_embs in prototype_embeddings_by_category.items():
                best_sim = torch.max(util.cos_sim(msg_emb, proto_embs)).item()
                curr_score = best_sim * category_weights[cat]
                if curr_score > sem_score: 
                    sem_score = curr_score
                    
            # Hybrid Score
            final_weight = ((0.3 * c_score) + (0.7 * sem_score)) * 2.0
            edge = (sender, receiver)
            directed_edge_weights[edge] = directed_edge_weights.get(edge, 0) + final_weight
        except Exception:
            continue

    # Build Graph & HITS
    G = nx.DiGraph()
    for (u, v), w in directed_edge_weights.items():
        if w > 1.2: 
            G.add_edge(u, v, weight=round(w, 2))
            
    if G.number_of_nodes() == 0:
        return None, None

    hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
    
    # Identify Roles
    avg_hub = sum(hubs.values()) / len(hubs) if hubs else 0
    avg_auth = sum(authorities.values()) / len(authorities) if authorities else 0
    
    potential_mms = [n for n in G.nodes() if hubs[n] > avg_hub * 1.2]
    mastermind = max(potential_mms, key=lambda x: hubs[x]) if potential_mms else None
    
    roles = {}
    for n in G.nodes():
        if n == mastermind: roles[n] = "Mastermind"
        elif authorities[n] > avg_auth * 1.5: roles[n] = "Middleman"
        else: roles[n] = "Follower"

    # EXACT Plotting Parameters from the Notebook
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80) 
    
    role_colors_map = {"Mastermind": "red", "Middleman": "orange", "Follower": "skyblue"}
    node_colors = [role_colors_map.get(roles.get(node, "Follower"), "skyblue") for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors='black', ax=ax)
    
    # Edges with thickness matching notebook
    edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[min(w, 5) for w in edge_weights_list], 
                           edge_color='gray', arrowsize=20, node_size=3000, ax=ax)
                           
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen', ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Mastermind (Top Hub)', markerfacecolor='red', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Middleman (High Authority)', markerfacecolor='orange', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Follower', markerfacecolor='skyblue', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Roles")
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=False)
    buf.seek(0)
    plt.close(fig)
    
    # DataFrame
    df = pd.DataFrame({
        "Actor": list(hubs.keys()),
        "Hub_Score": [round(v, 4) for v in hubs.values()],
        "Auth_Score": [round(v, 4) for v in authorities.values()]
    }).sort_values("Hub_Score", ascending=False)
    
    return buf, df

# ==========================================
# 4. STAGE 2: MULTIMODAL FORENSIC REPORT
# ==========================================
def process_stage_2(chat_data, graph_bytes, centrality_df, audio_files, image_files):
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
    
    parts = [
        prompt, 
        {"mime_type": "image/png", "data": graph_bytes.getvalue()}, 
        f"CENTRALITY DATA:\n{centrality_df.to_string(index=False)}"
    ]

    chat_text = "\n".join([f"{m['sender']} -> {m['receiver']}: {m['message_text']}" 
                           for m in chat_data if m.get("message_text")])
    parts.append(f"CHAT LOGS:\n{chat_text}")

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
# 5. USER INTERFACE
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
        with st.spinner("Analyzing Texts & Plotting Network (This may take a minute to run ML models)..."):
            graph_buf, stats_df = process_stage_1(chat_history)
            
        if graph_buf is None:
            st.warning("Graph empty (no strong evidence found based on weight threshold).")
        else:
            st.subheader("📍 Stage 1 Output: Network Analysis")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.image(graph_buf, caption="Criminal Network Graph (Passed to Stage 2)")
            with c2:
                st.write("**Node Importance**")
                st.dataframe(stats_df, hide_index=True)
                
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
