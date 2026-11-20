import os
import io
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'

def main():
    print("--- SCRIPT 08: MEDIUM-CONFIDENCE STRING FUNCTIONAL ASSOCIATION NETWORK ---")
    feat_df = pd.read_csv(os.path.join(DATA_DIR, 'Supplementary_Data_5.csv')).head(50)
    top50 = feat_df.iloc[:, 0].astype(str).tolist()
    imp_map = dict(zip(feat_df.iloc[:, 0].astype(str), feat_df.iloc[:, 1].astype(float)))

    G = nx.Graph()
    for api_url in ["https://string-db.org/api/tsv/network", "https://version-11-5.string-db.org/api/tsv/network"]:
        try:
            resp = requests.post(api_url, data={"identifiers": "%0d".join(top50), "species": 9606, "required_score": 400}, timeout=25)
            if resp.status_code == 200 and 'preferredName_A' in resp.text:
                edges = pd.read_csv(io.StringIO(resp.text), sep='\t')
                for _, r in edges.iterrows():
                    a, b, sc_v = str(r['preferredName_A']), str(r['preferredName_B']), float(r['score'])
                    if sc_v >= 0.400 and a != b:
                        G.add_edge(a, b, weight=round(sc_v, 4), combined_score=round(sc_v, 4))
                break
        except Exception:
            pass

    mucin_adhesion = {'MUC1', 'MUC5B', 'TFF3', 'CLDN3', 'CLDN4', 'CD24', 'CEACAM5'}
    alveolar_secretory = {'SCGB3A2', 'SCGB3A1', 'SFTPB', 'SFTPC', 'SFTPD', 'SLC34A2', 'AQP5'}
    complement_innate = {'C3', 'HP', 'S100A9', 'SPP1'}
    palette = {
        'Mucin / Cell-Adhesion Module': '#B55D6A',
        'Alveolar / Secretoglobin Module': '#2A9D8F',
        'Complement / Innate Immune Module': '#D9A441',
        'Other Lineage Classifier Feature': '#5C6F84'
    }

    manual_pos = {
        'ASS1': (4.40, 9.55), 'AQP1': (5.35, 8.45), 'S100A9': (3.45, 8.95), 'HP': (2.05, 8.30), 'C3': (1.05, 7.15),
        'CXCL14': (0.30, 5.95), 'CCL3': (0.85, 4.75), 'ISG15': (-0.55, 5.35), 'IFI27': (-1.45, 4.45),
        'ITGA1': (0.95, 3.05), 'SPP1': (2.45, 3.05), 'CHI3L1': (2.85, 1.35), 'MT-CO3': (1.50, -0.20),
        'TMSB10': (-0.05, 0.25), 'TMSB4X': (0.55, -1.00), 'TMEM176B': (2.65, 5.85), 'TMEM176A': (4.05, 5.85),
        'AQP5': (5.75, 6.65), 'SCGB3A2': (7.35, 6.85), 'SCGB3A1': (8.75, 5.85), 'SFTPB': (5.95, 5.05),
        'SFTPD': (4.65, 3.85), 'MUC5B': (7.35, 4.70), 'SFTPC': (8.80, 3.85), 'SLC34A2': (6.65, 3.25),
        'MUC1': (5.90, 1.80), 'TFF3': (8.15, 2.25), 'CEACAM5': (8.35, 0.85), 'CLDN3': (7.05, 0.65),
        'CLDN4': (5.60, -0.45), 'CD24': (7.00, -0.65)
    }

    pos = {}
    for idx, n in enumerate(G.nodes()):
        mod = ('Mucin / Cell-Adhesion Module' if n in mucin_adhesion else
               'Alveolar / Secretoglobin Module' if n in alveolar_secretory else
               'Complement / Innate Immune Module' if n in complement_innate else
               'Other Lineage Classifier Feature')
        G.nodes[n]['display_label'] = n
        G.nodes[n]['degree'] = int(G.degree(n))
        G.nodes[n]['rf_importance'] = round(float(imp_map.get(n, 0.0)), 6)
        G.nodes[n]['functional_module'] = mod
        G.nodes[n]['hex_color'] = palette[mod]
        pos[n] = np.array(manual_pos.get(n, (-1.0, 8.0 - idx * 0.5)), dtype=float)
        G.nodes[n]['x'], G.nodes[n]['y'] = float(pos[n][0] * 80.0), float(-pos[n][1] * 80.0)

    nx.write_graphml(G, os.path.join(FIG_DIR, 'Figure_8_STRING_Functional_Network.graphml'), named_key_ids=True)

    fig = plt.figure(figsize=(13.6, 9.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.15], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    for u, v, d in G.edges(data=True):
        sc_val = float(d.get('combined_score', 0.4))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color='#4A5568', lw=1.0 + (sc_val - 0.4) * 5.0, alpha=min(0.82, 0.35 + (sc_val - 0.4) * 0.65), zorder=1)

    degs = dict(G.degree())
    node_list = list(G.nodes())
    xy = np.array([pos[n] for n in node_list])
    ax.scatter(xy[:, 0], xy[:, 1], s=[420 + degs[n] * 175 for n in node_list],
               c=[G.nodes[n]['hex_color'] for n in node_list], edgecolors='#1A202C', linewidths=1.4, zorder=3)

    inside_hubs = {'MUC1', 'MUC5B', 'SFTPB', 'SFTPC', 'SFTPD', 'AQP5', 'CLDN3'}
    offsets = {
        'ASS1': (0, 0.36, 'center', 'bottom'), 'S100A9': (0, 0.36, 'center', 'bottom'), 'AQP1': (0.36, 0.14, 'left', 'center'),
        'HP': (-0.15, 0.36, 'center', 'bottom'), 'C3': (-0.36, 0.14, 'right', 'center'), 'CXCL14': (0, 0.35, 'center', 'bottom'),
        'ISG15': (0, 0.35, 'center', 'bottom'), 'IFI27': (0, -0.36, 'center', 'top'), 'CCL3': (0.38, 0.08, 'left', 'center'),
        'ITGA1': (0, 0.36, 'center', 'bottom'), 'SPP1': (0, 0.40, 'center', 'bottom'), 'CHI3L1': (-0.40, 0, 'right', 'center'),
        'MT-CO3': (0.40, -0.05, 'left', 'center'), 'TMSB10': (0, 0.36, 'center', 'bottom'), 'TMSB4X': (0, -0.38, 'center', 'top'),
        'TMEM176B': (0, 0.36, 'center', 'bottom'), 'TMEM176A': (0, 0.36, 'center', 'bottom'), 'SCGB3A2': (0, 0.46, 'center', 'bottom'),
        'SCGB3A1': (0.40, 0.10, 'left', 'center'), 'SLC34A2': (0, -0.44, 'center', 'top'), 'TFF3': (0.42, 0.05, 'left', 'center'),
        'CEACAM5': (0.40, 0, 'left', 'center'), 'CLDN4': (-0.10, -0.40, 'center', 'top'), 'CD24': (0.10, -0.40, 'center', 'top')
    }
    for n in node_list:
        x, y = pos[n]
        if n in inside_hubs:
            t = ax.text(x, y, n, fontsize=9.4, fontweight='bold', color='white', ha='center', va='center', zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=1.8, foreground='#1A202C')])
        else:
            dx, dy, ha, va = offsets.get(n, (0, 0.38, 'center', 'bottom'))
            t = ax.text(x + dx, y + dy, n, fontsize=9.5, fontweight='bold', color='#111111', ha=ha, va=va, zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=3.2, foreground='white')])

    ax.set_xlim(-2.35, 10.15)
    ax.set_ylim(-1.85, 10.45)
    ax.set_aspect('equal')
    ax.axis('off')

    mod_counts = {m: sum(1 for n in G.nodes() if G.nodes[n]['functional_module'] == m) for m in palette}
    leg1 = ax_leg.legend(handles=[Line2D([0], [0], marker='o', color='w', label=f"{m.replace(' Module', '')} (n={mod_counts[m]})",
                         markerfacecolor=c, markeredgecolor='#1A202C', markersize=11) for m, c in palette.items()],
                         title='Functional Modules', loc='upper left', bbox_to_anchor=(0, 0.92), frameon=False, fontsize=10, title_fontsize=11)
    leg1._legend_box.align = "left"
    ax_leg.add_artist(leg1)

    leg2 = ax_leg.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Degree = 1', markerfacecolor='#CBD5E0', markeredgecolor='#1A202C', markersize=8.5),
        Line2D([0], [0], marker='o', color='w', label='Degree = 6', markerfacecolor='#CBD5E0', markeredgecolor='#1A202C', markersize=13),
        Line2D([0], [0], marker='o', color='w', label='Degree = 11', markerfacecolor='#CBD5E0', markeredgecolor='#1A202C', markersize=17)],
        title='Node Size (Degree)', loc='upper left', bbox_to_anchor=(0, 0.58), frameon=False, fontsize=10, title_fontsize=11, labelspacing=1.15)
    leg2._legend_box.align = "left"
    ax_leg.add_artist(leg2)

    leg3 = ax_leg.legend(handles=[
        Line2D([0], [0], color='#4A5568', lw=1.1, alpha=0.45, label='Medium (0.40)'),
        Line2D([0], [0], color='#4A5568', lw=2.6, alpha=0.65, label='High (0.70)'),
        Line2D([0], [0], color='#4A5568', lw=4.1, alpha=0.82, label='Highest (0.99)')],
        title='STRING Combined Score', loc='upper left', bbox_to_anchor=(0, 0.28), frameon=False, fontsize=10, title_fontsize=11)
    leg3._legend_box.align = "left"
    ax_leg.add_artist(leg3)

    fig.savefig(os.path.join(FIG_DIR, 'Figure_8_STRING_Functional_Network.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("--- SCRIPT 08 COMPLETE ---")

if __name__ == "__main__":
    main()
