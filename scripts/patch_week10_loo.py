"""
patch_week10_loo.py
===================
Modifica week10_offline_evaluation.ipynb:
  1. Corrige el docstring de recall_at_k (celda 8) para documentar la limitación.
  2. Actualiza celda 1 (protocolo) para mencionar LOO añadido.
  3. Actualiza celda 22 (JSON summary) para incluir sección LOO.
  4. Añade 7 celdas nuevas al final con la evaluación LOO completa.
"""

import json
import re
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "week10" / "week10_offline_evaluation.ipynb"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": _uid(), "metadata": {}, "source": source}

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _uid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def _uid():
    import uuid
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Load notebook
# ---------------------------------------------------------------------------
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ---------------------------------------------------------------------------
# 1. Fix recall_at_k docstring (cell 8)
# ---------------------------------------------------------------------------
OLD_RECALL = '''\
def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    """Fraction of all relevant items retrieved in top-K."""
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant'''

NEW_RECALL = '''\
def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    """Fraction of all relevant items retrieved in top-K.

    NOTE (corrección): cuando se usa evaluación item-to-item por género,
    el `total_relevant` debe ser el número de ítems relevantes en el universo
    completo del catálogo — NO en la lista candidata del sistema.  Usar el
    pool candidato como denominador hace que el Recall dependa del output del
    propio sistema, sesgando la métrica al alza.

    En la evaluación Leave-One-Out (sección 11 en adelante), este problema
    no aplica porque hay exactamente 1 ítem de test por usuario; en ese caso
    Recall@K ≡ Hit Rate@K y se reporta como tal.
    """
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant'''

src8 = "".join(cells[8]["source"])
if OLD_RECALL in src8:
    cells[8]["source"] = src8.replace(OLD_RECALL, NEW_RECALL)
    print("✓ Celda 8: docstring recall_at_k corregido")
else:
    print("⚠ Celda 8: no encontró el texto exacto; revisa manualmente")

# ---------------------------------------------------------------------------
# 2. Update cell 1 (protocol markdown) — add LOO note
# ---------------------------------------------------------------------------
LOO_NOTE = """

---

> **Actualización (LOO):** Se añadió una segunda capa de evaluación —
> **Leave-One-Out (LOO)** — en las secciones 11–15.  Para cada usuario se
> oculta su último ítem (ordenado por timestamp) y se mide si los tres sistemas
> lo recuperan en el top-K.  La métrica central en LOO es **Hit Rate@K**
> (equivalente a Recall@K cuando hay exactamente 1 ítem de test).
>
> El Recall@K de la evaluación por género tenía un defecto: su denominador era
> el pool candidato del sistema, no el universo real.  Queda documentado en la
> función `recall_at_k` y el LOO lo reemplaza como métrica de recall correcta.
"""

cell1_src = "".join(cells[1]["source"])
if "Actualización (LOO)" not in cell1_src:
    cells[1]["source"] = cell1_src + LOO_NOTE
    print("✓ Celda 1: nota LOO añadida")
else:
    print("· Celda 1: ya tenía nota LOO")

# ---------------------------------------------------------------------------
# 3. New LOO cells
# ---------------------------------------------------------------------------

LOO_SECTION_HEADER = """\
## 11) Leave-One-Out (LOO) Evaluation

La evaluación por géneros mide *relevancia de catálogo* (¿los ítems recomendados
comparten género?), pero no mide si el sistema predice correctamente el próximo
consumo del usuario.

**Protocolo LOO**
- Cargar todos los ratings con timestamp.
- Filtrar usuarios con **≥ 5 ratings**.
- Por usuario: ocultar el **último ítem** (el más reciente) como test; el resto
  es entrenamiento.
- Medir si el ítem oculto aparece en el top-K generado por cada sistema.

**Métricas**
- **Hit Rate@K** (= Recall@K en LOO): ¿el ítem oculto está en top-K?
- **Precision@K**: fracción de top-K que son relevantes (para LOO = 1/K si hay hit).
- **NDCG@K**: hit ponderado por posición.

> Se usa una **muestra aleatoria de 10 000 usuarios** para mantener el tiempo
> de cómputo razonable (seed=42, reproducible).
"""

LOO_SPLIT_CODE = """\
import random

LOO_SAMPLE_USERS = 10_000
LOO_SEED = 42
K_VALUES_LOO = [5, 10, 20]

# Load ratings
ratings_all = pl.read_parquet(DATA_DIR / 'ratings_clean.parquet')

# Keep users with >= 5 ratings
user_counts = ratings_all.group_by('userId').agg(pl.len().alias('n'))
active_user_ids = user_counts.filter(pl.col('n') >= 5)['userId'].to_list()
print(f'Users with >=5 ratings: {len(active_user_ids):,}')

# Subsample for speed
random.seed(LOO_SEED)
sampled_user_ids = random.sample(active_user_ids, min(LOO_SAMPLE_USERS, len(active_user_ids)))
print(f'Sampled users: {len(sampled_user_ids):,}')

# Filter and rank by timestamp (rank 1 = most recent)
ratings_sampled = ratings_all.filter(pl.col('userId').is_in(sampled_user_ids))
ratings_ranked = ratings_sampled.with_columns(
    pl.col('timestamp')
    .rank(method='ordinal', descending=True)
    .over('userId')
    .alias('rank_desc')
)

loo_test  = (ratings_ranked
             .filter(pl.col('rank_desc') == 1)
             .select(['userId', 'movieId'])
             .rename({'movieId': 'test_movieId'}))

loo_train = (ratings_ranked
             .filter(pl.col('rank_desc') != 1)
             .select(['userId', 'movieId', 'rating']))

print(f'LOO test set:  {loo_test.height:,} users')
print(f'LOO train set: {loo_train.height:,} ratings')
print(loo_test.head(3))
"""

LOO_POP_CODE = """\
# --- LOO: Popularity baseline ---
# Top-500 most-rated items in train; exclude items already seen by the user.

POP_POOL = 500

pop_train_counts = (loo_train
                    .group_by('movieId')
                    .agg(pl.len().alias('n_ratings'))
                    .sort('n_ratings', descending=True))
pop_top_ids_loo = pop_train_counts.head(POP_POOL)['movieId'].to_list()

# Build per-user seen set (from train)
user_seen_map: dict[int, set] = {}
for row in loo_train.iter_rows(named=True):
    user_seen_map.setdefault(row['userId'], set()).add(row['movieId'])

# Evaluate
loo_pop_hits  = {k: 0 for k in K_VALUES_LOO}
loo_pop_ndcg  = {k: 0.0 for k in K_VALUES_LOO}
loo_pop_prec  = {k: 0.0 for k in K_VALUES_LOO}
loo_n_users = 0

for row in loo_test.iter_rows(named=True):
    uid, tid = row['userId'], row['test_movieId']
    seen = user_seen_map.get(uid, set())
    recs = [m for m in pop_top_ids_loo if m not in seen]
    for k in K_VALUES_LOO:
        top_k = recs[:k]
        hit = tid in top_k
        loo_pop_hits[k] += int(hit)
        loo_pop_prec[k] += (1 / k) if hit else 0.0
        if hit:
            pos = top_k.index(tid) + 1
            loo_pop_ndcg[k] += 1.0 / math.log2(pos + 1)
    loo_n_users += 1

print(f'Popularity LOO (n={loo_n_users:,}):')
for k in K_VALUES_LOO:
    print(f'  Hit Rate@{k}: {loo_pop_hits[k]/loo_n_users:.4f} | '
          f'Precision@{k}: {loo_pop_prec[k]/loo_n_users:.4f} | '
          f'NDCG@{k}: {loo_pop_ndcg[k]/loo_n_users:.4f}')
"""

LOO_CONTENT_CODE = """\
# --- LOO: Content-based (SVD item factors as item embedding) ---
# User profile = weighted average of train item vectors.
# Score = cosine similarity between profile and each candidate item.

factor_cols_loo = [c for c in item_factors.columns if c.startswith('svd_')]
item_ids_loo = item_factors['movieId'].to_list()
item_matrix_loo = item_factors.select(factor_cols_loo).to_numpy()
item_id_to_idx_loo = {mid: i for i, mid in enumerate(item_ids_loo)}
item_id_set_loo = set(item_ids_loo)

# L2-normalise
_norms = np.linalg.norm(item_matrix_loo, axis=1, keepdims=True)
_norms[_norms == 0] = 1
item_matrix_norm_loo = item_matrix_loo / _norms

# Build user train histories (items with known factors only)
user_history_loo: dict[int, list[tuple[int, float]]] = {}
for row in loo_train.iter_rows(named=True):
    mid = row['movieId']
    if mid in item_id_to_idx_loo:
        user_history_loo.setdefault(row['userId'], []).append((mid, row['rating']))

# Evaluate
loo_cb_hits  = {k: 0 for k in K_VALUES_LOO}
loo_cb_ndcg  = {k: 0.0 for k in K_VALUES_LOO}
loo_cb_prec  = {k: 0.0 for k in K_VALUES_LOO}
loo_cb_n = 0

for row in loo_test.iter_rows(named=True):
    uid, tid = row['userId'], row['test_movieId']
    if tid not in item_id_set_loo:
        continue
    history = user_history_loo.get(uid, [])
    if not history:
        continue
    seen = {m for m, _ in history}
    weights = np.array([r for _, r in history], dtype=np.float32)
    vecs = np.stack([item_matrix_norm_loo[item_id_to_idx_loo[m]] for m, _ in history])
    profile = (vecs * weights[:, None]).sum(axis=0)
    pnorm = np.linalg.norm(profile)
    if pnorm < 1e-9:
        continue
    profile /= pnorm
    scores = item_matrix_norm_loo @ profile
    # sort candidates (exclude seen)
    cand_idx = [i for i, m in enumerate(item_ids_loo) if m not in seen]
    cand_sorted = sorted(cand_idx, key=lambda i: -scores[i])
    top_ids = [item_ids_loo[i] for i in cand_sorted]
    for k in K_VALUES_LOO:
        top_k = top_ids[:k]
        hit = tid in top_k
        loo_cb_hits[k] += int(hit)
        loo_cb_prec[k] += (1 / k) if hit else 0.0
        if hit:
            pos = top_k.index(tid) + 1
            loo_cb_ndcg[k] += 1.0 / math.log2(pos + 1)
    loo_cb_n += 1

print(f'Content-based LOO (n={loo_cb_n:,}):')
for k in K_VALUES_LOO:
    print(f'  Hit Rate@{k}: {loo_cb_hits[k]/loo_cb_n:.4f} | '
          f'Precision@{k}: {loo_cb_prec[k]/loo_cb_n:.4f} | '
          f'NDCG@{k}: {loo_cb_ndcg[k]/loo_cb_n:.4f}')
"""

LOO_SVD_CODE = """\
# --- LOO: SVD collaborative (item-factor dot-product score) ---
# For each user, compute a user vector = mean of train item factors (unweighted).
# Score = dot product between user vector and each candidate item factor.
# (Approximates the standard SVD user-item score without re-fitting.)

loo_svd_hits  = {k: 0 for k in K_VALUES_LOO}
loo_svd_ndcg  = {k: 0.0 for k in K_VALUES_LOO}
loo_svd_prec  = {k: 0.0 for k in K_VALUES_LOO}
loo_svd_n = 0

for row in loo_test.iter_rows(named=True):
    uid, tid = row['userId'], row['test_movieId']
    if tid not in item_id_to_idx_loo:
        continue
    history = user_history_loo.get(uid, [])
    if not history:
        continue
    seen = {m for m, _ in history}
    # unweighted mean of raw (non-normalised) item factors as user vector
    vecs_raw = np.stack([item_matrix_loo[item_id_to_idx_loo[m]] for m, _ in history])
    user_vec = vecs_raw.mean(axis=0)
    scores = item_matrix_loo @ user_vec
    cand_idx = [i for i, m in enumerate(item_ids_loo) if m not in seen]
    cand_sorted = sorted(cand_idx, key=lambda i: -scores[i])
    top_ids = [item_ids_loo[i] for i in cand_sorted]
    for k in K_VALUES_LOO:
        top_k = top_ids[:k]
        hit = tid in top_k
        loo_svd_hits[k] += int(hit)
        loo_svd_prec[k] += (1 / k) if hit else 0.0
        if hit:
            pos = top_k.index(tid) + 1
            loo_svd_ndcg[k] += 1.0 / math.log2(pos + 1)
    loo_svd_n += 1

print(f'SVD collaborative LOO (n={loo_svd_n:,}):')
for k in K_VALUES_LOO:
    print(f'  Hit Rate@{k}: {loo_svd_hits[k]/loo_svd_n:.4f} | '
          f'Precision@{k}: {loo_svd_prec[k]/loo_svd_n:.4f} | '
          f'NDCG@{k}: {loo_svd_ndcg[k]/loo_svd_n:.4f}')
"""

LOO_RESULTS_CODE = """\
# --- LOO results table ---
loo_rows = []
for sys_name, hits, prec, ndcg, n_q in [
    ('popularity_global', loo_pop_hits, loo_pop_prec, loo_pop_ndcg, loo_n_users),
    ('content_cosine',    loo_cb_hits,  loo_cb_prec,  loo_cb_ndcg,  loo_cb_n),
    ('svd_collaborative', loo_svd_hits, loo_svd_prec, loo_svd_ndcg, loo_svd_n),
]:
    for k in K_VALUES_LOO:
        loo_rows.append({
            'system': sys_name,
            'k': k,
            'hit_rate_at_k':  round(hits[k] / n_q, 4),
            'precision_at_k': round(prec[k] / n_q, 4),
            'ndcg_at_k':      round(ndcg[k] / n_q, 4),
            'n_queries': n_q,
        })

loo_df = pl.DataFrame(loo_rows)
loo_df.write_csv(ARTIFACTS_DIR / 'week10_loo_evaluation_results.csv')
print('LOO evaluation results:')
display(loo_df.to_pandas().to_string(index=False))

# Bar chart comparison
fig_loo = make_subplots(
    rows=1, cols=3,
    subplot_titles=['Hit Rate@K (LOO)', 'Precision@K (LOO)', 'NDCG@K (LOO)'],
)
loo_colors = {'popularity_global': '#10b981', 'content_cosine': '#6366f1', 'svd_collaborative': '#f59e0b'}
loo_labels = {'popularity_global': 'Popularity', 'content_cosine': 'Content (cosine)', 'svd_collaborative': 'SVD (collab.)'}
k_labels_loo = [str(k) for k in K_VALUES_LOO]

for col_idx, (metric_col, metric_name) in enumerate(
    [('hit_rate_at_k', 'Hit Rate'), ('precision_at_k', 'Precision'), ('ndcg_at_k', 'NDCG')], 1
):
    for sys_name in ['popularity_global', 'content_cosine', 'svd_collaborative']:
        sys_data = loo_df.filter(pl.col('system') == sys_name).sort('k')
        fig_loo.add_trace(
            go.Bar(
                name=loo_labels[sys_name],
                x=k_labels_loo,
                y=sys_data[metric_col].to_list(),
                marker_color=loo_colors[sys_name],
                showlegend=(col_idx == 1),
            ),
            row=1, col=col_idx,
        )
        fig_loo.update_xaxes(title_text='K', row=1, col=col_idx)

fig_loo.update_layout(
    title='Leave-One-Out evaluation: Hit Rate, Precision, NDCG @ K ∈ {5, 10, 20}',
    height=500,
    template='plotly_white',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
)
fig_loo.write_html(ARTIFACTS_DIR / 'week10_loo_evaluation_comparison.html')
fig_loo.write_image(ARTIFACTS_DIR / 'week10_loo_evaluation_comparison.png', scale=2)
fig_loo.show()
"""

LOO_SAVE_CODE = """\
# --- Actualizar eval_summary con resultados LOO ---
# Re-leer el summary existente y añadir la sección LOO

with open(ARTIFACTS_DIR / 'week10_evaluation_summary.json') as f:
    eval_summary_updated = json.load(f)

eval_summary_updated['loo_evaluation'] = {
    'evaluation_type': 'leave_one_out_user_history',
    'split_criterion': 'last_item_by_timestamp',
    'min_ratings_per_user': 5,
    'sample_users': LOO_SAMPLE_USERS,
    'random_seed': LOO_SEED,
    'k_values': K_VALUES_LOO,
    'recall_note': (
        'En LOO, Recall@K = Hit Rate@K porque hay exactamente 1 ítem de test '
        'por usuario. El Recall@K de la evaluación por género (secciones 1–10) '
        'tenía denominador igual al pool candidato del sistema — no al universo '
        'real — lo que sobreestimaba la métrica. Queda documentado en recall_at_k().'
    ),
    'results': {
        sys_name: {
            f'k_{k}': {
                'hit_rate':  round(hits[k] / n_q, 6),
                'precision': round(prec[k] / n_q, 6),
                'ndcg':      round(ndcg[k] / n_q, 6),
            }
            for k in K_VALUES_LOO
        }
        for sys_name, hits, prec, ndcg, n_q in [
            ('popularity_global', loo_pop_hits, loo_pop_prec, loo_pop_ndcg, loo_n_users),
            ('content_cosine',    loo_cb_hits,  loo_cb_prec,  loo_cb_ndcg,  loo_cb_n),
            ('svd_collaborative', loo_svd_hits, loo_svd_prec, loo_svd_ndcg, loo_svd_n),
        ]
    },
    'artifacts': [
        'week10_loo_evaluation_results.csv',
        'week10_loo_evaluation_comparison.html',
        'week10_loo_evaluation_comparison.png',
    ],
}

with open(ARTIFACTS_DIR / 'week10_evaluation_summary.json', 'w') as f:
    json.dump(eval_summary_updated, f, indent=2)

print('✓ week10_evaluation_summary.json actualizado con sección LOO.')
print(json.dumps(eval_summary_updated['loo_evaluation'], indent=2))
"""

# Build new cells list to append
new_cells = [
    md_cell(LOO_SECTION_HEADER),
    code_cell(LOO_SPLIT_CODE),
    md_cell("### Popularity LOO"),
    code_cell(LOO_POP_CODE),
    md_cell("### Content-based LOO (SVD item embeddings)"),
    code_cell(LOO_CONTENT_CODE),
    md_cell("### SVD Collaborative LOO"),
    code_cell(LOO_SVD_CODE),
    md_cell("### LOO Results table and chart"),
    code_cell(LOO_RESULTS_CODE),
    md_cell("### Save updated evaluation summary"),
    code_cell(LOO_SAVE_CODE),
]

# Check not already patched
existing_sources = ["".join(c["source"]) for c in cells]
if any("LOO_SAMPLE_USERS" in s for s in existing_sources):
    print("· LOO cells already present — skipping append")
else:
    cells.extend(new_cells)
    print(f"✓ Añadidas {len(new_cells)} celdas LOO al final del notebook")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
nb["cells"] = cells
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Notebook guardado: {NB_PATH}")
print(f"  Total celdas: {len(cells)}")
