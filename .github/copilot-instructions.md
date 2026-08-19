# TRIDENT Application — AI Agent Instructions

## What this repo is
A **Streamlit** web app that serves TRIDENT deep-learning models for predicting acute/chronic
chemical ecotoxicity from SMILES strings. It is packaged as a Docker container and deployed at
https://trident.serve.scilifelab.se/. There is a separate, related repo
(https://github.com/StyrbjornKall/TRIDENT) that contains model training code — this repo is
**inference/serving only**. The app now serves **two generations of the model side by side**:
- **TRIDENT-1** (`pages/trident1_predict_page.py`, `src/trident1/`) — species-group-level model (fish,
  aquatic invertebrates, algae) predicting EC50/EC10.
- **TRIDENT-2** (`pages/trident2_page.py`, `src/trident2/`) — a newer, much broader multimodal
  model: any NCBI-taxid species, simultaneous EC50/EC10/NOEC/LOEC prediction, plus effect,
  administration route, organism lifestage, and concentration unit as explicit model inputs.
  See the "TRIDENT-2 model concept" section below — do not confuse its conventions with TRIDENT-1's.

## Architecture & entry point
- `app.py` is the Streamlit entry point. It builds a native multipage navigation using
  `st.navigation([...], position="top")` / `st.Page(...)` (top horizontal nav bar) and calls
  `pg.run()` to render the selected page. Each `st.Page` wraps a page module's `print_*_page()`
  function directly (no separate page scripts needed).
- `pages/` contains one module per page, each exposing a `print_*_page()` function passed to
  `st.Page` in `app.py`:
  - `home_page.py` — landing page content (marketing sections, logo/graphics)
  - `trident1_predict_page.py` — main prediction UI (single/batch SMILES input → model inference → results table/download)
  - `documentation_page.py`, `publications_page.py`, `contact_page.py`, `space_page.py` — static/content pages
- `src/` is the core library used by the pages (imported as `from src.xxx import ...`):
  - `src/trident1/inference_utils/TRIDENT_for_inference.py` — `TRIDENT_for_inference` class: loads the RoBERTa
    transformer + fine-tuned DNN head for a chosen `model_version` (e.g. `EC50EC10_fish`) from
    HuggingFace Hub (`StyrbjornKall/<model_version>`) and local DNN weights under `models/TRIDENT/`,
    then runs `predict_toxicity(...)`.
  - `src/trident1/inference_utils/model_utils.py` — `DNN_module` (configurable MLP head) and `TRIDENT`
    (combines a HF transformer + `DNN_module`) `nn.Module` classes.
  - `src/trident1/inference_utils/pytorch_data_utils.py` — SMILES preprocessing (canonicalization via RDKit),
    one-hot encoding of endpoint/effect, PyTorch `Dataset`/`DataLoader` construction for inference,
    plus SMILES/chemistry validation helpers and training-data lookup helpers.
  - `src/trident1/inference_utils/plots_for_space.py` — PCA/UMAP plotting of CLS embeddings for the "Explore
    the chemical space" page.
  - `src/trident2/streamlit_inference.py` — the **only** TRIDENT-2 module written for this app (thin
    Streamlit-facing wrapper). Everything else under `src/trident2/inference/` and
    `src/trident2/utils/` is copied verbatim from a separate training/inference repo
    (`trident2_inference`) and shared with its CLI entrypoint `src/trident2_quick_inference.py`
    (that CLI is **not used by the app** — reference/context only, don't wire it in). Because those
    files are copied wholesale, don't hand-edit them for app-specific needs; put
    app-specific logic in `streamlit_inference.py` instead, and if the upstream repo publishes a
    newer copy, a straight `cp -r` replace is expected to keep working with `streamlit_inference.py`
    unchanged as long as the public function/constant names below don't change.
  - `src/custom_download_button.py` — misc UI helper (custom file download button), shared by both
    TRIDENT-1 and TRIDENT-2 pages.
- `models/TRIDENT/*.pt` — pretrained DNN head weights per `model_version` (e.g.
  `final_model_EC50EC10_fish_dnn_saved_weights.pt`). The transformer backbone itself is downloaded
  from the HuggingFace Hub org `StyrbjornKall` at runtime (not stored in-repo).
- `models/TRIDENT2/<run_id>_final_model_epoch_70_hf_trident2/` — **fully self-contained** HF model
  directories (config + custom `modeling_trident2.py`/`configuration_trident2.py` + tokenizer +
  `pytorch_model.bin`) for TRIDENT-2, loaded **locally/offline** via
  `AutoModel.from_pretrained(path, trust_remote_code=True)` — no HuggingFace Hub download needed.
  Two near-identical checkpoints exist (`ftmjjb51_...` and `sxno3jhi_...`); the app uses
  `ftmjjb51_final_model_epoch_70_hf_trident2` (see `MODEL_PATH` in `streamlit_inference.py`).
- `assets/` — SVG images/logos used across pages.
- `data/` — sample SMILES/data files and a `predictions/` output folder for TRIDENT-1;
  `data/trident2/` holds TRIDENT-2's taxonomy/mapping data (see below).
- Usage analytics are handled by `streamlit-analytics2`, wrapped around `pg.run()` in `app.py`. It
  tracks page views/widget interactions and persists to `data/analytics_data.json` (gitignored).
  View the dashboard by appending `?analytics=on` to the app URL; it's gated behind
  `unsafe_password` read from the `ANALYTICS_PASSWORD` env var (defaults to `"changeme"` — always
  set a real value in deployment). This replaced the old hand-rolled `increment_user_stats.py` /
  `access_count.txt` print-based counter.

## Model versioning concept (important domain knowledge)
Models are named `<ENDPOINT>_<SPECIES>`, e.g. `EC50_fish`, `EC10_invertebrates`, `EC50EC10_algae`.
- `ENDPOINT` ∈ `{EC50, EC10, EC50EC10}` (EC50EC10 = combined model, best performance — used exclusively
  in the current UI via `model_type = {'Combined model (best performance)': 'EC50EC10'}`).
- `SPECIES` ∈ `{fish, invertebrates, algae}`.
- Each model version has an allowed list of `effect` values (e.g. MOR, ITX, DVP, REP, MPH, GRO, POP)
  and `endpoint` values (EC50/EC10) — see `effectordering`/`endpointordering` dicts duplicated in both
  `pages/trident1_predict_page.py` and `src/trident1/inference_utils/TRIDENT_for_inference.py`. **If you change the
  allowed effects/endpoints for a model, update both places.**
- `predict_toxicity` requires `SMILES`, `exposure_duration` (hours, log10-transformed internally),
  `endpoint`, and `effect`; it raises `RuntimeError` if endpoint/effect isn't valid for the loaded model.

## TRIDENT-2 model concept (important domain knowledge — distinct from TRIDENT-1 above)
- **No species-group model split**: one model predicts for *any* species (identified by NCBI
  `taxid`, passed straight into the model, not tokenized) and returns **all four endpoints
  simultaneously** (`EC50`, `EC10`, `NOEC`, `LOEC` — as log10(concentration), see below) — this is
  why `pages/trident2_page.py` has no endpoint/species-group radio buttons, unlike TRIDENT-1.
- Model inputs, each tokenized as an angle-bracket-wrapped special token (`<MOR>`, `<oral>`,
  `<adult>`, etc.) concatenated with the SMILES — wrapping is done in
  `generate_inference_metadata()` in `src/trident2/inference/preprocess_input.py`, not something
  the app needs to reimplement:
  - **Effect** — `VALID_EFFECTS` (10): MOR, BEH, CAR, REP, GRO, PHY, ITX, POP, DVP, MPH.
  - **Administration route** — `VALID_ADMINISTRATION_ROUTES` (23 values, optional — `None` skips
    the token entirely rather than tokenizing a placeholder).
  - **Organism lifestage** — `VALID_ORGANISM_LIFESTAGES` (early, mid, adult; optional, same as above).
  - All three constants (plus `validate_inference_effect`/`validate_administration_route`/
    `validate_organism_lifestage`) live in `preprocess_input.py` — **import them, don't hard-code
    duplicate lists** in page code (`streamlit_inference.py` re-exports `EFFECT_OPTIONS`,
    `ADMINISTRATION_ROUTE_OPTIONS`, `ORGANISM_LIFESTAGE_OPTIONS`, prefixed with a `NOT_SPECIFIED`
    sentinel for the two optional fields).
- **Concentration unit** is one-hot encoded (`CONC_UNIT_ENCODING`, 14 values, e.g. `mg/l`, `ppm`,
  `mg/kg`) and **duration** is a numeric feature: log10-transformed hours, with a **separate
  missing-duration one-hot flag** (not a token) used when duration is unknown for a species. The
  combined one-hot vector is **15-dim** (14 conc-unit + 1 duration-missing) — matches
  `onehot_length: 15` / `with_duration_neuron: true` in the model's `config.json`. If this ever
  changes, `CONC_UNIT_ENCODING`/`DURATION_MISSING_ENCODING` in `preprocess_input.py` and the model
  config must stay in sync.
- **Species metadata source of truth**: `data/trident2/taxid2commonname.json` (taxid → common name,
  some values are `null` — handled by falling back to the latin name alone),
  `data/trident2/taxid2sciname.json` (taxid → latin name), and
  `data/trident2/taxa_assigned_units_effects_durations_2026-04-17.csv` (taxid →
  `species_group_corrected` / `assigned_acute_effect` / `assigned_conc_unit` /
  `assigned_acute_duration`, used both as the species-group filter and to pre-fill
  effect/unit/duration defaults). `load_species_options()` in `streamlit_inference.py` only lists
  species present in **all three** sources so every option has a usable label + defaults, and sorts
  by **latin name** (so congeneric species are grouped together in the dropdown) — don't change
  this sort key without checking the UX rationale in the page. The page also exposes a "Filter by
  species group" dropdown (`ALL_SPECIES_GROUPS` sentinel + `species_group` column, sourced from
  `species_group_corrected`) that narrows the species dropdown's options before selection.
- The duration `st.slider` in `pages/trident2_page.py` auto-disables (and shows a warning caption)
  whenever the selected species has no `assigned_acute_duration` in the mapping CSV — in that case
  `duration_hours=None` is passed to `predict_toxicity()`, which relies on
  `generate_inference_metadata()`'s missing-duration fallback (sentinel `1e-6` before log10) rather
  than a user-supplied value.
- `predict_toxicity()`'s output DataFrame also carries the full prediction context as columns —
  `Species common name`, `Species latin name`, `NCBI taxid`, `Species group`, `Effect`,
  `Administration route`, `Organism lifestage`, `Concentration unit`, `Duration (h)` — before the
  endpoint columns; keep this column ordering (metadata → all log10 endpoints → all linear
  endpoints) if extending the output.
- Model output is assumed log10(concentration) in the *user-selected* unit (by analogy with
  TRIDENT-1's `predictions log10(mg/L)` convention) — `predict_toxicity()` in
  `streamlit_inference.py` back-transforms with `10**x` and emits both `"{endpoint} log10(unit)"`
  and `"{endpoint} (unit)"` columns for all four endpoints, log10 columns first then linear columns.
- No CLS-embedding, PCA/UMAP, or training-data-lookup features exist for TRIDENT-2 (unlike
  TRIDENT-1's "Results analysis" section) — this was an explicit scope decision, don't re-add them
  without being asked.

## Caching pattern
`load_automodel` and `load_autotokenizer` in `src/trident1/inference_utils/TRIDENT_for_inference.py`,
and `load_model_and_tokenizer`/`load_species_options` in `src/trident2/streamlit_inference.py`, all use
`@st.cache_resource` (model/tokenizer) or `@st.cache_data` (species/taxonomy tables) — this is how the
(large) transformer models and lookup tables avoid being reloaded on every Streamlit rerun. Keep this
pattern when modifying model-loading code; don't bypass Streamlit's caching or the app will become
very slow / re-load models per interaction.

## Running & building
- **Dependency management uses `uv`** (`uv.lock` is committed) — do **not** run bare `pip install`.
  Add new deps to *both* `pyproject.toml`'s `[project.dependencies]` and `requirements.txt` (kept in
  sync manually — `requirements.txt` is what the Docker image installs from), then run `uv sync` to
  update the local `.venv`. Run the app with `uv run streamlit run app.py` (or activate `.venv`).
- Local run (no Docker, without uv): `pip install -r requirements.txt` then `streamlit run app.py`.
- Docker: `docker build -t streamlit-app-trident-serve-v1 .` then
  `docker run -p 8501:8501 streamlit-app-trident-serve-v1` (entrypoint is `start-script.sh`, which
  runs `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`).
- `packages.txt` lists apt packages needed (RDKit/Streamlit ecosystem deps); `requirements.txt`
  pins Python deps (torch 2.1.2, streamlit 1.48.0, transformers 4.36.2, rdkit 2022.03.5, etc.) —
  keep versions in sync with what the pretrained model weights expect. Streamlit's native
  `st.navigation`/`st.Page` (top nav bar) requires a reasonably recent Streamlit; `hydralit_components`
  was removed in favor of this native approach.
- CI: `.github/workflows/build-docker.yml` is a manually-triggered (`workflow_dispatch`) workflow
  that builds the Docker image, optionally pushes to Docker Hub, and attaches the image tarball to
  a GitHub Release. There is no automated test suite in this repo.

## Conventions & gotchas
- No automated tests exist; validate changes by running the Streamlit app locally and exercising
  the "Use TRIDENT" page (single prediction + batch upload with `data/smiles.txt` or generated
  example molecules).
- Imports from the library code always go through the `src` package, e.g.
  `from src.trident1.inference_utils.TRIDENT_for_inference import TRIDENT_for_inference` or
  `from src.custom_download_button import download_button` — do not import `inference_utils`
  or `custom_download_button` at top level, those old paths no longer exist.
- `pages/` is a plain Python package (has `__init__.py`) of page modules with `print_*_page()`
  functions; it is **not** Streamlit's legacy auto-discovered `pages/` directory mechanism (which
  only activates when the entrypoint does *not* call `st.navigation`). `app.py` always calls
  `st.navigation(...)` + `pg.run()`, so add new pages by creating a module in `pages/` and adding
  an `st.Page(...)` entry in `app.py`, not by relying on file-based auto-routing.
- DNN checkpoint loading (`__loadcheckpoint__` in `TRIDENT_for_inference.py`) reads from
  `./models/TRIDENT/final_model_<version>_dnn_saved_weights.pt` relative to the process's working
  directory (repo root) — keep this path in sync if `models/TRIDENT/` is ever moved again.
- SMILES columns are normalized to `SMILES` (accepts `smiles`/`Smiles` input column names) — see
  `TRIDENT_for_inference.predict_toxicity`.
- Batch file uploads support `.csv` (comma), `.txt` (tab), `.xlsx` — parsing logic lives in
  `pages/trident1_predict_page.py`'s file-uploader branch; keep the three formats' handling consistent.
- Streamlit session state (`st.session_state`) is used heavily for batch data and UI toggles
  (`predict_all`, `batch`, `return_cls_embeddings`, `current_batch`, `example_mol`) — check existing
  keys before adding new ones to avoid collisions.
- RDKit is used for SMILES validation/canonicalization and structure rendering (`Draw.MolsToGridImage`);
  invalid SMILES should be handled gracefully (existing code wraps rendering in `try/except`).
- TRIDENT-2's `src/trident2/inference/*` and `src/trident2/utils/*` modules depend on `loguru`
  (added to `pyproject.toml`/`requirements.txt` for this reason) — don't remove it even though
  TRIDENT-1 code doesn't use it.
- TRIDENT-2 session state keys are prefixed `trident2_` (`trident2_batch`, `trident2_current_batch`,
  `trident2_example_mol`) to avoid collisions with TRIDENT-1's unprefixed keys (`batch`,
  `current_batch`, `example_mol`) in the same `st.session_state` — keep this prefix convention for
  any new TRIDENT-2 session state.
- `pages/trident2_page.py`'s batch mode only accepts a **SMILES-only** file (`.csv`/`.txt`/`.xlsx`,
  one column) — unlike the CLI (`trident2_quick_inference.py`), it does **not** support uploading a
  taxid column or per-row metadata; species/effect/route/lifestage/unit are chosen once via the
  sidebar dropdowns and applied to the whole batch.
