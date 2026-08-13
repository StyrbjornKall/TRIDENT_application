# TRIDENT Application — AI Agent Instructions

## What this repo is
A **Streamlit** web app that serves the TRIDENT deep-learning model for predicting acute/chronic
chemical ecotoxicity (fish, aquatic invertebrates, algae) from SMILES strings. It is packaged as a
Docker container and deployed at https://trident.serve.scilifelab.se/. There is a separate,
related repo (https://github.com/StyrbjornKall/TRIDENT) that contains model training code — this
repo is **inference/serving only**.

## Architecture & entry point
- `app.py` is the Streamlit entry point. It builds a native multipage navigation using
  `st.navigation([...], position="top")` / `st.Page(...)` (top horizontal nav bar) and calls
  `pg.run()` to render the selected page. Each `st.Page` wraps a page module's `print_*_page()`
  function directly (no separate page scripts needed).
- `pages/` contains one module per page, each exposing a `print_*_page()` function passed to
  `st.Page` in `app.py`:
  - `home_page.py` — landing page content (marketing sections, logo/graphics)
  - `predict_page.py` — main prediction UI (single/batch SMILES input → model inference → results table/download)
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
  - `src/custom_download_button.py` — misc UI helper (custom file download button).
- `models/TRIDENT/*.pt` — pretrained DNN head weights per `model_version` (e.g.
  `final_model_EC50EC10_fish_dnn_saved_weights.pt`). The transformer backbone itself is downloaded
  from the HuggingFace Hub org `StyrbjornKall` at runtime (not stored in-repo).
- `assets/` — SVG images/logos used across pages.
- `data/` — sample SMILES/data files and a `predictions/` output folder.
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
  `pages/predict_page.py` and `src/trident1/inference_utils/TRIDENT_for_inference.py`. **If you change the
  allowed effects/endpoints for a model, update both places.**
- `predict_toxicity` requires `SMILES`, `exposure_duration` (hours, log10-transformed internally),
  `endpoint`, and `effect`; it raises `RuntimeError` if endpoint/effect isn't valid for the loaded model.

## Caching pattern
`load_automodel` and `load_autotokenizer` in `src/trident1/inference_utils/TRIDENT_for_inference.py` use
`@st.cache_resource` — this is how the (large) transformer models avoid being reloaded on every
Streamlit rerun. Keep this pattern when modifying model-loading code; don't bypass Streamlit's
caching or the app will become very slow / re-download models per interaction.

## Running & building
- Local run (no Docker): `pip install -r requirements.txt` then `streamlit run app.py`.
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
  `pages/predict_page.py`'s file-uploader branch; keep the three formats' handling consistent.
- Streamlit session state (`st.session_state`) is used heavily for batch data and UI toggles
  (`predict_all`, `batch`, `return_cls_embeddings`, `current_batch`, `example_mol`) — check existing
  keys before adding new ones to avoid collisions.
- RDKit is used for SMILES validation/canonicalization and structure rendering (`Draw.MolsToGridImage`);
  invalid SMILES should be handled gracefully (existing code wraps rendering in `try/except`).
