import base64

import streamlit as st


def _svg_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def print_home_page(predict_pg, trident2_pg, docs_pg):
    logo_b64 = _svg_b64("assets/trident-logo.svg")

    # ---------------------------------------------------------------
    # Hero
    # ---------------------------------------------------------------
    st.markdown(
        f"""
        <div class="tri-hero">
            <img src="data:image/svg+xml;base64,{logo_b64}" style="width:min(1400px, 80%);">
            <h1>Deep learning models for <span class="tri-accent">chemical toxicity prediction</span></h1>
            <p class="tri-subtitle">
                TRIDENT is a set of transformer-based deep learning models that predict
                ecotoxicity endpoints from a chemical's structure (SMILES). TRIDENT-1 predicts
                acute and chronic effect concentrations for standard aquatic test species
                (fish, aquatic invertebrates, algae). TRIDENT-2 extends prediction to any species
                with an NCBI taxonomy identifier. Both models are intended to support
                environmental hazard and risk assessment, including under REACH.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, cta_col, _ = st.columns([1, 2, 1])
    with cta_col:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="tri-btn-primary-hero1"):
                st.page_link(predict_pg, label="TRIDENT-1", icon=":material/arrow_forward:", use_container_width=True)
        with c2:
            with st.container(key="tri-btn-primary-hero2"):
                st.page_link(trident2_pg, label="TRIDENT-2", icon=":material/arrow_forward:", use_container_width=True)

    # ---------------------------------------------------------------
    # Two model generations, side by side
    # ---------------------------------------------------------------
    st.markdown(
        """
        <div class="tri-section-head">
            <h2>TRIDENT-1 and TRIDENT-2</h2>
            <p>Both models learn a continuous representation of chemical structure and map it to
            toxicity endpoints. TRIDENT-1 is restricted to standard aquatic test species groups used in
            regulatory testing; TRIDENT-2 extends prediction to thousands of eukaryotic species.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, panel_col, _ = st.columns([1, 8, 1])
    with panel_col:
        p1, p2 = st.columns(2, gap="medium")
        with p1:
            st.markdown(
                """
                <div class="tri-model-panel tri-panel-1">
                    <span class="tri-model-tag">TRIDENT-1</span>
                    <h3>Aquatic ecotoxicity model</h3>
                    <p>
                        TRIDENT-1 is a transformer-based deep learning model trained on
                        experimental ecotoxicity data compiled from REACH dossiers, US EPA ECOTOX,
                        and EFSA openFoodTox. It predicts acute and chronic effect concentrations
                        (EC50, EC10/NOEC) for fish, aquatic invertebrates, and algae, and has been
                        benchmarked against several QSAR tools and published in Science Advances (2024).
                    </p>
                    <div class="tri-chip-row">
                        <span class="tri-chip">Fish</span>
                        <span class="tri-chip">Aquatic invertebrates</span>
                        <span class="tri-chip">Algae</span>
                        <span class="tri-chip">EC50 / EC10 / NOEC</span>
                        <span class="tri-chip">7 toxicological effects</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                """
                <div class="tri-model-panel tri-panel-2">
                    <span class="tri-model-tag">TRIDENT-2</span>
                    <h3>Toxicity prediction across Eukaryota</h3>
                    <p>
                        TRIDENT-2 is a multimodal model that takes a eukaryotic species, chemical structure, effect, administration route, organism lifestage,
                        exposure duration and concentration unit as explicit inputs, and predicts EC50, EC10, NOEC,
                        and LOEC simultaneously. This extends effect-concentration prediction
                        beyond the standard aquatic test species used in regulatory testing to
                        any thousands of eukaryotic species, supporting chemical risk
                        assessment that accounts for a broader range of biodiversity.
                    </p>
                    <div class="tri-chip-row">
                        <span class="tri-chip">20 eukaryotic groups</span>
                        <span class="tri-chip">Thousands of species</span>
                        <span class="tri-chip">EC50 / EC10 / NOEC / LOEC</span>
                        <span class="tri-chip">10 toxicological effects</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------
    # Why TRIDENT — feature cards
    # ---------------------------------------------------------------
    st.markdown(
        """
        <div class="tri-section-head">
            <h2>Method and validation</h2>
            <p>Both TRIDENT-1 and TRIDENT-2 have been thoroughly evaluated, demonstrating strong predictive performance surpassing other tools and models. Full methodological
            detail is provided in the associated publications.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        (
            "experiment",
            "Toxicological diversity",
            "TRIDENT supports prediction for a wide range of toxicological species, species groups, effects and endpoints.",
        ),
        (
            "verified",
            "State-of-the-art performance",
            "TRIDENT-1 and TRIDENT-2 are described in peer-reviewed publications demonstrating superior predictive performance in terms of both accuracy and predictability compared to previous methods.",
        ),
        (
            "public",
            "Taxonomic breadth",
            "TRIDENT-2 can predict toxicity towards thousands of eukaryotes, far beyond the standard aquatic test organisms.",
        ),
        (
            "hexagon",
            "Chemical variety",
            "Both TRIDENT-1 and TRIDENT-2 are trained on a large and diverse set of chemicals, including pesticides, pharmaceuticals, industrial chemicals, and consumer products. The models can therefore estimate toxicity for an unprecedentedly wide range of chemical structures, including novel compounds.",
        ),
    ]
    _, feat_col, _ = st.columns([1, 8, 1])
    with feat_col:
        cols = st.columns(4, gap="medium")
        for i, (col, (icon, title, body)) in enumerate(zip(cols, features)):
            with col:
                with st.container(key=f"tri-feature-{i}"):
                    st.markdown(f":material/{icon}:")
                    st.markdown(f"**{title}**")
                    st.caption(body)
