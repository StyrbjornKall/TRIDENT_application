import random

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

_transparent_draw_options = rdMolDraw2D.MolDrawOptions()
_transparent_draw_options.clearBackground = False

from src.custom_download_button import download_button
from src.trident2.streamlit_inference import (
    ALL_SPECIES_GROUPS,
    CONC_UNIT_OPTIONS,
    EFFECT_OPTIONS,
    NOT_SPECIFIED,
    get_valid_lifestage_options_for_species,
    get_valid_route_options_for_species_and_lifestage,
    load_species_options,
    predict_toxicity,
)

example_mols = [
    'O=P(O)(O)O', 'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'Cc1ccccc1Cl', 'C=CC(=O)OCC',
    'ClC(Cl)C(Cl)(Cl)Cl', 'O=C(O)CNCP(=O)(O)O', 'CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC',
    'CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl',
]


def get_example_mol():
    current_example_mol = st.session_state.trident2_example_mol
    while True:
        next_example_mol = example_mols[random.randint(0, len(example_mols) - 1)]
        if current_example_mol != next_example_mol:
            break
    st.session_state.trident2_example_mol = next_example_mol


def _option_index(options: list, value, default: int = 0) -> int:
    """Return the index of value in options, falling back to default if absent/NaN."""
    if value in options:
        return options.index(value)
    return default


def print_trident2_predict_page():
    if 'trident2_example_mol' not in st.session_state:
        st.session_state.trident2_example_mol = 'C1=CC=CC=C1'
    if 'trident2_current_batch' not in st.session_state:
        st.session_state.trident2_current_batch = pd.DataFrame()
    if 'trident2_batch' not in st.session_state:
        st.session_state.trident2_batch = False

    data = pd.DataFrame()
    species_df = load_species_options()

    col1, col2 = st.columns([1, 3])
    col1.markdown('## Prediction settings')
    col1.checkbox("Batch upload (.csv, .txt, .xlsx)", key="trident2_batch")

    species_group_choice = col1.selectbox(
        "Filter by species group",
        options=[ALL_SPECIES_GROUPS] + sorted(species_df['species_group'].dropna().unique().tolist()),
        help="Narrow the species dropdown below to a eukaryotic group, e.g. 'fish'.",
    )
    if species_group_choice != ALL_SPECIES_GROUPS:
        species_df = species_df[species_df['species_group'] == species_group_choice].reset_index(drop=True)

    species_label = col1.selectbox(
        "Select species (common name (latin name))",
        options=species_df['label'].tolist(),
        help="Start typing a common or latin name to search.",
    )
    species_row = species_df.set_index('label').loc[species_label]
    taxid = species_row['taxid']

    effect = col1.selectbox(
        "Select effect",
        EFFECT_OPTIONS,
        index=_option_index(EFFECT_OPTIONS, species_row['default_effect']),
        help="Defaults to the standard effect recorded for this species; change if needed.",
    )

    valid_lifestage_options = get_valid_lifestage_options_for_species(taxid)
    organism_lifestage_choice = col1.selectbox(
        "Select organism lifestage",
        valid_lifestage_options,
        index=0,
    )
    valid_route_options = get_valid_route_options_for_species_and_lifestage(
        taxid,
        organism_lifestage_choice,
    )
    administration_route_choice = col1.selectbox(
        "Select administration route",
        valid_route_options,
        index=0,
    )
    conc_unit = col1.selectbox(
        "Select concentration unit",
        CONC_UNIT_OPTIONS,
        index=_option_index(CONC_UNIT_OPTIONS, species_row['default_conc_unit']),
        help="Defaults to the standard unit recorded for this species; change if needed.",
    )

    duration_available = pd.notna(species_row['default_duration'])
    default_duration = int(species_row['default_duration']) if duration_available else 96
    duration_hours = col1.slider(
        "Select exposure duration (hours)",
        min_value=1, max_value=2160, value=default_duration,
        disabled=not duration_available,
        help="No standard exposure duration is available for this species; "
             "the model will use its 'duration unknown' flag instead."
        if not duration_available else None,
    )
    if not duration_available:
        col1.caption("Duration unavailable for this species — predicting with duration marked as unknown.")

    administration_route = None if administration_route_choice == NOT_SPECIFIED else administration_route_choice
    organism_lifestage = None if organism_lifestage_choice == NOT_SPECIFIED else organism_lifestage_choice
    duration_for_model = duration_hours if duration_available else None

    results = pd.DataFrame()

    with col2:
        st.markdown('# Predict chemical ecotoxicity with TRIDENT-2')
        st.caption(
            "TRIDENT-2 predicts EC50, EC10, NOEC, and LOEC simultaneously for the selected "
            "species, effect, administration route, organism lifestage, and concentration unit."
        )

        if st.session_state.trident2_batch:
            subcol1, subcol2 = st.columns([3, 1], vertical_alignment='bottom')
            with subcol1:
                file_up = st.file_uploader(
                    "Batch entry prediction. Ensure that isomeric information is provided in the "
                    "SMILES to get the best possible performance. Upload list of SMILES:",
                    type=["csv", "txt", "xlsx"],
                    help='''
                    .txt: file should be tab delimited\n
                    .csv: file should be comma delimited\n
                    .xlsx: file should be in excel format
                    ''',
                )
                if file_up:
                    if file_up.name.endswith('csv'):
                        st.session_state.trident2_current_batch = pd.read_csv(file_up, sep=',', names=['SMILES'])
                    elif file_up.name.endswith('txt'):
                        st.session_state.trident2_current_batch = pd.read_csv(file_up, sep='\t', names=['SMILES'])
                    elif file_up.name.endswith('xlsx'):
                        st.session_state.trident2_current_batch = pd.read_excel(file_up, header=None, names=['SMILES'])

            with subcol2:
                if st.button('Generate example', key='trident2_generate_example_batch'):
                    st.session_state.trident2_current_batch = pd.DataFrame(example_mols, columns=['SMILES'])

            data = st.session_state.trident2_current_batch

            if not data.empty:
                st.markdown('**Showing first 5 rows:**\n')
                st.write(data.head())
                mols = [Chem.MolFromSmiles(smiles) for smiles in data.iloc[:6].SMILES.tolist()]
                try:
                    img = Draw.MolsToGridImage(mols, legends=(data.iloc[:6].SMILES.tolist()), drawOptions=_transparent_draw_options, returnPNG=True)
                except Exception:
                    img = None
                st.markdown('''**Showing first 6 structures (generated using RDKit):**\n''')
                if img is not None:
                    st.image(img)
                else:
                    st.error("Not chemically valid", icon=":material/error:")

            with st.container(key="predict-btn-t2-batch"):
                predict_clicked = st.button("Predict", key='trident2_predict_batch')
            if predict_clicked:
                with st.spinner(text='Inference in Progress...'):
                    results = predict_toxicity(
                        smiles_list=data.SMILES.tolist(),
                        taxid=taxid,
                        effect=effect,
                        conc_unit=conc_unit,
                        duration_hours=duration_for_model,
                        administration_route=administration_route,
                        organism_lifestage=organism_lifestage,
                        species_common_name=species_row['common_name'],
                        species_latin_name=species_row['latin_name'],
                        species_group=species_row['species_group'],
                    )

        else:
            subcol1, subcol2 = st.columns([3, 1], vertical_alignment='bottom')
            with subcol1:
                text_input_holder = st.empty()
                single_input_smiles = text_input_holder.text_input(
                    "Single entry prediction. Ensure that isomeric information is provided in the "
                    "SMILES to get the best possible performance. Input SMILES below:",
                    st.session_state.trident2_example_mol,
                )
            with subcol2:
                if st.button('Generate example', key='trident2_generate_example_single'):
                    get_example_mol()
                    single_input_smiles = text_input_holder.text_input(
                        "Single entry prediction. Ensure that isomeric information is provided in the "
                        "SMILES to get the best possible performance. Input SMILES below:",
                        st.session_state.trident2_example_mol,
                    )

            mols = [Chem.MolFromSmiles(single_input_smiles)]
            try:
                img = Draw.MolsToGridImage(mols, legends=([single_input_smiles]), drawOptions=_transparent_draw_options, returnPNG=True)
            except Exception:
                img = None
            st.markdown('''Structure (generated using RDKit):\n''')
            if (img is not None) and ('*' not in single_input_smiles):
                st.image(img)
            else:
                st.error("Not chemically valid", icon=":material/error:")

            with st.container(key="predict-btn-t2-single"):
                predict_clicked = st.button("Predict", key='trident2_predict_single')
            if predict_clicked:
                with st.spinner(text='Inference in Progress...'):
                    results = predict_toxicity(
                        smiles_list=[single_input_smiles],
                        taxid=taxid,
                        effect=effect,
                        conc_unit=conc_unit,
                        duration_hours=duration_for_model,
                        administration_route=administration_route,
                        organism_lifestage=organism_lifestage,
                        species_common_name=species_row['common_name'],
                        species_latin_name=species_row['latin_name'],
                        species_group=species_row['species_group'],
                    )

        if not results.empty:
            with col2:
                st.success('Predicted effect concentration(s):')
                st.write(results.head())

                download_button_str = download_button(
                    results, 'TRIDENT2_prediction_results.csv', 'Download results', pickle_it=False
                )
                st.markdown(download_button_str, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .footer {
            height: 300px;
        }
        </style>
        <div class="footer"></div>
        """,
        unsafe_allow_html=True,
    )
