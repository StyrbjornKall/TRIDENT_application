import streamlit as st
import io
import numpy as np
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from custom_download_button import download_button
from inference_utils.TRIDENT_for_inference import TRIDENT_for_inference
from inference_utils.pytorch_data_utils import check_training_data, check_closest_chemical, check_valid_smiles, check_valid_chemistry
from inference_utils.plots_for_space import PlotPCA_CLSProjection, PlotUMAP_CLSProjection

example_mols = ['O=P(O)(O)O', 'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'Cc1ccccc1Cl','C=CC(=O)OCC','ClC(Cl)C(Cl)(Cl)Cl','O=C(O)CNCP(=O)(O)O','CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC','CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl']
def get_example_mol():
    current_example_mol = st.session_state.example_mol
    while True:
        next_example_mol = example_mols[random.randint(0,len(example_mols)-1)]
        if current_example_mol != next_example_mol:
            break
    st.session_state.example_mol = next_example_mol

# effectordering = {
#             'EC50_algae': {'POP':'POP'},
#             'EC10_algae': {'POP':'POP'},
#             'EC50EC10_algae': {'POP':'POP'}, 
#             'EC50_invertebrates': {'MOR':'MOR','ITX':'ITX'},
#             'EC10_invertebrates': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP'} ,
#             'EC50EC10_invertebrates': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP'} ,
#             'EC50_fish': {'MOR':'MOR'},
#             'EC10_fish': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP','GRO': 'GRO'} ,
#             'EC50EC10_fish': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP','GRO': 'GRO'} 
#             }

# endpointordering = {
#             'EC50_algae': {'EC50':'EC50'},
#             'EC10_algae': {'EC10':'EC10'},
#             'EC50EC10_algae': {'EC50':'EC50', 'EC10': 'EC10'}, 
#             'EC50_invertebrates': {'EC50':'EC50'},
#             'EC10_invertebrates': {'EC10':'EC10'},
#             'EC50EC10_invertebrates': {'EC50':'EC50', 'EC10': 'EC10'},
#             'EC50_fish': {'EC50':'EC50'},
#             'EC10_fish': {'EC10':'EC10'},
#             'EC50EC10_fish': {'EC50':'EC50', 'EC10': 'EC10'} 
#             }


endpointordering = {
            'EC50EC10_algae': {'EC50':'EC50', 'EC10': 'EC10'}, 
            'EC50EC10_invertebrates': {'EC50':'EC50', 'EC10': 'EC10'},
            'EC50EC10_fish': {'EC50':'EC50', 'EC10': 'EC10'} 
            }
effectordering = {
            'EC50EC10_algae': 
            {'EC50': {'POP':'POP'},
             'EC10': {'POP': 'POP'}
            },
            'EC50EC10_invertebrates':
            {'EC50': {'MOR':'MOR','ITX':'ITX', 'POP': 'POP'},
             'EC10': {'MOR':'MOR','ITX':'ITX', 'REP': 'REP', 'POP': 'POP'}
            },
            'EC50EC10_fish':
            {'EC50': {'MOR':'MOR'},
             'EC10': {'MOR':'MOR','GRO': 'GRO'}
            }
            }

def print_predict_page():
    if 'example_mol' not in st.session_state:
        st.session_state.example_mol = 'C1=CC=CC=C1'
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = pd.DataFrame()

    # Page begins here    
    data = pd.DataFrame()

    col1, col2 = st.columns([1,3])
    col1.markdown('## Prediction metrics')
    col1.checkbox("Predict all", key="predict_all", on_change=None, value=False, help="Currently is only supported in batch mode.")
    col1.checkbox('Return CLS embeddings', key='return_cls_embeddings', on_change=None, value=False, help="The model represents the chemical as a high dimensional vector (CLS embedding). This can be useful for clustering or further analysis.")

    # Disable metric selection if "Predict all" is checked
    predict_all_toggled = st.session_state.get("predict_all", False)

    # If Predict all is toggled, force batch upload to be True and disabled
    if predict_all_toggled:
        st.session_state.batch = True

    col1.checkbox("Batch upload (.csv, .txt, .xlsx)", key="batch", disabled=predict_all_toggled)
    
    species_group = {'fish': 'fish', 'aquatic invertebrates': 'invertebrates', 'algae': 'algae'}
    model_type = {'Combined model (best performance)': 'EC50EC10'}
    
    PREDICTION_SPECIES = species_group[col1.radio("Select Species group", tuple(species_group.keys()), on_change=None, disabled=predict_all_toggled, help="Don't know which to use? \n Check the `Species groups` section under `Documentation`")]
    MODELTYPE = model_type[col1.radio("Select Model type", tuple(model_type), on_change=None, disabled=predict_all_toggled, help="Don't know which to use?\n Check the `Models` section under `Documentation`")]
    endpoints = endpointordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
    PREDICTION_ENDPOINT = endpoints[col1.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None, disabled=predict_all_toggled, help="Don't know which to use?\n Check the `Endpoints` section under `Documentation`")]
    effects = effectordering[f'{MODELTYPE}_{PREDICTION_SPECIES}'][PREDICTION_ENDPOINT]
    PREDICTION_EFFECT = effects[col1.radio("Select Effect ",tuple(effects.keys()), on_change=None, disabled=predict_all_toggled, help="Don't know which to use?\n Check the `Effects` section under `Documentation`")]
    
    results = pd.DataFrame()

    with col2:
        st.markdown('# Predict chemical ecotoxicity')
        if st.session_state.batch:
            subcol1, subcol2 = st.columns([3,1])
            with subcol1:
                file_up = st.file_uploader("Batch entry prediction. Ensure that isomeric information is provided in the SMILES to get the best possible performance. Upload list of SMILES:", type=["csv", 'txt','xlsx'], help='''
                    .txt: file should be tab delimited\n
                    .csv: file should be comma delimited\n
                    .xlsx: file should be in excel format
                    ''')

                if file_up:
                    if file_up.name.endswith('csv'):
                        st.session_state.current_batch=pd.read_csv(file_up, sep=',', names=['SMILES']) #Read our data dataset
                    elif file_up.name.endswith('txt'):
                        st.session_state.current_batch=pd.read_csv(file_up, sep='\t', names=['SMILES']) #Read our data dataset
                    elif file_up.name.endswith('xlsx'):
                        st.session_state.current_batch=pd.read_excel(file_up, header=None, names=['SMILES']) 
                    
            with subcol2:
                st.markdown('<pre><div style="padding: 36px;"> </div></pre>', unsafe_allow_html=True) 
                if st.button('Generate example'):
                    st.session_state.current_batch = pd.DataFrame(example_mols, columns=['SMILES'])

            data = st.session_state.current_batch

            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)
            
            if not data.empty:
                st.markdown('**Showing first 5 rows:**\n')
                st.write(data.head())
                mols = [Chem.MolFromSmiles(smiles) for smiles in data.iloc[:6].SMILES.tolist()]
                try:
                    img = Draw.MolsToGridImage(mols, legends=(data.iloc[:6].SMILES.tolist()))
                except:
                    img = None
                st.markdown('''**Showing first 6 structures (generated using RDKit):**\n''')
                if img is not None:
                    st.image(img)
                else:
                    st.markdown('⚠️ **Not chemically valid**')

            if st.button("Predict"):
                with st.spinner(text = 'Inference in Progress...'):

                    if predict_all_toggled:
                        # Write warning that notifies the user that this may take a while
                        st.warning("Predicting all combinations of model, species group, endpoint, and effect. This may take a while depending on the number of SMILES provided and the number of combinations. Please be patient...")

                        # Predict all logic
                        all_results = []

                        # Initialize progress bar
                        progress = st.progress(0, text="Processing species groups...")
                        for species_idx, PREDICTION_SPECIES in enumerate(species_group.values()):
                            
                            TRIDENT = TRIDENT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}', device='cpu')
                            TRIDENT.load_fine_tuned_model()
                            endpoints = endpointordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']

                            for endpoint_idx, PREDICTION_ENDPOINT in enumerate(endpoints.values()):
                                effects = effectordering[f'{MODELTYPE}_{PREDICTION_SPECIES}'][PREDICTION_ENDPOINT]

                                for PREDICTION_EFFECT in effects.keys():
                                    total_steps = len(species_group) * len(endpoints) * len(effects)
                                    current_step = species_idx * len(endpoints) * len(effects) + endpoint_idx * len(effects) + list(effects.keys()).index(PREDICTION_EFFECT) + 1
                                    progress.progress(current_step / total_steps, text=f"Processing: {PREDICTION_SPECIES} | {PREDICTION_ENDPOINT}")
                                    results = TRIDENT.predict_toxicity(
                                        SMILES=data.SMILES.tolist(),
                                        exposure_duration=EXPOSURE_DURATION,
                                        endpoint=PREDICTION_ENDPOINT,
                                        effect=PREDICTION_EFFECT,
                                        return_cls_embeddings=True
                                    )
                                    results['Species group'] = PREDICTION_SPECIES
                                    results['Endpoint'] = PREDICTION_ENDPOINT
                                    results['Effect'] = PREDICTION_EFFECT
                                    results['SMILES Alert'] = results.SMILES.apply(lambda x: check_valid_smiles(x))
                                    results['Chemical Alert'] = results.SMILES.apply(lambda x: check_valid_chemistry(x))
                                    results = check_training_data(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                                    results = check_closest_chemical(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                                    results.loc[(results['SMILES Alert']=='SMILES not valid'), ['SMILES_Canonical_RDKit', 'predictions log10(mg/L)', 'predictions (mg/L)', 'CLS_embeddings', 'most similar chemical', 'max cosine similarity', 'mean cosine similarity']] = None

                                    all_results.append(results)

                        # Concatenate all results
                        results = pd.concat(all_results, ignore_index=True, axis=0)
                        progress.progress(total_steps/total_steps, text=f"Processing: Done!")
                    
                    else:
                        TRIDENT = TRIDENT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}', device='cpu')
                        TRIDENT.load_fine_tuned_model()
                    
                        results = TRIDENT.predict_toxicity(
                            SMILES = data.SMILES.tolist(), 
                            exposure_duration=EXPOSURE_DURATION, 
                            endpoint=PREDICTION_ENDPOINT, 
                            effect=PREDICTION_EFFECT,
                            return_cls_embeddings=True
                            )
                            
                        results['SMILES Alert'] = results.SMILES.apply(lambda x: check_valid_smiles(x))
                        results['Chemical Alert'] = results.SMILES.apply(lambda x: check_valid_chemistry(x))
                        results = check_training_data(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                        results = check_closest_chemical(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                        results.loc[(results['SMILES Alert']=='SMILES not valid'), ['SMILES_Canonical_RDKit', 'predictions log10(mg/L)', 'predictions (mg/L)', 'CLS_embeddings', 'most similar chemical', 'max cosine similarity', 'mean cosine similarity']] = None

        elif ~st.session_state.batch:
            subcol1, subcol2 = st.columns([3,1])        
            with subcol1:
                text_input_holder = st.empty()
                single_input_smiles = text_input_holder.text_input(
                "Single entry prediction. Ensure that isomeric information is provided in the SMILES to get the best possible performance. Input SMILES below:",
                st.session_state.example_mol,
                )
            with subcol2:
                st.markdown('<pre><div style="padding: 27px;"> </div></pre>', unsafe_allow_html=True)
                if st.button('Generate example'):
                    get_example_mol()
                    single_input_smiles = text_input_holder.text_input(
                    "Single entry prediction. Ensure that isomeric information is provided in the SMILES to get the best possible performance. Input SMILES below:",
                    st.session_state.example_mol
                    )
            
            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)

            mols = [Chem.MolFromSmiles(single_input_smiles)]
            try:
                img = Draw.MolsToGridImage(mols,legends=([single_input_smiles]))
            except:
                img = None
            st.markdown('''Structure (generated using RDKit):\n''')
            if (img is not None) and ('*' not in single_input_smiles):
                st.image(img)
            else:
                st.markdown('⚠️ Not chemically valid')

            if st.button("Predict"):
                data = pd.DataFrame()
                data['SMILES'] = [single_input_smiles]
                
                with st.spinner(text = 'Inference in Progress...'):
                    TRIDENT = TRIDENT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    TRIDENT.load_fine_tuned_model()
                    results = TRIDENT.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=True
                        )
                    
                results['SMILES Alert'] = results.SMILES.apply(lambda x: check_valid_smiles(x))
                results['Chemical Alert'] = results.SMILES.apply(lambda x: check_valid_chemistry(x))
                results = check_training_data(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                results = check_closest_chemical(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                results.loc[(results['SMILES Alert']=='SMILES not valid'), ['SMILES_Canonical_RDKit', 'predictions log10(mg/L)', 'predictions (mg/L)', 'CLS_embeddings', 'most similar chemical', 'max cosine similarity', 'mean cosine similarity']] = None
                        
        if results.empty == False:
            with col2:                
                st.success(f'Predicted effect concentration(s):')
                st.write(results.drop(columns=['CLS_embeddings']).head())

                if st.session_state.return_cls_embeddings:
                    download_button_str = download_button(results, 'TRIDENT_prediction_results.csv', 'Download results', pickle_it=False)
                else:
                    download_button_str = download_button(results.drop(columns=['CLS_embeddings']), 'TRIDENT_prediction_results.csv', 'Download results', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

            # Only show Results analysis if Predict all is NOT checked
            if not predict_all_toggled:
                with col2:
                    st.markdown('# Results analysis')
                    with st.expander("Expand results analysis"):
                        st.markdown('''
                        ## Chemical alerts
                        If RDKit asserts any SMILES with an error feedback is provided as either an "SMILES Alert" or an "Chemical Alert". Most often the errors are SMILES parsing errors ("SMILES Alerts") or valence errors ("Chemical Alerts"). In some cases, RDKit cannot handle the provided SMILES but the structure is still valid when for example run through PubChem. In those cases, the recommendation is to first run the SMILES through e.g. PubChem and retrieve a canonical SMILES from there. 
                        For example, the `|` character always produce parsing errors, but the structure is still valid when checked in PubChem. `*`-symbols are also set as invalid since no polymers were included in the training.

                        To ensure adequate predictions, predictions for SMILES with the "SMILES Alert" flag are not provided. 
                        ''')

                        st.write(results[['SMILES','predictions log10(mg/L)','SMILES Alert', 'Chemical Alert']].head())

                        st.markdown('''
                        ## Training data alerts
                        If the chemical is inside the training data of the model, a 1 is present in the respective training column. A chemical 
                        can be inside the training data in two ways.
                        1. As an **endpoint-match**, i.e. when the chosen model was developed for this species group, experimental data for this chemical was present for the chosen endpoint.
                        2. As an exact **effect-match**, i.e. when the chosen model was developed for this combination of species and endpoint, experimental data for the chosen effect was present.
                        
                        A match is denoted **1**.
                        
                        Note this does not include exact exposure duration matches since most of the trainable parameters are found in the transformer architecture which only uses the SMILES.''')
                        
                        st.write(results[['SMILES','predictions log10(mg/L)','endpoint match', 'effect match']].head())

                        # Closest chemical in training set
                        st.markdown('''
                        ## Chemical similarity to the training set
                        To better understand the toxicity prediction, the predicted chemical's closest resemblance in terms of chemical structure with regards to its toxicity is determined together with the mean similarity to the training dataset. 
                        This is calculated as the cosine similarity of the CLS-embedding for the predicted chemical and all chemicals in the training set. Low similarity usually indicates a weaker prediction. 
                                    
                        High similarity may be interpreted as a `mean cosine similarity` of [1,0.3), intermediate [0.3,0.2) and low similarity [0.2,-1].
                        This score is more reliable way of understanding how the model places the chemical in terms of its toxicity, as compared to e.g., fingerprints, since the embedding is derived from the model itself.''')

                        st.write(results[['SMILES','predictions log10(mg/L)','most similar chemical','max cosine similarity','mean cosine similarity']].head())

                        # Space location
                        st.markdown('''
                        ## CLS-embedding projection (PCA)
                        The CLS-embeddings from the model may be projected onto a 2D plane using PCA to visualize the training data. The predicted chemicals are present as squares. 
                        ''')

                        plot_results = results[results['SMILES Alert'].isna()]
                        plot_results = (plot_results.drop_duplicates(subset=['SMILES_Canonical_RDKit']) if len(plot_results.drop_duplicates(subset=['SMILES_Canonical_RDKit'])) < 50 else plot_results.drop_duplicates(subset=['SMILES_Canonical_RDKit']).iloc[:50])

                        if plot_results.empty == False:
                            fig = PlotPCA_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=False, inference_df=plot_results)
                            st.plotly_chart(fig, use_container_width=True, theme='streamlit')
                            
                            buffer = io.StringIO()
                            fig.write_html(buffer, include_plotlyjs='cdn')
                            html_bytes = buffer.getvalue().encode()

                            download_button_str = download_button(html_bytes, 'interactive_CLS_projection.html', 'Lagging ➡ Download HTML', pickle_it=False)
                            st.markdown(download_button_str, unsafe_allow_html=True)
                        else:
                            st.write('No valid SMILES to plot.')

                    
    # Add padding element at the bottom of the app
        st.markdown(
            """
            <style>
            .footer {
                height: 300px; /* Change this to adjust the height of the padding element */
            }
            </style>
            <div class="footer"></div>
            """,
            unsafe_allow_html=True
        )