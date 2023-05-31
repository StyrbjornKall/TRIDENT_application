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
from inference_utils.pytorch_data_utils import check_training_data, check_closest_chemical, check_valid_structure
from inference_utils.plots_for_space import PlotPCA_CLSProjection, PlotUMAP_CLSProjection

example_mols = ['O=P(O)(O)O', 'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'Cc1ccccc1Cl','C=CC(=O)OCC','ClC(Cl)C(Cl)(Cl)Cl','O=C(O)CNCP(=O)(O)O','CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC','CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl']
def get_example_mol():
    st.session_state.example_mol = example_mols[random.randint(0,len(example_mols)-1)]
get_example_mol()
def get_example_batch():
    st.session_state.example_batch = pd.DataFrame(example_mols, columns=['SMILES'])
def delete_example_batch():
    st.session_state.example_batch = pd.DataFrame()

effectordering = {
            'EC50_algae': {'POP':'POP'},
            'EC10_algae': {'POP':'POP'},
            'EC50EC10_algae': {'POP':'POP'}, 
            'EC50_invertebrates': {'MOR':'MOR','ITX':'ITX'},
            'EC10_invertebrates': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP'} ,
            'EC50EC10_invertebrates': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP'} ,
            'EC50_fish': {'MOR':'MOR'},
            'EC10_fish': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP','GRO': 'GRO'} ,
            'EC50EC10_fish': {'MOR':'MOR','DVP':'DVP','ITX':'ITX', 'REP': 'REP', 'MPH': 'MPH', 'POP': 'POP','GRO': 'GRO'} 
            }

endpointordering = {
            'EC50_algae': {'EC50':'EC50'},
            'EC10_algae': {'EC10':'EC10'},
            'EC50EC10_algae': {'EC50':'EC50', 'EC10': 'EC10'}, 
            'EC50_invertebrates': {'EC50':'EC50'},
            'EC10_invertebrates': {'EC10':'EC10'},
            'EC50EC10_invertebrates': {'EC50':'EC50', 'EC10': 'EC10'},
            'EC50_fish': {'EC50':'EC50'},
            'EC10_fish': {'EC10':'EC10'},
            'EC50EC10_fish': {'EC50':'EC50', 'EC10': 'EC10'} 
            }

def print_predict_page():
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = pd.DataFrame()
    if 'example_mol' not in st.session_state:
        st.session_state.example_mol = 'C1=CC=CC=C1'
    if 'prediction_button' in st.session_state and st.session_state.prediction_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
    if 'batch' in st.session_state and st.session_state.batch == True:
        st.session_state.batch_input = True
    else:
        st.session_state.batch_input = False
    data = pd.DataFrame()

    col1, col2 = st.columns([1,3])
    col1.markdown('## Prediction metrics')
    col1.checkbox("Batch upload (.csv, .txt, .xlsx)", key="batch")
    species_group = {'fish': 'fish', 'aquatic invertebrates': 'invertebrates', 'algae': 'algae'}
    model_type = {'Combined model (best performance)': 'EC50EC10', 'EC50 model': 'EC50','EC10 model': 'EC10'}
    
    PREDICTION_SPECIES = species_group[col1.radio("Select Species group", tuple(species_group.keys()), on_change=None, help="Don't know which to use? \n Check the `Species groups` section under `Documentation`")]
    MODELTYPE = model_type[col1.radio("Select Model type", tuple(model_type), on_change=None, help="Don't know which to use?\n Check the `Models` section under `Documentation`")]
    endpoints = endpointordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
    effects = effectordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
    PREDICTION_ENDPOINT = endpoints[col1.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None, help="Don't know which to use?\n Check the `Endpoints` section under `Documentation`")]
    PREDICTION_EFFECT = effects[col1.radio("Select Effect ",tuple(effects.keys()), on_change=None, help="Don't know which to use?\n Check the `Effects` section under `Documentation`")]
    
    results = pd.DataFrame()

    with col2:
        st.markdown('# Predict chemical ecotoxicity')
        if st.session_state.batch:
            subcol1, subcol2 = st.columns([3,1])
            print(1, st.session_state.current_batch.empty)      
            with subcol1:
                file_up = st.file_uploader("Batch entry prediction. Upload list of SMILES:", type=["csv", 'txt','xlsx'], help='''
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
                    
                
                #if not st.session_state.current_batch.empty:
                #    print(2, st.session_state.current_batch.empty)   
                #    data=st.session_state.current_batch
                #    st.session_state.current_batch = pd.DataFrame()
                

            with subcol2:
                st.markdown('<pre><div style="padding: 26px;"> </div></pre>', unsafe_allow_html=True) 
                if st.button('Generate example'):#, on_click=get_example_batch())
                    st.session_state.current_batch = pd.DataFrame(example_mols, columns=['SMILES'])

            data = st.session_state.current_batch

            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)
            
            if not data.empty:
                st.markdown('**Showing first 5 rows:**\n')
                st.write(data.head())

            if st.button("Predict"):
                with st.spinner(text = 'Inference in Progress...'):
                    
                    TRIDENT = TRIDENT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}', device='cpu')
                    TRIDENT.load_fine_tuned_model()
                
                    results = TRIDENT.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=True)
                    
                mols = [Chem.MolFromSmiles(smiles) for smiles in results.iloc[:6].SMILES.unique().tolist()]
                try:
                    img = Draw.MolsToGridImage(mols,legends=(results.iloc[:6].SMILES.unique().tolist()))
                except:
                    img = None
                st.markdown('''**Showing first 6 structures (generated using RDKit):**\n''')
                if img is not None:
                    st.image(img)
                else:
                    st.markdown('⚠️ **Not chemically valid**')
                    

        elif ~st.session_state.batch:
            subcol1, subcol2 = st.columns([3,1])        
            with subcol1:
                st.text_input(
                "Single entry prediction. Input SMILES below:",
                st.session_state.example_mol,
                key="smile",
                )
            with subcol2:
                st.markdown('<pre><div style="padding: 16px;"> </div></pre>', unsafe_allow_html=True) 
                st.button('Generate example', on_click=get_example_mol())
            
            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)

            if st.button("Predict"):
                data = pd.DataFrame()
                data['SMILES'] = [st.session_state.smile]
                
                with st.spinner(text = 'Inference in Progress...'):
                    TRIDENT = TRIDENT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    TRIDENT.load_fine_tuned_model()
                    results = TRIDENT.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=True)
                mols = [Chem.MolFromSmiles(smiles) for smiles in results.SMILES.unique().tolist()]
                try:
                    img = Draw.MolsToGridImage(mols,legends=(results.SMILES.unique().tolist()))
                except:
                    img = None
                st.markdown('''Structure (generated using RDKit):\n''')
                if img is not None:
                    st.image(img)
                else:
                    st.markdown('⚠️ Not chemically valid')
                        

        if results.empty == False:
            with col2:
                results['Alert'] = results.SMILES.apply(lambda x: check_valid_structure(x))
                results = check_training_data(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                results = check_closest_chemical(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                st.success(f'Predicted effect concentration(s):')
                st.write(results.head())

                download_button_str = download_button(results, 'TRIDENT_prediction_results.csv', 'Download results', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

            with col2:
                st.markdown('# Results analysis')
                with st.expander("Expand results analysis"):
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
                    ## Closest chemical in training set
                    To better understand the toxicity prediction, the predicted chemical's closest resemblance in terms of chemical structure can be determined
                    by calculating the cosine similarity of the CLS-embedding for the predicted chemical and all chemicals in the training set.
                    This similarity score is a better way of understanding how the model places the chemical in terms of its toxicity as compared to e.g., fingerprints, since the embedding is derived from the model itself.''')

                    st.write(results[['SMILES','predictions log10(mg/L)','most similar chemical','cosine similarity']].head())

                    # Space location
                    st.markdown('''
                    ## CLS-embedding projection (PCA)
                    ''')

                    plot_results = (results.drop_duplicates(subset=['SMILES_Canonical_RDKit']) if len(results.drop_duplicates(subset=['SMILES_Canonical_RDKit'])) < 50 else results.drop_duplicates(subset=['SMILES_Canonical_RDKit']).iloc[:50])

                    fig = PlotPCA_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=False, inference_df=plot_results)
                    st.plotly_chart(fig, use_container_width=True, theme='streamlit')
                    
                    buffer = io.StringIO()
                    fig.write_html(buffer, include_plotlyjs='cdn')
                    html_bytes = buffer.getvalue().encode()

                    download_button_str = download_button(html_bytes, 'interactive_CLS_projection.html', 'Lagging ➡ Download HTML', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    
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