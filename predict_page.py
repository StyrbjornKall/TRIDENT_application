import streamlit as st
import io
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from inference_utils.ecoCAIT_for_inference import ecoCAIT_for_inference
from inference_utils.pytorch_data_utils import check_training_data, check_closest_chemical, check_valid_structure
from inference_utils.plots_for_space import PlotPCA_CLSProjection, PlotUMAP_CLSProjection

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
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown('## Prediction metrics')
        input_type = st.checkbox("Batch upload (.csv, .txt, .xlsx)", key="batch")
        species_group = {'fish': 'fish', 'aquatic invertebrates': 'invertebrates', 'algae': 'algae'}
        model_type = {'Combined model (best performance)': 'EC50EC10', 'EC50 model': 'EC50','EC10 model': 'EC10'}
        
        PREDICTION_SPECIES = species_group[st.radio("Select Species group", tuple(species_group.keys()), on_change=None, help="Don't know which to use? \n Check the `Species groups` section under `Documentation`")]
        MODELTYPE = model_type[st.radio("Select Model type", tuple(model_type), on_change=None, help="Don't know which to use?\n Check the `Models` section under `Documentation`")]
        endpoints = endpointordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
        effects = effectordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
        PREDICTION_ENDPOINT = endpoints[st.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None, help="Don't know which to use?\n Check the `Endpoints` section under `Documentation`")]
        PREDICTION_EFFECT = effects[st.radio("Select Effect ",tuple(effects.keys()), on_change=None, help="Don't know which to use?\n Check the `Effects` section under `Documentation`")]
        
        results = pd.DataFrame()

    with col2:
        st.markdown('# Make prediction')
        if st.session_state.batch:
            file_up = st.file_uploader("Batch entry prediction. Upload list of SMILES:", type=["csv", 'txt','xlsx'], help='''
            .txt: file should be tab delimited\n
            .csv: file should be comma delimited\n
            .xlsx: file should be in excel format
            ''')

            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)

            if file_up:
                if file_up.name.endswith('csv'):
                    data=pd.read_csv(file_up, sep=',') #Read our data dataset
                elif file_up.name.endswith('txt'):
                    data=pd.read_csv(file_up, sep='\t', names=['SMILES']) #Read our data dataset
                elif file_up.name.endswith('xlsx'):
                    data=pd.read_excel(file_up)
                st.write(data.head())

            if st.button("Predict"):
                with st.spinner(text = 'Inference in Progress...'):
                    
                    placeholder = st.empty()
                    placeholder.write(
                        '<img width=100 src="http://static.skaip.org/img/emoticons/180x180/f6fcff/fish.gif" style="margin-left: 5px; brightness(1.1);">',
                        unsafe_allow_html=True,
                            )
                    
                    ecocait = ecoCAIT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    ecocait.load_fine_tuned_model()
                    
                    results = ecocait.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=True)
                    
                    placeholder.empty()
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
                    

        elif ~st.session_state.batch:        
            st.text_input(
            "Single entry prediction. Input SMILES below:",
            "C1=CC=CC=C1",
            key="smile",
            )
            
            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=24, max_value=720, step=24)

            if st.button("Predict"):
                data = pd.DataFrame()
                data['SMILES'] = [st.session_state.smile]
                
                with st.spinner(text = 'Inference in Progress...'):
                    ecocait = ecoCAIT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    ecocait.load_fine_tuned_model()
                    
                    results = ecocait.predict_toxicity(
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
                st.success(f'Predicted effect concentration(s):')
                st.write(results.head())

                csv = results.to_csv().encode('utf-8')

                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='ecoCAIT_prediction_results.csv',
                    mime='text/csv',
                    on_click=None
                )

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
                    
                    results = check_training_data(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                    st.write(results.head())

                    # Closest chemical in training set
                    st.markdown('''
                    ## Closest chemical in training set
                    To better understand the toxicity prediction, the predicted chemical's closest resemblence in terms of chemical structure can be determined
                    by calculating the cosine similarity of the CLS-embedding for the predicted chemical and all chemicals in the training set.
                    This similarity score is a better way of understanding how the model places the chemical in terms of its toxicity as compared to e.g., fingerprints, since the embedding is derived from the model itself.''')

                    results = check_closest_chemical(results, MODELTYPE, PREDICTION_SPECIES, PREDICTION_ENDPOINT, PREDICTION_EFFECT)

                    # Add molecules to df
                    PandasTools.AddMoleculeColumnToFrame(results, smilesCol='SMILES', molColName='Molecule')
                    PandasTools.AddMoleculeColumnToFrame(results, smilesCol='most similar chemical', molColName='most similar chemical rendered')

                    st.write(results.head())

                    # Download results
                    st.download_button(
                        label="Download results as CSV",
                        data=results.to_csv().encode('utf-8'),
                        file_name='ecoCAIT_prediction_results.csv',
                        mime='text/csv',
                        on_click=None
                    )

                    # Space location
                    st.markdown('''
                    ## CLS-embedding projection (PCA)
                    ''')

                    fig = PlotPCA_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=False, inference_df=results)
                    st.plotly_chart(fig, use_container_width=True, theme='streamlit')
                    
                    buffer = io.StringIO()
                    fig.write_html(buffer, include_plotlyjs='cdn')
                    html_bytes = buffer.getvalue().encode()

                    st.download_button(
                        label='Lagging? --> Download HTML',
                        data=html_bytes,
                        file_name='interactive_CLS_projection.html',
                        mime='text/html'
                    )

