import streamlit as st
import numpy as np
import pandas as pd
import torch
import tokenizers
from inference_utils.fishbAIT_for_inference import fishbAIT_for_inference
from inference_utils.pytorch_data_utils import check_training_data

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

@st.cache(hash_funcs={
    tokenizers.Tokenizer: lambda _: None, 
    tokenizers.AddedToken: lambda _: None})
def loadmodel(version):
    fishbait = fishbAIT_(model_version=version)
    fishbait.load_fine_tuned_model()
    return fishbait

def loadtokenizer(version):
    tokenizer = AutoTokenizer.from_pretrained(f'StyrbjornKall/fishbAIT_{version}')
    return tokenizer


def print_predict_page():
    col1, col2, col3 = st.columns([2,3,3])
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
                min_value=1, max_value=300, step=2)

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
                    
                    fishbait = fishbAIT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    fishbait.load_fine_tuned_model()
                    
                    results = fishbait.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=False)
                    
                    placeholder.empty()
                    

        elif ~st.session_state.batch:        
            st.text_input(
            "Single entry prediction. Input SMILES below:",
            "C1=CC=CC=C1",
            key="smile",
            )
            
            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=0, max_value=300, step=2)

            if st.button("Predict"):
                data = pd.DataFrame()
                data['SMILES'] = [st.session_state.smile]
                
                with st.spinner(text = 'Inference in Progress...'):
                    fishbait = fishbAIT_for_inference(model_version=f'{MODELTYPE}_{PREDICTION_SPECIES}')
                    fishbait.load_fine_tuned_model()
                    
                    results = fishbait.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT,
                        return_cls_embeddings=False)

                        

        if results.empty == False:
            with col2:
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

            with col3:
                st.markdown('# Results analysis')
                with st.expander("Expand results analysis"):
                    data['SMILES inside training data'] = data.SMILES.apply(lambda x: check_training_data(x, f'{MODELTYPE}_{PREDICTION_SPECIES}'))
                    st.write(data.head())

