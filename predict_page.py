import streamlit as st
import numpy as np
import pandas as pd
import torch
import tokenizers
from inference_utils.fishbAIT_for_inference import fishbAIT

@st.cache(hash_funcs={
    tokenizers.Tokenizer: lambda _: None, 
    tokenizers.AddedToken: lambda _: None})
def loadmodel(version):
    fishbait = fishbAIT(model_version=version)
    fishbait.load_fine_tuned_model()
    return fishbait

def loadtokenizer(version):
    tokenizer = AutoTokenizer.from_pretrained(f'StyrbjornKall/fishbAIT_{version}')
    return tokenizer


def print_predict_page():
    col1, col2, col3 = st.columns([1,2,2])
    with col1:
        st.markdown('## Prediction metrics')
        input_type = st.checkbox("Batch upload (.csv, .txt, .xlsx)", key="batch")
        endpoints = {'EC50': 'EC50', 'EC10': 'EC10'}
        effects = {'MOR': 'MOR', 'DVP': 'DVP', 'GRO': 'GRO','POP': 'POP','MPH':'MPH'}
        model_type = {'EC50': 'EC50','Chronic': 'EC10','Combined model': 'EC50EC10'}
        PREDICTION_ENDPOINT = endpoints[st.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None)]
        PREDICTION_EFFECT = effects[st.radio("Select Effect ",tuple(effects.keys()))]
        MODELTYPE = model_type[st.radio("Select Model type", tuple(model_type))]
        results = pd.DataFrame()

    with col2:
        if st.session_state.batch:
            file_up = st.file_uploader("Upload data containing SMILES", type=["csv", 'txt','xlsx'], help='''
            .txt: file should be tab delimited\n
            .csv: file should be comma delimited\n
            .xlsx: file should be in excel format
            ''')

            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=0, max_value=300, step=2)

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
                    
                    fishbait = loadmodel(version=MODELTYPE)
                    
                    results = fishbait.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT)
                    
                    placeholder.empty()
                    

        elif ~st.session_state.batch:        
            st.text_input(
            "Input SMILES 👇",
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
                    fishbait = loadmodel(version=MODELTYPE)
                    
                    results = fishbait.predict_toxicity(
                        SMILES = data.SMILES.tolist(), 
                        exposure_duration=EXPOSURE_DURATION, 
                        endpoint=PREDICTION_ENDPOINT, 
                        effect=PREDICTION_EFFECT)

                        

        if results.empty == False:
            with col3:
                st.markdown('#');st.markdown('#');st.markdown('#');
                st.success(f'Predicted effect concentration(s):')
                st.write(results.head())