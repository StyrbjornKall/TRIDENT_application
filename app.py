import streamlit as st
import pandas as pd
import numpy as np
import torch
import tokenizers
from inference_utils.fishbAIT_for_inference import fishbAIT


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)


#st.set_page_config(layout="wide")
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

# APP
st.sidebar.image("logo.jpg", use_column_width=True, caption='logo - Generated through DALL-E')
app_mode = st.sidebar.selectbox('Select Page',['Predict','Documentation'])
input_type = st.sidebar.checkbox("Batch (.csv, .txt, .xlsx) input", key="batch")
st.title('''fishbAIT''')
st.markdown('A deep learning software that lets you predict chemical ecotoxicity to fish')
endpoints = {'EC50': 'EC50', 'EC10': 'EC10'}
effects = {'MOR': 'MOR', 'DVP': 'DVP', 'GRO': 'GRO','POP': 'POP','MPH':'MPH'}
model_type = {'EC50': 'EC50','Chronic': 'EC10','Combined model': 'EC50EC10'}

PREDICTION_ENDPOINT = endpoints[st.sidebar.radio("Select Endpoint ",tuple(endpoints.keys()))]
PREDICTION_EFFECT = effects[st.sidebar.radio("Select Effect ",tuple(effects.keys()))]
MODELTYPE = model_type[st.sidebar.radio("Select Model type", tuple(model_type))]

col1, col2 = st.columns(2)

with col1:
    if app_mode=='Predict': 
        if st.session_state.batch:
            file_up = st.file_uploader("Upload data containing SMILES", type=["csv", 'txt','xlsx'], help='''
            .txt: file should be tab delimited\n
            .csv: file should be comma delimited\n
            .xlsx: file should be in excel format
            ''')
            if file_up:
                if file_up.name.endswith('csv'):
                    data=pd.read_csv(file_up, sep=',') #Read our data dataset
                elif file_up.name.endswith('txt'):
                    data=pd.read_csv(file_up, sep='\t') #Read our data dataset
                elif file_up.name.endswith('xlsx'):
                    data=pd.read_excel(file_up)

            EXPOSURE_DURATION = st.slider(
                'Select exposure duration (e.g. 96 h)',
                min_value=0, max_value=300, step=2)


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

                    st.success(f'Predicted effect concentration(s):')
                    st.write(results.head())

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

                    st.success(f'Predicted effect concentration(s):')
                    st.write(results.head())



st.markdown("""
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)