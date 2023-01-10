import streamlit as st
import pandas as pd
import numpy as np
import documentation_page
import predict_page
import contact_page
import publications_page
import space_page
import hydralit_components as hc
import base64
from PIL import Image

st.set_page_config(layout="wide", page_title='ecoCAIT - predicting ecotoxicity using AI', page_icon="🌍")

#import styles
with open('styles.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


menu_data = [
    {'icon': "far fa-chart-bar", 'label':"Use ecoCAIT"},
    {'icon': "far fa-copy", 'label':'Publication'},
    {'icon': "far fa-file-earmark", 'label':"Documentation"},#no tooltip message
    {'icon':"🌌",'label':"Explore the chemical space"},
    {'icon': "envelope",'label':"Contact"},
]

over_theme = over_theme = {'txc_inactive': 'black','menu_background':'white', 'txc_active':'#F63366'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name=None,
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

footer = '''<style>
footer{visibility:visible;}
footer:after{
    content:'Copyright @ 2022: Styrbjörn Käll';
    display:block;
    position:relative;
    color:tomato;
    padding:5px;
    top:3px;
}
</style>'''
st.markdown(footer, unsafe_allow_html=True)


### APP #############################
with open("background.svg", "rb") as image_file:
        image_data = image_file.read()
        bg_base64 = base64.b64encode(image_data).decode()

if menu_id == 'Home':
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: None;
             background-attachment: fixed;
             background-size: cover;
         }}
        
         </style>
         """,
         unsafe_allow_html=True
     )
    # url("https://www.devblog.no/sites/devblog.no/files/styles/800px_width/public/field/image/network.jpg?itok=yaHdzKIc");
    with open("new-new-logo.svg", "rb") as image_file:
        image_data = image_file.read()
        img_64 = base64.b64encode(image_data).decode()
    

    st.markdown(f'''
    <div style="text-align:center;">
    <img src="data:image/svg+xml;base64,{img_64}" style="width:1200px; filter: brightness(30%);">
    </div>
    ''', unsafe_allow_html=True)

    #<h1 style="font-family:Helvetica;color:black;font-size:600%;">
    #fishbAIT
    #</h1>
    #<h1 style="font-family:Helvetica;color:black;font-size:300%;">
    #USING DEEP LEARNING TO PREDICT CHEMICAL AQUATIC ECOTOXICITY
    #</h1>

    col1, col2 = st.columns((2,3))
    with col1:
        with open("imagine.svg", "rb") as image_file:
            image_data = image_file.read()
            img_64 = base64.b64encode(image_data).decode()
        st.markdown(f'''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; margin-left:40%;padding-top:200px; ">
        <img src="data:image/svg+xml;base64,{img_64}" style="max-height: 150px;">
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px;padding-top:200px; ">
        <span style="text-align:center;"><h2>
        <span class="highlighted-text">Reimagined</span> ecotoxicity modelling 
        </h2></span>
        <p style="font-size:130%;">
        ecoCAIT is a novel <em>in silico</em> deep learning ecotoxicity model trained on a vast collection of experimental data to make highly accurate 
        acute and chronic effect concentration predictions. The model outperforms traditional models in both accuracy and applicability, and is the largest 
        ecotoxicity model to date in terms of model- and training data-complexity.
        </p>
        </div>''', unsafe_allow_html=True)

    col1, col2 = st.columns((3,2))
    with col1:
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; padding-top:120px;">
        <span style="text-align:center;"><h2>
        <span class="highlighted-text">Diverse</span> endpoint prediction  
        </h2></span>
        <p style="font-size:130%;">
        ecoCAIT can make predictions on fish, aquatic invertebrates and algae for various acute and chronic toxicity endpoints.
        To date, it can predict effect concentrations for mortality, intoxication, growth, population, reproduction, morphology and development across any exposure duration.
        </p>
        </div>''', unsafe_allow_html=True)
    with col2:
        with open("nemo.svg", "rb") as image_file:
            image_data = image_file.read()
            img_64 = base64.b64encode(image_data).decode()
        st.markdown(f'''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; padding-top:120px; margin-left:25%;">
        <img src="data:image/svg+xml;base64,{img_64}" style="max-height: 150px;">
        </div>
        ''', unsafe_allow_html=True)

    col1, col2 = st.columns((2,3))
    with col1:
        with open("space.svg", "rb") as image_file:
            image_data = image_file.read()
            img_64 = base64.b64encode(image_data).decode()
        st.markdown(f'''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; padding-top:120px; margin-left:40%;">
        <img src="data:image/svg+xml;base64,{img_64}" style="max-height: 150px;">
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; padding-top:120px; padding-bottom:300px">
        <span style="text-align:center;"><h2>
        One <span class="highlighted-text">continuous</span> space
        </h2></span>
        <p style="font-size:130%;">
        Unlike traditional QSAR-models, ecoCAIT has one continuous chemical space which is built with ecotoxicity in mind. This space is used to determine
        the ecotoxicity of a chemical by its chemical structure. Similar to how other language models associate semantically similar words with each other, ecoCAIT 
        can determine the similarity within the SMILES "language" in order to make highly accurate predictions.
        </p>
        </div>''', unsafe_allow_html=True)

    

## Other pages

if menu_id=='Use ecoCAIT': 
    predict_page.print_predict_page()

if menu_id == 'Documentation':
    documentation_page.print_doc_page()

if menu_id == 'Contact':
    contact_page.print_contact_page()

if menu_id == 'Publication':
    publications_page.print_publications_page()

if menu_id == 'Explore the chemical space':
    space_page.print_space_page()