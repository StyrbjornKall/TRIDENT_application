import streamlit as st
import pandas as pd
import numpy as np
import torch
import tokenizers
import documentation_page
import predict_page
import contact_page
import publications_page
import hydralit_components as hc


st.set_page_config(layout="wide", page_title='fishbAIT - predicting ecotoxicity using AI', page_icon='🐟')

menu_data = [
    {'icon': "far fa-chart-bar", 'label':"Use fishbAIT"},
    {'icon': "far fa-copy", 'label':"Publications"},
    {'icon': "far fa-file-earmark", 'label':"Documentation"},#no tooltip message
    {'icon':"🌌",'label':"Explore Chemical space"},
    {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
    {'icon': "far fa-copy", 'label':"Right End"},
    {'icon': "envelope",'label':"Contact"},
]

over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name=None,
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

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


### predictions #############################
if menu_id=='Use fishbAIT': 
    predict_page.print_predict_page()

if menu_id == 'Documentation':
    documentation_page.print_doc_page()

if menu_id == 'Contact':
    contact_page.print_contact_page()

if menu_id == 'Publications':
    publications.print_publications_page()