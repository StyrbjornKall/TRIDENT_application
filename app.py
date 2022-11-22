import streamlit as st
import pandas as pd
import numpy as np
import documentation_page
import predict_page
import contact_page
import publications_page
import space_page
import hydralit_components as hc


st.set_page_config(layout="wide", page_title='fishbAIT - predicting ecotoxicity using AI', page_icon='🐟')

menu_data = [
    {'icon': "far fa-chart-bar", 'label':"Use fishbAIT"},
    {'icon': "far fa-copy", 'label':'Publication'},
    {'icon': "far fa-file-earmark", 'label':"Documentation"},#no tooltip message
    {'icon':"🌌",'label':"Explore the Chemical Space"},
    {'icon': "envelope",'label':"Contact"},
]

over_theme = over_theme = {'txc_inactive': 'black','menu_background':'white', 'txc_active':'tomato'}
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



### predictions #############################
if menu_id == 'Home':
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.devblog.no/sites/devblog.no/files/styles/800px_width/public/field/image/network.jpg?itok=yaHdzKIc");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
    st.markdown('''
    <h1 style="font-family:Helvetica;color:black;font-size:600%;text-align:center;">
    fishbAIT
    </h1>
    <h1 style="font-family:Helvetica;color:tomato;font-size:300%;text-align:center;">
    USING DEEP LEARNING TO PREDICT CHEMICAL ECOTOXICITY
    </h1>
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns((1,1,1))
    with col1:
        
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .5); padding:10px;margin-top:20px;">
        <span style="text-align:center;"><h1>
        BETTER  
        </h1></span>
        <p style="font-size:200%;">
        fishbAIT is a better software
        </p>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .5); padding:10px;margin-top:20px;">
        <span style="text-align:center;"><h1>
        FASTER  
        </h1></span>
        <p style="font-size:200%;">
        fishbAIT is a faster software
        </p>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .5); padding:10px;margin-top:20px;">
        <span style="text-align:center;"><h1>
        STRONGER  
        </h1></span>
        <p style="font-size:200%;">
        fishbAIT is a stronger software
        </p>
        </div>''', unsafe_allow_html=True)




if menu_id=='Use fishbAIT': 
    predict_page.print_predict_page()

if menu_id == 'Documentation':
    documentation_page.print_doc_page()

if menu_id == 'Contact':
    contact_page.print_contact_page()

if menu_id == 'Publication':
    publications_page.print_publications_page()

if menu_id == 'Explore the Chemical Space':
    space_page.print_space_page()