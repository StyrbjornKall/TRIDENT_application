import streamlit as st
import base64


def print_home_page():
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
     
    with open("assets/trident-logo.svg", "rb") as image_file:
        image_data = image_file.read()
        img_64 = base64.b64encode(image_data).decode()
    

    st.markdown(f'''
    <div style="text-align:center;">
    <img src="data:image/svg+xml;base64,{img_64}" style="width:1400px;">
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns((2,3))
    with col1:
        with open("assets/imagine.svg", "rb") as image_file:
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
        <span class="highlighted-text">Improved</span> ecotoxicity modelling 
        </h2></span>
        <p style="font-size:130%;">
        TRIDENT-1 is a novel in silico deep learning ecotoxicity model trained using a vast collection of experimental data. TRIDENT-1 performs highly accurate predictions of both acute and chronic effect concentration and outperforms traditional models in both accuracy and applicability. 
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
        TRIDENT-1 can make predictions for several different acute and chronic toxicity endpoints for both fish, aquatic invertebrates and algae. To date, it can predict effect concentrations for mortality, intoxication, growth, population, reproduction, morphology and development for exposure duration ranging between 1-30 days.
        </p>
        </div>''', unsafe_allow_html=True)
    with col2:
        with open("assets/nemo.svg", "rb") as image_file:
            image_data = image_file.read()
            img_64 = base64.b64encode(image_data).decode()
        st.markdown(f'''
        <div style="border-radius: 10px; background-color:rgba(255, 255, 255, .0); padding:10px; padding-top:120px; margin-left:25%;">
        <img src="data:image/svg+xml;base64,{img_64}" style="max-height: 150px;">
        </div>
        ''', unsafe_allow_html=True)

    col1, col2 = st.columns((2,3))
    with col1:
        with open("assets/space.svg", "rb") as image_file:
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
        TRIDENT-1 uses a single continuous chemical space that is built with ecotoxicity in mind to determine the ecotoxicity of a chemical from its chemical structure. Similar to how modern language models associate semantically similar words to each other, TRIDENT-1 uses state-of-the-art deep learning transformers to determine the similarity within the SMILES "language", allowing it to make highly accurate predictions. 
        </p>
        </div>''', unsafe_allow_html=True)
