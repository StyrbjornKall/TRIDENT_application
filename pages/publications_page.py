import streamlit as st

def print_publications_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Publications')

        st.markdown('''
        This application is associated with the publication "Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms". For citation of the presented models, or this application, please cite the referenced publications.

        | **Publication**                                                                                                | **Journal**      | **Doi**                                   |
        |----------------------------------------------------------------------------------------------------------------|------------------|-------------------------------------------|
        | (Preprint) Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms | bioRxiv          | https://doi.org/10.1101/2023.04.17.537138 |
        | Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms            | Science Advances | https://doi.org/10.1126/sciadv.adk6669    |

        ''')