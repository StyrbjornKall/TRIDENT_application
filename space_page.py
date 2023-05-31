import streamlit as st
import io
import pandas as pd
import numpy as np
from custom_download_button import download_button
from inference_utils.plots_for_space import PlotPCA_CLSProjection, PlotUMAP_CLSProjection, PlotPaCMAP_CLSProjection


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

def print_space_page():
    col1, col2 = st.columns((1,3))
    with col1:
        st.markdown('## Projection metrics')
        projection = st.selectbox('Projection method', ('PCA','UMAP'))
        species_group = {'fish': 'fish', 'aquatic invertebrates': 'invertebrates', 'algae': 'algae'}
        model_type = {'Combined model (best performance)': 'EC50EC10'}
        
        PREDICTION_SPECIES = species_group[st.radio("Select Species group", tuple(species_group.keys()), on_change=None, help="Don't know which to use? \n Check the `Species groups` section under `Documentation`")]
        MODELTYPE = model_type[st.radio("Select Model type", tuple(model_type), on_change=None, help="Don't know which to use?\n Check the `Models` section under `Documentation`")]
        endpoints = endpointordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
        effects = effectordering[f'{MODELTYPE}_{PREDICTION_SPECIES}']
        PREDICTION_ENDPOINT = endpoints[st.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None, help="Don't know which to use?\n Check the `Endpoints` section under `Documentation`")]
        PREDICTION_EFFECT = effects[st.radio("Select Effect ",tuple(effects.keys()), on_change=None, help="Don't know which to use?\n Check the `Effects` section under `Documentation`")]
        
        PREDICTION_EXTENDED_DATA = st.checkbox('show predictions outside training data')
        if projection == 'UMAP':
            MIN_DISTNACE = st.number_input('min distance')
            N_NEIGHBORS = st.number_input('n neighbors')

        run_prediction = st.button('Predict')
    
    with col2:
        if run_prediction:
            with st.spinner(text = 'Inference in Progress...'):
                if projection == 'PCA':
                    fig = PlotPCA_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=PREDICTION_EXTENDED_DATA, inference_df=None)
                    st.plotly_chart(fig, use_container_width=True, theme='streamlit')
                    
                if projection == 'UMAP':
                    fig = PlotUMAP_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=PREDICTION_EXTENDED_DATA, inference_df=None, n_neighbors=N_NEIGHBORS, min_dist=MIN_DISTNACE)
                    st.plotly_chart(fig, use_container_width=True, theme='streamlit')
                    
                if projection == 'PaCMAP':
                    fig = PlotPaCMAP_CLSProjection(model_type=MODELTYPE, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, species_group=PREDICTION_SPECIES, show_all_predictions=PREDICTION_EXTENDED_DATA, inference_df=None)
                    st.plotly_chart(fig, use_container_width=True, theme='streamlit')

            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            html_bytes = buffer.getvalue().encode()

            download_button_str = download_button(html_bytes, 'interactive_CLS_projection.html', 'Lagging ➡ Download HTML', pickle_it=False)
            st.markdown(download_button_str, unsafe_allow_html=True)
