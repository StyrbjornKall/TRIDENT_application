import streamlit as st

import pandas as pd
import numpy as np
import json
import pickle
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA


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
        projection = st.selectbox('Projection method', ('UMAP','PCA'))
        species_group = {'fish': 'fish', 'aquatic invertebrates': 'invertebrates', 'algae': 'algae'}
        model_type = {'Combined model (best performance)': 'EC50EC10', 'EC50 model': 'EC50','EC10 model': 'EC10'}
        
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
                    st.plotly_chart(PlotPCA_CLSProjection(MODELTYPE, PREDICTION_ENDPOINT, PREDICTION_EFFECT, PREDICTION_SPECIES, PREDICTION_EXTENDED_DATA), use_container_width=True, theme='streamlit')
                if projection == 'UMAP':
                    st.plotly_chart(PlotUMAP_CLSProjection(PREDICTION_ENDPOINT, N_NEIGHBORS, MIN_DISTNACE), use_container_width=True, theme='streamlit')

@st.cache
def PlotPCA_CLSProjection(model_type, endpoint, effect, species_group, show_all_predictions):

    all_preds = pd.read_pickle(f'data/{model_type}_model_predictions/{species_group}_{endpoint}_{effect}_predictions.zip', compression='zip')

    embeddings = np.array(all_preds.CLS_embeddings.tolist())

    pcomp = PCA(n_components=3)
    pcac = pcomp.fit_transform(embeddings)
    all_preds['pc1'], all_preds['pc2'] = pcac[:,0], pcac[:,1]

    train_effect_preds = all_preds[all_preds['in effect training data']]
    train_species_preds = all_preds[all_preds['in species group training data']]
    remaining_preds = all_preds[(~all_preds['in effect training data']) | (~all_preds['in species group training data'])]


    fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['']),
        horizontal_spacing=0.02)

    if show_all_predictions:
        hover = (remaining_preds['SMILES_Canonical_RDKit'])
        fig.add_trace(go.Scatter(x=remaining_preds.pc1, y=remaining_preds.pc2, 
                        mode='markers',
                        text=hover,
                        name='Not in training data',
                        marker=dict(colorscale='turbo_r',
                                    cmax=4,
                                    cmin=-4,
                                    color=remaining_preds['predictions log10(mg/L)'],
                                    size=5,
                                    colorbar=dict(
                                        title='mg/L',
                                        tickvals=[2,0,-2,-4],
                                        ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                        orientation='h'),
                        )),
                        row=1, col=1)

    hover = (train_species_preds['SMILES_Canonical_RDKit'])
    fig.add_trace(go.Scatter(x=train_species_preds.pc1, y=train_species_preds.pc2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_species_preds['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=1,
                                        color='black'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                    )),
                    row=1, col=1)
    
    hover = (train_effect_preds['SMILES_Canonical_RDKit'])
    fig.add_trace(go.Scatter(x=train_effect_preds.pc1, y=train_effect_preds.pc2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group with Effect match',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_effect_preds['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=1,
                                        color='red'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                                )),
                    row=1, col=1)

    fig.update_xaxes(title_text=f"PC1 {np.round(100*pcomp.explained_variance_ratio_[0],1)}%",
        row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 {np.round(100*pcomp.explained_variance_ratio_[1],1)}%",
        row=1, col=1)

    fig.update_layout(height=800)

    return fig


@st.cache
def PlotUMAP_CLSProjection(endpoint, n_neighbors, min_dist):
    fig = make_subplots(rows=1, cols=1,
    subplot_titles=(['']),
    horizontal_spacing=0.02)

    if endpoint=='EC50':
        results = pd.read_pickle('data/REACH_predictions_EC50_EC50EC10.zip', compression='zip')
        our_smiles = pd.read_pickle('data/oursmilesec50mor96_EC50_EC50.zip', compression='zip')
    else:
        results = pd.read_pickle('data/REACH_predictions_EC10_EC50EC10.zip', compression='zip')
        our_smiles = pd.read_pickle('data/oursmilesec50mor96_EC50_EC50.zip', compression='zip')

    results = pd.concat([results, our_smiles])
    results.CLS_embeddings = results.CLS_embeddings.apply(lambda x: json.loads(x))
    embeddings = np.array(results.CLS_embeddings.tolist())

    umap_model = umap.UMAP(metric = "cosine",
                      n_neighbors = n_neighbors,
                      n_components = 2,
                      low_memory = False,
                      min_dist = min_dist)
    
    umapc = umap_model.fit_transform(embeddings)
    results['u1'], results['u2'] = umapc[:,0], umapc[:,1]

    pl1 = results[~results.SMILES_Canonical_RDKit.isin(our_smiles.SMILES_Canonical_RDKit.tolist())]
    pl2 = results[results.SMILES_Canonical_RDKit.isin(our_smiles.SMILES_Canonical_RDKit.tolist())]
    
    hover = (
    pl1['SMILES_Canonical_RDKit'])

    fig.add_trace(go.Scatter(x=pl1.u1, y=pl1.u2, 
                    mode='markers',
                    text=hover,
                    name='REACH',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=pl1['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=2,
                                        color='black'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                                )),
                    row=1, col=1)
    hover = (
    pl2['SMILES_Canonical_RDKit'])
    fig.add_trace(go.Scatter(x=pl2.u1, y=pl2.u2, 
                    mode='markers',
                    text=hover,
                    name='Training data',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=pl2['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=2,
                                        color='black'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                                )),
                    row=1, col=1)
    
    fig.update_layout(height=800)
    
    return fig