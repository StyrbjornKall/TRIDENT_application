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

def print_space_page():
    col1, col2 = st.columns((1,3))
    with col1:
        st.markdown('## Projection metrics')
        projection = st.selectbox('Projection method', ('UMAP','PCA'))
        species_groups = {'fish': 'fish', 'invertebrates': 'invertebrates', 'algae': 'algae'}
        endpoints = {'EC50': 'EC50', 'EC10': 'EC10'}
        effects = {'MOR': 'MOR', 'DVP': 'DVP', 'ITX':'ITX', 'GRO': 'GRO','POP': 'POP','MPH':'MPH'}
        PREDICTION_SPECIES = species_groups[st.radio("Select Endpoint ",tuple(species_groups.keys()), on_change=None)]
        PREDICTION_ENDPOINT = endpoints[st.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None)]
        PREDICTION_EFFECT = effects[st.radio("Select Effect ",tuple(effects.keys()), on_change=None)]
        PREDICTION_EXTENDED_DATA = st.checkbox('show predictions outside training data')
        if projection == 'UMAP':
            MIN_DISTNACE = st.number_input('min distance')
            N_NEIGHBORS = st.number_input('n neighbors')

        run_prediction = st.button('Predict')
    
    with col2:
        if run_prediction:
            with st.spinner(text = 'Inference in Progress...'):
                if projection == 'PCA':
                    st.plotly_chart(PlotPCA_CLSProjection(PREDICTION_ENDPOINT, PREDICTION_EFFECT, PREDICTION_SPECIES, PREDICTION_EXTENDED_DATA), use_container_width=True, theme='streamlit')
                if projection == 'UMAP':
                    st.plotly_chart(PlotUMAP_CLSProjection(PREDICTION_ENDPOINT, N_NEIGHBORS, MIN_DISTNACE), use_container_width=True, theme='streamlit')

@st.cache
def PlotPCA_CLSProjection(endpoint, effect, species_group, show_all_predictions):

    all_preds = pd.read_pickle(f'data/{species_group}_{endpoint}_{effect}_predictions.zip', compression='zip')

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