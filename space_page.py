import streamlit as st

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA

def print_space_page():
    col1, col2 = st.columns((1,3))
    with col1:
        st.markdown('## Projection metrics')
        st.selectbox('Projection method', ('UMAP','PCA'))
        endpoints = {'EC50': 'EC50', 'EC10': 'EC10'}
        effects = {'MOR': 'MOR', 'DVP': 'DVP', 'GRO': 'GRO','POP': 'POP','MPH':'MPH'}
        PREDICTION_ENDPOINT = endpoints[st.radio("Select Endpoint ",tuple(endpoints.keys()), on_change=None)]
        PREDICTION_EFFECT = effects[st.radio("Select Effect ",tuple(effects.keys()))]
        run_prediction = st.button('Predict')
    
    with col2:
        if run_prediction:
            with st.spinner(text = 'Inference in Progress...'):
                st.plotly_chart(PlotPCA_CLSProjection_REACH(PREDICTION_ENDPOINT), use_container_width=True, theme='streamlit')

@st.cache
def PlotPCA_CLSProjection_REACH(endpoint):

    if endpoint=='EC50':
        results = pd.read_csv('data/REACH_predictions_EC50_EC50EC10.csv',sep='\t')
    else:
        results = pd.read_csv('data/REACH_predictions_EC10_EC50EC10.csv',sep='\t')
    results.CLS_embeddings = results.CLS_embeddings.apply(lambda x: json.loads(x))
    embeddings = np.array(results.CLS_embeddings.tolist())

    pcomp = PCA(n_components=3)
    pca = pd.DataFrame(data = pcomp.fit_transform(embeddings), columns = ['pc1', 'pc2','pc3'])
    results = pd.concat([results, pca], axis=1)

    hover = (
    results['SMILES'])

    fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['']),
        horizontal_spacing=0.02)
    
    fig.add_trace(go.Scatter(x=results.pc1, y=results.pc2, 
                    mode='markers',
                    text=hover,
                    marker=dict(colorscale=[(0, '#67000d'),
                        (0.25, '#fb6a4a'),
                        (0.5, '#c994c7'),
                        (0.75, '#4393c3'), 
                        (1, '#023858')],
                                cmax=3,
                                cmin=-4,
                                color=results['predictions log10(mg/L)'],
                                size=7,
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"]),

                                )),
                    row=1, col=1)

    fig.update_xaxes(title_text=f"PC1 {np.round(100*pcomp.explained_variance_ratio_[0],1)}%",
        row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 {np.round(100*pcomp.explained_variance_ratio_[1],1)}%",
        row=1, col=1)

    fig.update_layout(height=800)

    return fig