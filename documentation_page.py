import streamlit as st

def print_doc_page():
    
    st.title('Documentation')

    st.markdown('''
    ## Model architecture
    TRIDENT is a deep learning model for ecotoxicity predictions. It is based on a transformer language model (ChemBERTa ~83M paramters) and a deep neural network (~1M parameters). The transformer encodes a chemical structure from its SMILES into a continuous vector of length 768, called the CLS-embedding. The CLS-embedding together with information on toxicity endpoint, effect and exposure duration is used as input in the deep neural network that finally predicts the effect concentration as a log10 concentration (log10 mg/L).

    ### Training
    The model's two modules are trained simultaneously so that the transformer encodes the SMILES with the chemical's toxicity in mind. Thus, the CLS-embedding is a more effective representation of the chemical structure with respect to its toxicity compared to e.g. traditional chemical fingerprinting techniques that are primarily built on the presence and absence of pre-specified groups.
    
    The model is trained using supervised learning and thus depends on labelled data to adjust the ~84M trainable parameters in the model. The data consists of toxicity endpoints collected from REACH dossiers, the US EPA database ECOTOX and the EFSA openFoodTox collection of pesticide registration data.

    ## Model versions
    In total there are 9 TRIDENT models available. The model versions depend on species group and endpoint combinations.

    ### Species groups
    The training data belong of three different trophic groups: fish, aquatic invertebrates and algae. Thus, there are three major model groups, one for each of these groups.

    ### Endpoints
    The training data contained several different measurements of EC50, EC10 and NOEC endpoints. EC50 is the concentration at which 50 % of the tested population exhibit an effect. EC10 refers to the concentration where 10% of the population exhibit the measured effect and NOEC is the No Observed Effect Concentration, the last tested concentration which showed no statistically significant effects when compared to the control. To ensure that sufficient data was available for training the EC10 models, EC10 and NOEC were grouped together during model development.

    Within each species group, three models were trained. One model for only EC50, one for the combined EC10/NOEC datasets and one for the combination of EC50, EC10 and NOEC (called `Combined model`). Results show that combining the endpoints, i.e. using the `Combined model`, increased the performance in all species groups. Therefore, this model is recommended for the most accurate predictions. However, for comparative reasons we currently also allow users to receive predictions from the specific EC50 and EC10 models. Thus, in total, there are currently nine different models available under the TRIDENT umbrella.

    ### Effects
    The training data contain the following effects: 

    1. MOR (mortality)
    2. DVP (development)
    3. ITX (intoxication) 
    4. MPH (morphology)
    5. REP (reproduction)
    6. POP (population)
    7. GRO (growth)

    ### Summary of all models
    | Model version          | Endpoint         | Effect                            |
    |------------------------|------------------|-----------------------------------|
    | Fish-EC50              | EC50             | MOR                               |
    | Fish-EC10              | EC10, NOEC       | MOR, DVP, ITX, MPH, REP, POP, GRO |
    | Fish-EC50EC10          | EC50, EC10, NOEC | MOR, DVP, ITX, MPH, REP, POP, GRO |
    | Invertebrates-EC50     | EC50             | MOR, ITX                          |
    | Invertebrates-EC10     | EC10             | MOR, DVP, ITX, MPH, REP, POP      |
    | Invertebrates-EC50EC10 | EC50, EC10, NOEC | MOR, DVP, ITX, MPH, REP, POP      |
    | Algae-EC50             | EC50             | POP                               |
    | Algae-EC10             | EC10             | POP                               |
    | Algae-EC50EC10         | EC50, EC10, NOEC | POP                               |

    For a more elaborate description of all models and datasets involved in model development and training, refer to the associated publication.

    
    ''', unsafe_allow_html=True)



