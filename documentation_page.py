import streamlit as st

def print_doc_page():
    
    st.title('Documentation')

    st.markdown('''
    ## Model architecture
    ecoCAIT is a deep learning model for ecotoxicity predictions. It is based on a large language model (ChemBERTa ~83M paramters) and a deep neural network (~1M parameters).
    The language model encodes a chemical structure from its SMILES into a 768 continuous vector, called the CLS-embedding. The deep neural network transforms the vector and adds information on toxicity 
    endpoint, effect and exposure duration to finally predict a log10 concentration (log10 mg/L). 

    ### Training
    The model's two modules are trained simultaneously so that the language model encodes the SMILES with the chemical's toxicity in mind. Thus the CLS-embedding is a more effective representation
    of the chemical structure compared to e.g. fingerprinting which is built solely on the presence and absence of pre-specified funtional groups.

    The model is trained using supervised learning and thus depends on labelled data to adjust the ~84M trainable parameters in the model. The data consists of toxicity endpoints collected from L!.

    ## Model versions
    In total there are 9 ecoCAIT models available. The model versions depend on species group and endpoint combinations.

    ### Species groups
    The training data belong of three different species groups: fish, aquatic invertebrates and algae. Thus there are three major model groups, one for each species group.

    ### Endpoints
    The training data contain EC50, EC10 and NOEC toxicity endpoints. EC50 refer to the effect concentration of the target chemical where 50 % of the population exhibit an effect. EC10 refers to the concentration where 
    10% of the population exhibit the effect and NOEC is the No Observed Effect Concentration. To ensure sufficient data for the EC10 models, EC10 and NOEC were grouped together during model development and training.

    Within each species group, three models were trained. One model for only EC50, one for the combined EC10/NOEC datasets and one for the combination of EC50, EC10 and NOEC (called `Combined model`). Results prove that the combination of endpoints,
    i.e. `Combined model`, showed best performance in all species groups. Therefore, this model is recommended for best predictions. Thus in total, there are 9 different models available under ecoCAIT. 

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

    For a more elaborate description of all models and datasets involved in model development and training, refer to L!.

    
    ''', unsafe_allow_html=True)



