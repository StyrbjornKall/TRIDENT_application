import streamlit as st
import base64

with open("assets/final_model.svg", "rb") as image_file:
    image_data = image_file.read()
    tri1_img_64 = base64.b64encode(image_data).decode()

with open("assets/trident2_model.svg", "rb") as image_file:
    image_data = image_file.read()
    tri2_img_64 = base64.b64encode(image_data).decode()

def print_doc_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('TRIDENT-2')
        st.markdown(f'''
        TRIDENT-2 is the second model generation in the TRIDENT family. Rather than being
        restricted to a fixed set of standard test species, TRIDENT-2 accepts any species
        identified by its NCBI taxonomy ID (`taxid`) as an explicit model input, alongside the
        chemical structure, toxicity effect, administration route, organism lifestage, exposure
        duration, and concentration unit. This allows a single model to predict toxicity for
        organisms spanning Eukaryota — including fish, birds, insects, mollusks, plants, and
        fungi — extending effect-concentration prediction beyond the standard test species used
        in regulatory testing to a broader range of biodiversity.

        ## Model architecture
        TRIDENT-2 is a multimodal deep learning model. A transformer encoder reads the chemical
        structure from its SMILES representation, while the species (via taxid), effect,
        administration route, organism lifestage, and concentration unit are encoded as
        additional model inputs. The model then simultaneously predicts four toxicity endpoints
        — EC50, EC10, NOEC, and LOEC — as log10(concentration) in the user-selected unit.

        <div style="text-align:center;width:100%;margin: 0 auto;">
        <img src="data:image/svg+xml;base64,{tri2_img_64}" style="max-width: 100%;"><br>
        <small><b>TRIDENT-2 model architecture | </b> Chemical structure, species taxid, effect, administration route, organism lifestage, exposure duration, and concentration unit are combined and passed through the model to jointly predict EC50, EC10, NOEC, and LOEC.</small>
        </div>

        ''', unsafe_allow_html=True)

        st.title('TRIDENT-1')

        st.markdown(f'''
        For a detailed explanation of the model framework and model performance compared to traditional QSARs, please refer to the published paper (see Publications).

        ## Model architecture
        TRIDENT is a deep learning model for ecotoxicity predictions. It is based on a transformer language model (ChemBERTa ~83M paramters) and a deep neural network (~1M parameters). The transformer encodes a chemical structure from its SMILES into a continuous vector of length 768, called the CLS-embedding. The CLS-embedding together with information on toxicity endpoint, effect and exposure duration is used as input in the deep neural network that finally predicts the effect concentration as a log10 concentration (log10 mg/L).

        <div style="text-align:center;width:100%;margin: 0 auto;">
        <img src="data:image/svg+xml;base64,{tri1_img_64}" style="max-width: 100%;"><br>
        <small><b>TRIDENT-1 model architecture | </b> The model uses a pre-trained 6-encoder layer RoBERTa transformer (ChemBERTa) to interpret the SMILES into an embedding vector of dimension 768 (E[CLS]), representing the molecular structure with regard to its toxicity. The embedding vector is then amended with information on exposure duration, toxicological effect, and endpoint and used as input to a deep neural network. The network then predicts the associated toxicity in the form of an effect concentration (EC50 and EC10).</small>
        </div>
        

        ### Training
        The model's two modules are trained simultaneously so that the transformer encodes the SMILES with chemical toxicity in mind. Thus, the CLS-embedding is a more effective representation of the chemical structure with respect to its toxicity compared to e.g. traditional chemical fingerprinting techniques that are primarily built on the presence and absence of pre-specified groups.
        
        The model is trained using supervised learning and thus depends on labeled data to adjust the ~84M trainable parameters in the model. The data consists of toxicity endpoints collected from REACH dossiers, the US EPA database ECOTOX and the EFSA openFoodTox collection of pesticide registration data.

        ## Model versions
        For model development, 9 different TRIDENT models were examined for clear evaluation of model performance (as compared to the QSAR tools ECOSAR, VEGA, T.E.S.T.) and behaviour. The model versions depend on organism group, endpoint, and effect combinations.
        
        ### Organism groups
        The training data belong of three different trophic groups: fish, aquatic invertebrates and algae. Thus, there are three major model groups, one for each of these groups.

        ### Endpoints
        The training data contained several different measurements of EC50, EC10 and NOEC endpoints. EC50 is the concentration at which 50 % of the tested population exhibit an effect. EC10 refers to the concentration where 10% of the population exhibit the measured effect and NOEC is the No Observed Effect Concentration, the last tested concentration which showed no statistically significant effects when compared to the control. To ensure that sufficient data was available for training the EC10 models, EC10 and NOEC were grouped together during model development.

        Within each organism group, three models were trained. One model for only EC50, one for the combined EC10/NOEC datasets and one for the combination of EC50, EC10 and NOEC (called `Combined model`). Results show that combining the endpoints, i.e. using the `Combined model`, increased the performance in all organism groups. Therefore, this model is recommended for the most accurate predictions. However, for comparative reasons we currently also allow users to receive predictions from the specific EC50 and EC10 models. Thus, in total, there are currently nine different models available under the TRIDENT umbrella.

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
        Results show that three of the models, namely the ones trained on a combination of endpoints (both EC50 and EC10) simultaneously, have the highest predictive power. Thus, these are the models that can be used with on this web application, the remaining individual models are available through the associated Github (see below).

        | Model version          | Endpoint         | Effect                            |
        |------------------------|------------------|-----------------------------------|
        | Fish-EC50              | EC50             | MOR                               |
        | Fish-EC10              | EC10, NOEC       | MOR, DVP, ITX, MPH, REP, POP, GRO |
        | <b>Fish-EC50EC10*</b>          | EC50, EC10, NOEC | MOR, DVP, ITX, MPH, REP, POP, GRO |
        | Invertebrates-EC50     | EC50             | MOR, ITX                          |
        | Invertebrates-EC10     | EC10             | MOR, DVP, ITX, MPH, REP, POP      |
        | <b>Invertebrates-EC50EC10*</b> | EC50, EC10, NOEC | MOR, DVP, ITX, MPH, REP, POP      |
        | Algae-EC50             | EC50             | POP                               |
        | Algae-EC10             | EC10             | POP                               |
        | <b>Algae-EC50EC10*</b>        | EC50, EC10, NOEC | POP                               |
        
        <small><b>*Available on this platform</b></small>

        ## Code availability
        The code used to generate this webpage is available through: https://github.com/StyrbjornKall/TRIDENT_application

        The code associated with the TRIDENT-1 model development and publication "Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms" is available through: https://github.com/StyrbjornKall/TRIDENT.

        The code associated with the TRIDENT-2 model development and publication "TRIDENT-2" is available through: https://github.com/StyrbjornKall/TRIDENT-2.

        ## Citation
        If you use TRIDENT-1 or TRIDENT-2 in your research, please cite the relevant publication under the Publications tab.
        ''', unsafe_allow_html=True)



