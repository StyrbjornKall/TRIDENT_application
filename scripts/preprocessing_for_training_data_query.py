# %% [markdown]
# # Find training data

# %%
from src.trident1.inference_utils.pytorch_data_utils import check_training_data
import pandas as pd
from tqdm.notebook import tqdm

# %%
species_groups = ['fish', 'algae', 'invertebrates']
models = ['EC50EC10', 'EC50', 'EC10']
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

default_durations = {
    'algae': 72,
    'fish': 96,
    'invertebrates': 48
}

# %% [markdown]
# ## Get CLS embeddings

# %%
raw_data = pd.read_excel('/cephyr/users/skall/Alvis/Ecotoxformer/Inference_2/ecoCAIT/data/development/Preprocessed_complete_data.xlsx', sheet_name='dataset')

# %%
SMILES_COLUMN_NAME = 'SMILES_Canonical_RDKit'
for SPECIES in tqdm(species_groups):
    for model in models:
        cls_dict = {}
        MODEL_VERSION = f'{model}_{SPECIES}'
        EXPOSURE_DURATION = default_durations[SPECIES]
        PREDICTION_ENDPOINT = list(endpointordering[MODEL_VERSION].keys())[0]
        PREDICTION_EFFECT = list(effectordering[MODEL_VERSION].keys())[0]
        ecocait = ecoCAIT_for_inference(model_version=MODEL_VERSION, path_to_model_weights='/cephyr/users/skall/Alvis/Ecotoxformer/Inference_2/ecoCAIT/ecoCAIT/')
        ecocait.load_fine_tuned_model()
        data = raw_data.copy()
        
        data = data.drop_duplicates(subset=['SMILES_Canonical_RDKit'])

        results = ecocait.predict_toxicity(SMILES = data[SMILES_COLUMN_NAME].tolist(), exposure_duration=EXPOSURE_DURATION, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, return_cls_embeddings=True)
        results.reset_index(drop=True, inplace=True)
        results['CLS_embeddings'] = results['CLS_embeddings'].apply(lambda x: np.asarray(x, dtype=np.float32))
        results = results[['SMILES_Canonical_RDKit','CLS_embeddings']]        
        results.to_pickle(f'./data/trident1/predictions/{MODEL_VERSION}_CLS_embeddings.pkl.zip', compression='zip')

# %% [markdown]
# ## Get predictions

# %%
SMILES_COLUMN_NAME = 'SMILES_Canonical_RDKit'
ct=0
for SPECIES in tqdm(species_groups):
    
    for model in models:
        MODEL_VERSION = f'{model}_{SPECIES}'
        ecocait = ecoCAIT_for_inference(model_version=MODEL_VERSION, path_to_model_weights='/cephyr/users/skall/Alvis/Ecotoxformer/Inference_2/ecoCAIT/ecoCAIT/')
        ecocait.load_fine_tuned_model()
        data = raw_data.copy()
        
        for PREDICTION_ENDPOINT in endpointordering[MODEL_VERSION]:
            for PREDICTION_EFFECT in effectordering[MODEL_VERSION]:
                print(MODEL_VERSION, PREDICTION_ENDPOINT, PREDICTION_EFFECT)
                try:
                    EXPOSURE_DURATION = data[(data.species_group==SPECIES) & (data.endpoint==PREDICTION_ENDPOINT) & (data.effect==PREDICTION_EFFECT)].Duration_Value.value_counts().index[0]
                except:
                    EXPOSURE_DURATION = default_durations[SPECIES]
                data = data.drop_duplicates(subset=['SMILES_Canonical_RDKit'])

                results = ecocait.predict_toxicity(SMILES = data[SMILES_COLUMN_NAME].tolist(), exposure_duration=EXPOSURE_DURATION, endpoint=PREDICTION_ENDPOINT, effect=PREDICTION_EFFECT, return_cls_embeddings=True)
                results.reset_index(drop=True, inplace=True)
                results['exposure_duration'] = EXPOSURE_DURATION
                
                if ct == 0: # first time model is used:
                    for column in results.columns:
                        results.rename(columns={column: f'{MODEL_VERSION}_{PREDICTION_ENDPOINT}_{PREDICTION_EFFECT} {column}'}, inplace=True)
                    combined_results = results[['SMILES_Canonical_RDKit', 'predictions log10(mg/L)', 'exposure_duration']]
                    ct += 1
                else:
                    results = results[['predictions log10(mg/L)', 'exposure_duration']]
                    ct+=1
                
                for column in results.columns:
                    results.rename(columns={column: f'{MODEL_VERSION}_{PREDICTION_ENDPOINT}_{PREDICTION_EFFECT} {column}'}, inplace=True)
                
                if ct > 1:
                    combined_results = pd.concat([combined_results, results], axis=1)

# %%
combined_results

# %%
for col in combined_results.columns:
    if 'exposure_duration' in col:
        combined_results[col] = combined_results[col].astype(np.float32)

# %%
combined_results.info(memory_usage='deep')

# %%
combined_results.to_pickle(f'./data/trident1/predictions/combined_predictions.pkl.zip', compression='zip')

# %% [markdown]
# ## Get training data matches

# %%
training_data = pd.read_pickle('./data/trident1/Preprocessed_complete_data_fixed_smiles_format.zip', compression='zip')
all_preds = pd.read_pickle(f'./data/trident1/predictions/combined_predictions.pkl.zip', compression='zip')

# %%
all_preds

# %%
for model in models:
    for species in tqdm(species_groups):
        MODELTYPE = f'{model}_{species}'
        for endpoint in endpointordering[MODELTYPE]:
            for effect in effectordering[MODELTYPE]:
                all_preds = check_training_data(all_preds, model, species, endpoint, effect)

                for col in all_preds.columns:
                    if ((col == 'species match') | (col == 'endpoint match') | (col == 'effect match')):
                        all_preds.rename(columns={col: f'{model}_{species}_{endpoint}_{effect} {col}'}, inplace=True)

# %%
all_preds

# %%
all_preds.to_pickle(f'./data/trident1/predictions/combined_predictions.pkl.zip', compression='zip')

# %% [markdown]
# ## Add errors from 10x10 CV

# %%
from development.figures.figure_utils.preprocess_data import Preprocess10x10Fold, GroupDataForPerformance
from development.development_utils.preprocessing.Get_data_for_model import PreprocessData
from tqdm.notebook import tqdm

# %%
import pandas as pd
training_data = pd.read_pickle('./data/tutorials/combined_predictions.pkl.zip', compression='zip')
training_data['SMILES'] = training_data['SMILES_Canonical_RDKit'].copy()
training_data = PreprocessData(training_data).GetCanonicalSMILES()

# %%
def match(x, error_dict):
    try:
        return error_dict[x]
    except:
        return None

# %%
for species_group in ['fish', 'invertebrates','algae']:
    for model in tqdm(['EC50','EC10','EC50EC10']):
        if model != 'EC50EC10':
            cvpreds = Preprocess10x10Fold(name=f'{model}_{species_group}', uselogdata=True, full_filepath=f'./data/results/{model}_{species_group}_predictions_100x_CV_RDkit.zip')
        else:
            cvpreds = Preprocess10x10Fold(name=f'{model}_{species_group}', uselogdata=True, full_filepath=f'./data/results/{model}_{species_group}_withoverlap_predictions_100x_CV_RDkit.zip')
        
        wavgcv = GroupDataForPerformance(cvpreds)
        wavgcv['SMILES'] = wavgcv['Canonical_SMILES_figures'].copy()
        wavgcv = PreprocessData(wavgcv).GetCanonicalSMILES()
        error_dict = dict(zip(wavgcv.SMILES_Canonical_RDKit.tolist(), wavgcv.L1error.tolist()))

        training_data[f'{model}_{species_group} L1Error'] = training_data.SMILES_Canonical_RDKit.apply(lambda x: match(x, error_dict))

# %%
training_data.to_pickle('./data/tutorials/combined_predictions_and_errors.pkl.zip', compression='zip')
