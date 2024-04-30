# Explaining text classifiers with counterfactual representations

## Environment

Create and start a new virtual environment:
```sh
conda create -n CFR python=3.10.9 anaconda
conda activate CFR
```

## Data

Download and pre-process the data before running the experiments. 

- EEEC+:  data and pre-processing in "./datasets/EEEC/EEEC_3race"
- BiasInBios: data and pre-processing in "./datasets/biasbios"
- CEBaB: 
    - data: https://cebabing.github.io/CEBaB/
    - pre-processing in "./datasets/CEBaB-v1.1"
- GloVe : 
    - data: 
        https://nlp.biu.ac.il/~ravfogs/rlace/glove/glove-gender-data.pickle (GloVe embeddings with gender-bias labels)
        https://nlp.biu.ac.il/~ravfogs/rlace/glove/glove-top-50k.pickle (150k GloVe embeddings)

        

## Experiments on synthetic data (sections 5.1 and 5.2)

Run the notebooks:
- `CFRs_EEECp_gender_balanced.ipynb` 
- `CFRs_EEECp_gender_aggressive.ipynb`
- `CFRs_EEECp_race_balanced.ipynb` 
- `CFRs_EEECp_race_aggressive.ipynb`

## Experiments on the natural dataset BiasInBios (sections 5.4 and Supplementary material D)

Run the notebook:
- `CFRs_biasbios.ipynb` 

## Experiments on CEBaB (section 5.3)

Run:
- `CFRs_CEBaB_compare_methods.ipynb` 

## Experiment on GloVe embeddings (supplementary material C)

Run the notebook:
- `CFRs_GloVe.ipynb` 



