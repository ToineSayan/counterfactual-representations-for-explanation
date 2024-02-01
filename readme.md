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

## Experiments on synthetic data (sections 5.1 and 5.2)

Run the notebooks:
- `CFRs_EEECp_gender_balanced.ipynb` 
- `CFRs_EEECp_gender_aggressive.ipynb`
- `CFRs_EEECp_race_balanced.ipynb` 
- `CFRs_EEECp_race_aggressive.ipynb`

## Experiments on the natural dataset BiasInBios (sections 5.3 and 5.4)

Run the notebook:
- `CFRs_biasbios.ipynb` 
