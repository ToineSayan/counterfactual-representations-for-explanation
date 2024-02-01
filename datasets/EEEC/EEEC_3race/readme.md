# EEEC+ dataset

## Dataset

The dataset observations are stored in the 'new_data.json' file.

Each line contains an entry of the following format:

```json
{
    "id": "53", 
    "template": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, <person> found <gender-pronoun> in an <emotion-situation-adjective> situation, appreciating creativity and expression.", 
    "split": "train", # the split the observation belongs to among 'train', 'validation' and 'test'
    "balanced": { # the observation in the balanced version
        "id": "53_balanced", 
        "person": "Madeline", 
        "sentence": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, Madeline found herself in a frustrating situation, appreciating creativity and expression.", 
        "gender_label": "female", # gender
        "race_label": "White American", # race among 'White American', 'Black or African-American', 'Asian American'
        "emotion_word": "frustrating", 
        "poms_label": "anger" # mood state
    }, 
    "CF_race": { # the genuine CF with respect to race of the balanced observation
        "id": "53_CF_race", 
        "person": "Marquita", 
        "sentence": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, Marquita found herself in a frustrating situation, appreciating creativity and expression.", 
        "gender_label": "female", 
        "race_label": "Black or African-American", 
        "emotion_word": "frustrating", 
        "poms_label": "anger"
    }, 
    "CF_gender": { # the genuine CF with respect to gender of the balanced observation
        "id": "53_CF_gender", 
        "person": "Brendan", 
        "sentence": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, Brendan found himself in a frustrating situation, appreciating creativity and expression.", 
        "gender_label": "male", 
        "race_label": "White American", 
        "emotion_word": "frustrating", 
        "poms_label": "anger"
    }, 
    "aggressive_gender": { # the observation in the aggressive version with respect to gender
        "id": "53_aggressive_gender", 
        "person": "Jake", 
        "sentence": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, Jake found himself in a daunting situation, appreciating creativity and expression.", 
        "gender_label": "male", 
        "race_label": "White American", 
        "emotion_word": "daunting", 
        "poms_label": "fear"
    }, 
    "aggressive_race": { # the observation in the aggressive version with respect to race
        "id": "53_aggressive_race", 
        "person": "Jyothi", 
        "sentence": "The museum curator explained the history behind the artifacts. Amidst the artistic exhibition, Jyothi found herself in an incensing situation, appreciating creativity and expression.", 
        "gender_label": "female", 
        "race_label": "Asian American", 
        "emotion_word": "incensing", 
        "poms_label": "anger"
    }
}
```


## Statistics

Number of observations for balanced and aggressive versions:
- train: 25600
- validation: 6400
- test: 8000

## Embedding calculation

To calculate embeddings, run :

```
python encode_bert_states.py
```
6 .npz files will be generated in this repo., containing bert's embeddings, labels for manipulated attributes and labels for the downstream task. \
These files are the ones used to run the experiments and must be generated upstream. 

The files to be generated : 
- D_balanced.npz : the balanced version of EEEC+.
- D_aggressive_gender.npz : the aggressive version with respect to gender of EEEC+.
- D_aggressive_race.npz : the aggressive version with respect to race of EEEC+.
- D_CF_gender.npz : the genuine gender counterfactuals of balanced data.
- D_CF_race.npz : the genuine race counterfactuals of balanced data.
- D_balanced_templates : an encoding of the template without gender, race or mood state indicators.