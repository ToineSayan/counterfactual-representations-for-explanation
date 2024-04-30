# CEBaB

## Dataset

### Download 

The official repository for CEBaB is here: https://github.com/CEBaBing/CEBaB


One typical entry:

```json
{
	"id":"000011_000005",
	"original_id":"000011",
	"edit_id":"000005",
	"is_original":false,
	"edit_goal":"unknown",
	"edit_type":"service",
	"edit_worker":"w17",
	"description":"Disappointing. Food was fair at best and seem to lack seasoning or something to bring out the flavor.",
	"review_majority":"2",
	"review_label_distribution":{"2":4,"4":1},
	"review_workers":{"w168":"2","w233":"2","w230":"2","w40":"4","w33":"2"},
	"food_aspect_majority":"Negative",
	"ambiance_aspect_majority":"unknown",
	"service_aspect_majority":"unknown",
	"noise_aspect_majority":"unknown",
	"food_aspect_label_distribution":{"Negative":5},
	"ambiance_aspect_label_distribution":{"unknown":5},
	"service_aspect_label_distribution":{"unknown":5},
	"noise_aspect_label_distribution":{"unknown":3,"Negative":2},
	"food_aspect_validation_workers":{"w43":"Negative","w50":"Negative","w2":"Negative","w128":"Negative","w76":"Negative"},
	"ambiance_aspect_validation_workers":{"w21":"unknown","w74":"unknown","w1":"unknown","w89":"unknown","w128":"unknown"},
	"service_aspect_validation_workers":{"w148":"unknown","w1":"unknown","w150":"unknown","w5":"unknown","w9":"unknown"},
	"noise_aspect_validation_workers":{"w63":"Negative","w83":"Negative","w35":"unknown","w76":"unknown","w103":"unknown"},
	"opentable_metadata":{
		"restaurant_id":17602,
		"restaurant_name":"Serafina at The Time Hotel",
		"cuisine":"italian",
		"price_tier":"low",
		"dining_style":"Casual Elegant",
		"dress_code":"Smart Casual",
		"parking":"Public Lot",
		"region":"northeast",
		"rating_ambiance":3,
		"rating_food":2,
		"rating_noise":2,
		"rating_service":2,
		"rating_overall":2
	}
}
```


## Statistics

Number of observations:
- train (exclusive): 1,755
- validation: 1,673
- test: 1,689

## Embedding calculation

To calculate embeddings, run :

```
python encode_bert_states.py
```
1 .npz files will be generated in this repo., containing bert's embeddings, labels for manipulated attributes and labels for the downstream task. \
This file is the one used to run the experiments and must be generated upstream. 

The file to be generated : 
- D_train_exclusive.npz 

## Bert finetuning


For CEBaB experiments, Bert finetuning is performed by running the following program, located in the './datasets/CEBaB-v1.1' folder:

```
python finetune.py
```

1 Bert model will be stored in the repository (filename: finetuned_bert_uncased_cebab_train_exclusive)