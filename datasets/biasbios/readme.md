# Bias in Bios

## Dataset

### Download 

(Version used in the article "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection, Ravfogel et al., 2O20")
```bash
mkdir -p data/biasbios
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/
```


Two typical entries:

```json
{
	g:	f
	p:	professor
	text:	Janine Langan is an Associate Professor at the University of Toronto. She founded and teaches the Christianity and Culture program. Janine has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
	start:	69
	hard_text:	She founded and teaches the Christianity and Culture program . Janine has written and lectured extensively on art , the family , the media , and the problems of Catholic education .
	hard_text_untokenized:	She founded and teaches the Christianity and Culture program. Janine has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
	text_without_gender:	_ founded and teaches the Christianity and Culture program. _ has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
}

{
	g:	f
	p:	psychologist
	text:	Gillian Burrell is a psychoanalytic psychotherapist in private practice in Sydney. She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that she worked with Relationships Australia as a family and relationship therapist.
	start:	82
	hard_text:	She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy . Before that she worked with Relationships Australia as a family and relationship therapist .
	hard_text_untokenized:	She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that she worked with Relationships Australia as a family and relationship therapist.
	text_without_gender:	_ is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that _ worked with Relationships Australia as a family and relationship therapist.
}
```


## Statistics

Number of observations:
- train: 255,710
- validation: 39,369
- test: 98,344

## Embedding calculation

To calculate embeddings, run :

```
python encode_bert_states_final.py
```
1 .npz files will be generated in this repo., containing bert's embeddings, labels for manipulated attributes and labels for the downstream task. \
This file is the one used to run the experiments and must be generated upstream. 

The file to be generated : 
- D_original.npz 
