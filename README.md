
Methodology:

I used BERT pretrained model from huggingface transformers library. I used small-uncased BERT.

I create an ensemble of N BERT models, by training every model on (N-1)/N observations and using 1/N for evaluation. The N parameter is specified in `/src/bert_model/parameters.py` `N_PARTS`. Using `N_PARTS=20` gave me the best result. 

Reasoning for using ensemble:

1) *I want to be able to train all the available data.* If I train on 80% of the data, and use 20% for validation, then I am not using 20% of the data to make my model smarter.

2) *I want to have a validation set.* I could train on 100% of the data, but then I would not know if my model is overfitting and the training needs to be stopped.

3) *Ensembles outperform single models.* It has been shown in the literature that ensembles of models trained on the same data (just different initial weight allocations) outperform single models. I believe the fact tha I am training the models on different, though slightly, data makes ensembles even more powerful. 



Run the code:

`test.csv` and `train.csv` need to be put into a `/data` directory or modify the names of the files in the `/src/data_sources`.

Build the models:

```
python src/bert_model/run_bert_model.py
```

Make `/output/predictions.csv` or modify the name of the output file in the `/src/data_sources`.

```
python src/bert_model/predictor.py
```

Result:

The highest score I achieved is 0.83941.


