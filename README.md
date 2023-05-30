# BERT pre-training on The Stack
Exploration of BERT-like models trained on The Stack.

- Code used to train [StarEncoder](https://huggingface.co/bigcode/starencoder).
  - StarEncoder was fine-tuned for PII detection to pre-process the data used to train [StarCoder](https://arxiv.org/abs/2305.06161)

- This repo also contains functionality to train encoders with contrastive objectives.

- [More details.](https://docs.google.com/document/d/1gjf7Y2Ek64xSyl8HE3GoK1kxDgsV8kjy-9pyIBkR-RQ/edit?usp=sharing)


## To launch pre-training:

After installing requirements, training can be launched via the example launcher script:

```
./launcher.sh
```

### Note that:

- ```--train_data_name``` can be used to use to set the training dataset.

- Hyperparamaters can be changed in ```exp_configs.py```.
  - The tokenizer to be used is treated as a hyperparameter and then must also be set in ```exp_configs.py```.
  - alpha is used to weigh the BERT losses (NSP+MLM) and the contrastive objective.
    - Setting alpha to 1 corresponds to the standard BERT objective.
  - Token masking probabilities are set as separate hyperparameters, one for MLM and another one for the contrastive loss.
