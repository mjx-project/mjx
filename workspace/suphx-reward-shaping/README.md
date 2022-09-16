## Suphnx-like reward shaping

## How to train the model

Prepare the directories for data and result under this directory. After that, we can train the model thorough cli.

```
$python train.py 0.001 10 16 --use_saved_data 0 --data_path resources/mjxproto --result_path result.
```

Here is the information about argument.

The first three are learning rate, epochs, batch size respectively.

`--use_saved_data` 0 means not to use saved data and other than 0 means otherwise. The default is 0.

`--round_candidates` We can specify rounds to use for training by this argument.

`--data_path` Please specify the data path.

`--result_path` Please specify the result path.





 



