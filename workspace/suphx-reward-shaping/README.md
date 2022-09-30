## Reward shaping
In order to handle a round as an episode in RL, it is important to align the game reward to each round appropriately.
We call it reward shaping. we will prepare 8 NN (NN_0, ...NN_7) for each round and leaning procedure is as follows

- train NN_7: input: features at the begining of round 7, target: game reward.
- train NN_6: input: features at the begining of round 7, target: prediction by NN_7 on the round7.
- ... 


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





 



