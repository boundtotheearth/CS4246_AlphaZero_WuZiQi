### Setup
Please install the required libraries listed in `requirements.txt`.

### Play against Human
To play a game against the ALphaZero agent, execute the command:

`python human_play.py`

This uses our trained model saved in `models/best_policy_664_1.pt`

### Play against MCTS
To play a game against a pure MCTS agent, execute the command:

`python evaluation.py`

This uses our trained model saved in `models/best_policy_664_1.pt`

### Training a Model
To train a model from scratch, execute the command:

`python train.py`

The best performing policy will be saved as `best_policy.pt`, while the latest policy at the end of training is saved as `current_policy.pt`

### References
This implementation references the implementation found in [this project](https://github.com/junxiaosong/AlphaZero_Gomoku)