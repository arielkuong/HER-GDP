# Goal Density-based Hindsight Experience Prioritization (GDP) for Multi-Goal Robot Manipulation Reinforcement Learning

Here is the code for our paper presented in RoMAN 2020: https://www.researchgate.net/publication/344237429_Goal_Density-based_Hindsight_Experience_Prioritization_for_Multi-Goal_Robot_Manipulation_Reinforcement_Learning
This code is developed based on the OpenAI baseline demo code for HER but using PyTorch instead of TensorFlow.

## Requisition

This code requires Python3(>=3.5) with the development headers. You also need to install OpenAI Gym, PyTorch, NumPy with the latest version.

## Running the code
Simply run the train.py file in python3 with arguments set. 
For example, train the agent for FetchPush-v1 environment with HER+GDP using the hyperparameters same as used in our paper
```
python train.py --env-name=FetchPush-v1 --n-epochs=50 --n-cycles=50 --num-rollouts-per-cycle=40 --n-batches=40 --batch-size=5120 --buffer-size=5000000 --prioritization=goaldensity --temperature=1.0 --rank-method=rank --fit-interval=5 --replay-strategy=future --n-test-rollouts=100 --seed=123 --cuda
```
Or you can run the .sh script file provided with the code to run experiments with several random seed as benchmark.
