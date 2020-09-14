# Goal Density-based Hindsight Experience Prioritization (GDP) for Multi-Goal Robot Manipulation Reinforcement Learning

Here is the code for our paper presented in RoMAN 2020: https://www.researchgate.net/publication/344237429_Goal_Density-based_Hindsight_Experience_Prioritization_for_Multi-Goal_Robot_Manipulation_Reinforcement_Learning
This code is developed based on the OpenAI baseline demo code for HER but using PyTorch instead of TensorFlow.

## Requisition

This code requires Python3(>=3.5) with the development headers. You also need to install OpenAI Gym, PyTorch, NumPy with the latest version.

## Run the code
Simply run the train.py file in python3 with arguments set. 
For example, train the agent for FetchPush-v1 environment with HER+GDP and seed as 574 using the hyperparameters same as used in our paper
```
python train.py --env-name=FetchPush-v1 --n-epochs=50 --n-cycles=50 --num-rollouts-per-cycle=40 --n-batches=40 --batch-size=5120 --buffer-size=5000000 --prioritization=goaldensity --temperature=1.0 --rank-method=rank --fit-interval=5 --replay-strategy=future --n-test-rollouts=100 --seed=574 --cuda
```

Train the agent for HandManipulateEggRotate-v0 environment with HER only and seed as 835 using the hyperparameters same as used in our paper
```
python3 train.py --env-name=HandManipulateEggRotate-v0 --n-epochs=40 --n-cycles=50 --num-rollouts-per-cycle=40 --n-batches=40  --batch-size=5120 --buffer-size=2000000 --prioritization=none --replay-strategy=future --n-test-rollouts=100 --save-dir=saved_models_her/ --seed=835 --cuda
```

Or you can run the .sh script file provided with the code to run experiments with several random seed as benchmark.

## Citation
To cite our paper, using the following version:
```
@inproceedings{kuang2020goal,
  title={Goal Density-based Hindsight Experience Prioritization for Multi-Goal Robot Manipulation Reinforcement Learning},
  author={Kuang, Yingyi and Weinberg, Abraham Itzhak and Vogiatzis, George and Faria, Diego R},
  booktitle={2020 29th IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN)},
  year={2020},
  organization={IEEE}
}

```
