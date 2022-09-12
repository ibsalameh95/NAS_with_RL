# Neural Architecture Search with Reinforcement Learning

This repository is the official implementation of [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01578v2.pdf). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

>Some information about the data and where to get...

## Project files
 
* __policy_gradient.py__
	* Aim of this policy try to find best hyperparameters for a CNN model
	* Controller agent builds hyperparameters with probabilty p, to train CNN model. 
	* Accuracy value of CNN model is a reward for the controller, it is optimized to generate .	
* __model.py__ 
	* Includes a CNN model which applies 2D convolution over the input.
	* Returns forward pass output. 
* __controller.py__
	* Controller agent creates hyperparameters for CNN, has Long Short-Term Memory (LSTM) network.
* __train.py__
	* Includes hyperparameters # of epochs, learning rate, batch size, # of hiden nodes etc.
	* Downloads train and test dataset, and passes it with the config parameters to the model.


## Training

Agent is a Neural Architecture Search cell, creates each epoch logits to configure CNN model architecture. Action tensor generated from pre-defined action space. In this solution, the agent predicts kernels, filters and paddings parameters of the CNN model. </ br> 
After training and evaluating CNN model, accuracy and entropy are calculated for optimize the agent. The agent gets accuracy value as a reward, thus it picks more accurate actions form action space. </br></br>   
To train the model in the paper, run this command:

```train
python train.py 
--use_cuda: True/False, default is False
--mode: (generate/evaluate), default is generate
--dataset: default is MNIST
```

## Results

Our model achieves the following performance on :

### [Neural Architecture Search with Reinforcement Learning](https://paperswithcode.com/paper/neural-architecture-search-with-reinforcement)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Refrence

@misc{https://doi.org/10.48550/arxiv.1611.01578,
			doi = {10.48550/ARXIV.1611.01578},  
			url = {https://arxiv.org/abs/1611.01578},  
			author = {Zoph, Barret and Le, Quoc V.},  
			keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},  
			title = {Neural Architecture Search with Reinforcement Learning},  
			publisher = {arXiv},  
			year = {2016},  
			copyright = {arXiv.org perpetual, non-exclusive license}
}
