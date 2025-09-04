# GP-WAITER
A deep-learning prediction model for crops.
# Usage
## Environment
### Hardware
GPU with 24GB and CPU
### Software
CUDA==11.(compatiable with pytorch)
python==1.12

Install nessasary library referring to requirements.txt:

`pip install -r requirements.txt`
## Demo
The demo file contains only a subset of phenotypes, SNP sites, and samples. Its purpose is solely to facilitate rapid testing and evaluate the usability of the model. Results obtained from this file do not reflect the model’s optimal performance and are provided for reference only.
``
The demo was trained on the provided sample file using an RTX3080 GPU, with an estimated training time of approximately 2 minutes.
## Model Training-Testing
Run training-testing.py file while import GP-WAITER model from `./model/GP-WAITER.py`. Then generate trained models and test the models on a test dataset.  
