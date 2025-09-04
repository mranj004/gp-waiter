# GP-WAITER
A deep-learning prediction model for crops.
# Usage
## Environment
### Hardware
GPU with 24GB cache and CPU
### Software
Ubuntu 20.04
CUDA==11.3(compatiable with pytorch)
python==1.12
Install nessasary library referring to requirements.txt:

`pip install -r requirements.txt`
 The install time is short and you needn't wait for a long time.
## Demo
The demo file contains only a subset of phenotypes, SNP sites, and samples. Its purpose is solely to facilitate rapid testing and evaluate the usability of the model. Results obtained from this file do not reflect the model’s optimal performance and are provided for reference only.
You can run the demo.script.py file under Demo path to help understand the training and testing process. 
`python3 demo.script.py`
When using the demo script, adjust the data paths as needed to ensure that the sample data is correctly loaded.
The demo was trained on the provided sample file using an RTX3080 GPU, with an estimated training time of approximately 2 minutes.
## Model Training-Testing
Run training-testing.py file while import GP-WAITER model from `./model/GP-WAITER.py`. Then generate trained models and test the models on a test dataset.  
