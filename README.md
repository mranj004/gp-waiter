# GP-WAITER
GP-WAITER (Genome-Phenotype prediction using Weighted self-AttentIon TransformER) is a novel model that integrates GWAS-derived SNP weights into a hybrid CNN-Transformer architecture via an efficient tokenized embedding scheme. The model's design enables dynamic learning of trait-associated feature weights and effective modeling of long-range interactions within ultra-long genomic sequences.
# Usage
## Environment
### Hardware
GPU with 24GB cache and CPU
### Software
Ubuntu 20.04
CUDA==11.3(compatiable with pytorch)
python==1.12
Install nessasary library referring to requirements.txt:
'torch==1.12
tensorboard
numpy
pandas
scikit-learn'

`pip install -r requirements.txt`

 The install time is short and you needn't wait for a long time.
## Demo
The demo file contains only a subset of phenotypes, genotype, weighted information. Its purpose is solely to facilitate rapid testing and evaluate the usability of the model. Results obtained from this file do not reflect the model’s optimal performance and are provided for reference only.
You can run the demo.script.py file under Demo path to help understand the training and testing process.

Running instruction:

`python3 demo.script.py`

When using the demo script, modify the data paths as needed to ensure that the sample data is correctly loaded.
After Running the demo, output best model parameters, prediction accuracy and training results for all epoches.
The demo was trained on the provided sample file using an RTX3080 GPU, with an estimated training time of approximately 2 minutes.
## Model Training-Testing
Running training-testing.py file while importing GP-WAITER model from `./model/GP-WAITER.py`. Then generate trained models and test the models on a test dataset.  
