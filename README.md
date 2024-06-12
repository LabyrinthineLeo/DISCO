### DISCO: A Hierarchical Disentangled Cognitive Diagnosis Framework for Interpretable Job Recommendation
The code in this repository is the implementation of the proposed DISCO framework, where one of the datasets used Edu-Rec is publicly available, details of which can be found in the paper description.

#### DISCO Description

we propose a novel framework termed as DISCO (a hierarchical Disentangling based Cognitive diagnosis framework), which aims to flexibly accommodate the underlying representation learning model for job recommendations. Our approach comprises several key components. Initially, we designed a hierarchical representation disentangling module to mine the hierarchical skill-related factors embedded in the representations of job seekers and jobs. To further enhance information communication and robust representation learning, we proposed the level-aware association modeling, which consists of the inter-level knowledge influence module and level-wise contrastive learning. we devised an interaction diagnosis module is introduced that integrates a neural diagnosis function, aimed at effectively capturing the multi-level recruitment interaction process between job seekers and jobs. Finally, we developed an interaction diagnosis module incorporating a neural diagnosis function for effectively modeling the multi-level recruitment interaction process between job seekers and jobs, which introduces the cognitive measurement theory. Extensive experiments on two real-world recruitment recommendation datasets and an educational recommendation dataset clearly demonstrate the effectiveness and interpretability of our proposed DISCO framework. 

#### Dependencies:
* python==3.8
* pytorch==1.10
* numpy
* pandas
* scipy
* scikit-learn

#### Usage

Download the public dataset and preprocessing according to the detailed description in the paper, and place it in the directory `./datas/{dataset_edu-rec}`.

Train & Test model:
```
cd ./train
python3 train_ngcf.py --model_name {model_name} --dataset_name {dataset_name} --lr {lr} --batch_size {batch_size} --epoch {epoch}
```
For example:
```
python3 train_ngcf.py --model_name "ngcf_disco" --dataset_name "edu_rec" --lr 8e-4 --batch_size 512 --epoch 80 
```
