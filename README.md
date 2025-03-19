# BIRADS classification and feature identification in breast imaging
This repository contains the code I wrote for the 6th digix global ai challenge:[BI_RADS](https://www.saikr.com/vse/2024/DIGIX).
Final Ranking: 23rd out of 240.
Now I am open-sourcing the code, hoping to provide relevant assistance to those working in this field.  

## Getting Started  
### Prerequisites
Before running the code, ensure that you have a suitable Python environment with necessary dependencies installed. You can install the dependencies via:  
`pip install -r requirements.txt`  
### Dataset
If you want to conduct training, I recommend using publicly available open-source [datasets](https://www.kaggle.com/datasets/jimitdand/mammographic-mass-data-set-for-breast-cancer).  
### Training
To train feature model, use the following command:  
`python train_fea.py`    
To train classification model, use the following command:  
`python train_cla.py`  
**Note**:We use four expert models to achieve feature detection.  
### Predict
After training the model and using the correct model address, input the following code to make predictions.  
`python run.py`  
