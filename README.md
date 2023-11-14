# nSGC-MDA
nSGC-MDA is a model for miRNA-disease association prediction based on Simple Graph Convolution with DropMessage and Jumping Knowledge

# Dependencies
Our model is implemented by Python 3.6 with Pytorch 1.10.0
- pytorch==1.10
- numpy==1.16.6
- sklearn==0.24.2
- pandas=1.1.5
- dgl==0.6.1
- torch-geometric==2.0.3
- matplotlib==3.3.4

# Data

The data in this study are derived from the paper "Dai Q, Chu Y, Li Z, et al. MDA-CF: predicting miRNA-disease associations based on a cascade forest model by fusing multi-source information. ".

- `data/diseaseSim/DiseaseSimilarity.txt:`Disease semantic similarity
- `data/diseaseSim/D_GIP2.txt:`GIP similarity of diseases
- `data/miRNASim/Famsim.txt:`miRNA family similarity
- `data/miRNASim/Funcsim.txt:`miRNA target similarity
- `data/miRNASim/SeqSim2.txt:`miRNA sequence similarity
- `data/miRNASim/M_GIP.txt:`GIP similarity of miRNAs

# src
* The implementation of nSGC-MDA  
    ``model.py：`` The overall implementation code of the model        

    ``train.py：`` Setting of experimental parameters and training of the model    

    ``utils.py：``Data reading and construction of attribute bipartite graph

    ``GIP.py：``Calculate the GIP similarity of nodes
# Run

* The required packages can be installed by running `pip install -r requirements.txt`.
* python train.py
