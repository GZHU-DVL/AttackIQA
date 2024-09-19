# AttackIQA
**Black-box Adversarial Attacks Against Image Quality Assessment Models**     
*<a href="mailto:ranyu@e.gzhu.edu.cn">Yu Ran</a>, Ao-Xiang Zhang, Mingjie Li, Weixuan Tang, and Yuan-Gen Wang*          
Expert Systems with Applications(ESWA)  

## Setup
### Requirements  
* PyTorch 2.3.1  

### Datasets   
We evaluate the proposed method on LIVE, CSIQ, and TID2013 datasets.    
* In main.py, set the following three variables:  
```python
dataset_path_dict = {
    "LIVE": path to LIVE dataset,
    "CSIQ": path to CSIQ dataset,
    "TID": path to TID2013 dataset
}
```
* The *to_be_attack* folder includes *LIVE_sample.txt*, *CSIQ_sample.txt*, and *TID_sample.txt*, each containing 50 randomly sampled images for attack. To attack different images, please modify the contents of these .txt files.

### Models
We evaluate the proposed method on DBCNN, UNIQUE, TReS, and LIQE models.    
The *model_pt* folder contains the checkpoint files for these models. However, we only provide the checkpoint files for DBCNN and UNIQUE models due to repository size limitations. If users require the checkpoint files for TReS and LIQE, please contact the first author.

##  Running Attack
The following provides the arguments to run the attacks described in the paper.    

* Attack performance against DBCNN model on LIVE/CSIQ/TID dataset.
```bash
python main.py --dataset LIVE --model DBCNN --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset CSIQ --model DBCNN --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset TID --model DBCNN --n_queries 10000  --n_squares 2  --p_init 0.04
```
* Attack performance against UNIQUE model on LIVE/CSIQ/TID dataset.
```bash
python main.py --dataset LIVE --model UNIQUE --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset CSIQ --model UNIQUE --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset TID --model UNIQUE --n_queries 10000  --n_squares 2  --p_init 0.04
```
* Attack performance against TReS model on LIVE/CSIQ/TID dataset.
```bash
python main.py --dataset LIVE --model TReS --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset CSIQ --model TReS --n_queries 10000  --n_squares 2  --p_init 0.04
python main.py --dataset TID --model TReS --n_queries 10000  --n_squares 2  --p_init 0.04
```
* Attack performance against LIQE model on LIVE/CSIQ/TID dataset.
```bash
python main.py --dataset LIVE --model LIQE --n_queries 10000  --n_squares 8  --p_init 0.09
python main.py --dataset CSIQ --model LIQE --n_queries 10000  --n_squares 8  --p_init 0.09
python main.py --dataset TID --model LIQE --n_queries 10000  --n_squares 8  --p_init 0.09
```

## Result 
Two folders are created after the attack: *results* and *logs*   
Attack results are stored in *results* folder    
Attack logs are stored in *logs* folder    

## License
This source code is made available for research purposes only.     

## Acknowledgment
Our code is inspired by [**Square Attack**](https://github.com/max-andr/square-attack).


