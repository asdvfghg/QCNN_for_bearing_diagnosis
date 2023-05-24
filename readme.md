# Attention-embedded Quadratic Network (Qttention) for Effective and Interpretable Bearing Fault Diagnosis
This is the offical repository of the paper "Attention-embedded Quadratic Network (Qttention) for Effective and Interpretable Bearing Fault Diagnosis" in IEEE Transactions on Instrumentation and Measurement [https://ieeexplore.ieee.org/abstract/document/10076833](https://ieeexplore.ieee.org/abstract/document/10076833).

In this work,

1. We propose a simple and effective model that is made of quadratic neurons for bearing fault diagnosis. The quadratic neurons directly augment the model to outperform other state-of-the-arts in noisy data settings. Different from previous structural modifications, our innovation is at the neuronal level, applying different neurons to bearing fault diagnosis and upgrading CNNs-based bearing fault diagnosis models. 
2. Methodologically, we derive the qttention mechanism from quadratic neurons, which greatly facilitates the understanding to the model made of quadratic neurons, with an emphasis on the model's physics mechanism when combined with envelope spectrum analysis.



All experiments are conducted with Windows 10 on an Intel i9 10900k CPU at 3.70 GHz and one NVIDIA RTX 3080Ti 12GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## Citing
If you find this repo useful for your research, please consider citing it:
```
@ARTICLE{10076833,
  author={Liao, Jing-Xiao and Dong, Hang-Cheng and Sun, Zhi-Qi and Sun, Jinwei and Zhang, Shiping and Fan, Feng-Lei},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Attention-Embedded Quadratic Network (Qttention) for Effective and Interpretable Bearing Fault Diagnosis}, 
  year={2023},
  volume={72},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2023.3259031}}
```



## Quadratic Convolutional Neural Network (QCNN)

### Quadratic Neurons
A quadratic neuron was proposed by [1], It computes two inner products  and  one  power  term  of  the  input  vector  and  integrates them for a nonlinear activation function. The output function of a quadratic neuron is expressed as 

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/小书匠/1641001696385.png),

where $\sigma(\cdot)$ is a nonlinear activation function, $\odot$ denotes the Hadamard product, $\boldsymbol{w}^r,\boldsymbol{w}^g, \boldsymbol{w}^b\in\mathbb{R}^n$ are weight vectors, and $b^r, b^g, c\in\mathbb{R}$ are biases. When $\boldsymbol{w}^g=0$, $b^g=1$, and $\boldsymbol{w}^b=0$, a quadratic neuron degenerates to a conventional neuron:  $\sigma(f(\boldsymbol{x}))= \sigma(\boldsymbol{x}^\top\boldsymbol{w}^{r}+b^{r})$. 

### QCNN Structure
We propose a quadratic neuron empowered 1DCNN for bearing fault diagnosis. Specifically, the QCNN substitutes the conventional convolution with the quadratic convolution operations in convolutional layers. Structurally, we inherit the structure of the WDCNN [2] as the structure of the proposed model because the WDCNN is a well-established model whose design has been considered by many follow-up studies. This structure stacks 6 CNN blocks and 1 fully-connected layer.
![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/1666079629967.png)

### Qttention
We find that a quadratic neuron also contains an attention mechanism, referred to as *qttention*, by factorizing the learned quadratic function in analogue to attention. Mathematically, we factorize the expression of a quadratic neuron as follows (we remove constant terms for conciseness):

![](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/qttention1.png)

![](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/qttention2.png)

![](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/qttention3.png)

Such that, we can recover the qttention map of each layer through the above equations. The workflow of qttention map are shown as follows.
![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/qttention.png)

## Repository organization

### Requirements
We use PyCharm 2021.2 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.8
* PyTorch == 1.10.1
* CUDA == 11.3 if use GPU
* wandb == 0.12.11
* anaconda == 2021.05
 
### Organization
```
QCNN_for_bearing_diagnosis
│   train.py # training QCNN and WDCNN
│   qttention.py # generating qttention map
└─  data # bearing fault datasets 
     │   
     └─ 0HP # example dataset
	 │ xxx.mat
└─  utils
     │   data_split.py # spliting dataset to train set and test set, then add noise to the raw signal. 
     │   DatasetLoader.py # dataloder class for pytorch
     │   preprocess.py # preprocessing signal
     │   train_function.py # Train function for quadratic network
└─  Model
     │   ConvQuadraticOperation.py # the quadratic convolutional neuron function 
     │   QCNN.py # QCNN model
     │   WDCNN.py # WDCNN model
```

### Datasets
We use the CWRU dataset [3] and HIT dataset in our article. The CWRU dataset is a public bearing fault dataset  that can be found in [CWRU Dataset](https://github.com/s-whynot/CWRU-dataset).

### How to Use

Our deep learning models use **Weight & Bias** for training and fine-tuning. It is the machine learning platform for developers to build better models faster. [Here](https://docs.wandb.ai/quickstart) is a quick start for Weight & Bias. You need to create an account and install the CLI and Python library for interacting with the Weights and Biases API:
```
pip install wandb
```
Then login 
```
wandb login
```

After that, you can run our code for training.

1. For noisy data, you need to generate train dataset and test dataset by running ```data_split.py```.

2. Run ```train.py``` to train a QCNN or WDCNN model. First, you need to fill in your username in the wandb initialization function:
	```
	wandb.init(project="bearinganomaly", entity="username")
	```
	The results will be saved to your **Weight & Bias** project online.  

3. Run ```qttention.py``` to show the qttention map of the QCNN. The results will be saved to the ***'data'*** folder.
 


## Main Results
### Classification Performance
Here we give the main results of our paper. We use the false positive rate (FPR), fault detection rate (FDR), presicion (PRE) and F1 score to validate the performance of the proposed method. All results are run 5 times to calculate the average. The proposed method outperforms other compared baseline methods.

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/1684933742120.png)


All the baseline methods we utilize their official implement as follows:

RNN-WDCNN [4]: https://github.com/al3xsh/rolling-element-bearing-fault-detection

AResNet [5]: https://github.com/dongfangxiaotian/CWRU-AttentionMechanism-ResNet

MA1DCNN [6]: https://github.com/erphm/MA1DCNN

DCA-BiGRU [7]:https://github.com/liguge/Fault-diagnosis-for-small-samples-based-on-attention-mechanism

WDCNN [2]: https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis

### Initial Settings and Hyperparameters
First, we summarise the techniques and hyperparameters used in our experiments. Since the official codes of baseline methods use different training strategies, we keep their strategies and only fine-tune their hyperparameters.
![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/setings.png)

Second, initial hyperparameters of all baseline methods on test dataset. lr denotes the learning rate. For the QCNN, we use so called ReLinear algorithm for training [8]. A normal learning rate $\gamma_r$ is cast for $(\boldsymbol{w}^r, b^r)$, and a relatively small learning rate $\gamma_{g,b}$ for $(\boldsymbol{w}^g, b^g,  \boldsymbol{w}^b,c)$. We use a scale facor $\alpha$ to control the ratio of $\gamma_r$ and $\gamma_{g,b}$, that is, $\gamma_r = \alpha \cdot \gamma_{g,b}$.

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/hyper.png)
### Interpretable Qttention Maps
The qttention maps derived from the QCNN inference process. Note that the QCNN pays attention to significant fault-related vibration signals.
![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/qttentionmap.png)

## Contact
If you have any questions about our work, please contact the following email address:

jingxiaoliao@hit.edu.cn

Enjoy your coding!
## Reference
[1] Fenglei Fan, Wenxiang Cong, and Ge Wang. A new type of neurons for machine learning. International journal for numericalmethods in biomedical engineering, 34(2):e2920, 2018.

[2]	Zhang, W., Peng, G., Li, C., Chen, Y., & Zhang, Z. (2017). A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals. Sensors, 17(2), 425.

[3] https://csegroups.case.edu/bearingdatacenter/pages/download-data-file

[4]	Shenfield, A., & Howarth, M. (2020). A novel deep learning model for the detection and identification of rolling element-bearing faults. Sensors, 20(18), 5112.

[5]  X. Zhong, F. Wang, and H. Ban, Development of a plug-and-play anti-noise module for fault diagnosis of rotating machines in nuclear power plants.

[6] Wang, H., Liu, Z., Peng, D., & Qin, Y. (2019). Understanding and learning discriminant features based on multiattention 1DCNN for wheelset bearing fault diagnosis. IEEE Transactions on Industrial Informatics, 16(9), 5735-5745.

[7] Zhang, X., He, C., Lu, Y., Chen, B., Zhu, L., & Zhang, L. (2022). Fault diagnosis for small samples based on attention mechanism. Measurement, 187, 110242.

[8] Feng-Lei Fan, Mengzhou Li, Fei Wang, Rongjie Lai, and Ge Wang. Expressivity and trainability of quadratic networks. arXiv preprint, 2021.
