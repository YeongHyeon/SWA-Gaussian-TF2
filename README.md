[TensorFlow 2] A Simple Baseline for Bayesian Uncertainty in Deep Learning: SWA-Gaussian (SWAG)
=====
TensorFlow implementation of "A Simple Baseline for Bayesian Uncertainty in Deep Learning"

## Concept
<div align="center">
  <img src="./figures/algorithm.png" width="750">  
  <p>Algorithm to utilize the SWAG [1].</p>
</div>

<div align="center">
  <img src="./figures/swag_sampling.png" width="750">  
  <p>Equation for the weight sampling from SWAG [1].</p>
</div>

## Results
<div align="center">
  <img src="./figures/smce.svg" width="450">  
  <p>Loss convergence within training procedure.</p>
</div>

<div align="center">
  <img src="./figures/weights.gif" width="400">  
  <img src="./figures/weights_to_theta1.gif" width="400">
  <p>Histogram change of &theta; without stochastic weight averaging and difference between &theta; and &theta;_SWA.</p>
</div>

<div align="center">
  <img src="./figures/theta_1.gif" width="400">  
  <img src="./figures/theta_2.gif" width="400">  
  <p>Histogram change of &theta;_SWA and &sum;_diag sequentially. The red color and the blue color represent the initial state and current state respectively.</p>
</div>

## Performance
|Method|Accuracy|Precision|Recall|F1-Score|
|:---|:---:|:---:|:---:|:---:|
|Final Epoch|0.99230|0.99231|0.99222|0.99226|
|Best Loss|0.99350|0.99350|0.99338|0.99344|
|SWAG (S = 30)|0.99310|0.99305|0.99299|0.99302|
|SWAG (Last Momentum)|0.99340|0.99340|0.99330|0.99335|

## Requirements
* Python 3.7.6  
* Tensorflow 2.3.0  
* Numpy 1.18.15
* whiteboxlayer 0.1.13

## Reference
[1] Wesley Maddox et al. (2019). <a href="https://arxiv.org/abs/1902.02476">A Simple Baseline for Bayesian Uncertainty in Deep Learning</a>. arXiv preprint arXiv:1902.02476.
