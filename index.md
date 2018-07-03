## Improving L-BFGS Initialization For Trust Region Methods in Deep Learning

You can choose different method of initialization for L-BFGS matrices in form of 

<div>
$$
g_{\mu \nu} =  \left( \begin{array}{cccc}
- \alpha^{2} & \beta &  0  & 0 \\
\beta & a^{2} & 0 & 0 \\
0 & 0 & r^{2}  & 0 \\
0 & 0 & 0 & r^{2} \sin^{2} \theta
\end{array} \right) \;\;\;
\int_{-1}^{1} \frac{1}{\sqrt{1-x^{2}}} dx = \pi \;\;\;\;
1 < 2 \;\;\; 3 > 2
$$
</div>

B0 = gamma * I and run the trust region algorithm. The example here is using the classification task of MNIST dataset. 

TensorFlow is used to compute the gradients. Numpy and Scipy is used for the matrix computations. 
### Run the Python program

```shell
$ python L_BFGS_TR_init_B0.py -m=10 -minibatch=1000 -use-init-methods -B0-method=3

args:
-m=10             # the L-BFGS memory storage
-minibatch=1000   # minibatch size
-use-init-methods # default: False
-B0-method        # method 1, 2, 3 if -use-init-methods
```
