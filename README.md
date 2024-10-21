This is my implementation of the paper [Discrete Key Value Bottleneck](https://arxiv.org/abs/2207.11240), by Trauble, Goyal et al.

## Introduction and context

Deep neural networks perform exceedingly well on classification tasks where data streams are i.i.d. and labelled data is abundant. Challenges emerge in a production-level scenario where the data streams are non-stationary. 

One good approach is the fine-tuning paradigm: pre-train large encoderson volumes of readily available data followed by task-specific tuning. However, this approach faces challenges in that during the fine-tuning of a large number of weights, information about the previous task is lost in a process called **Catastrophic Forgetting**.

## Model

The authors build upon a discrete key-value bottleneck containing a number of **separate, *learnable* key-value pairs**. The paradigm followed is 
$$ \text{Encode} \rightarrow \textbf{BOTTLENECK} \rightarrow \text{Decode} .$$

![*The model architecture proposed in Discrete Key Value Bottleneck*](https://i.imgur.com/WBQiBfi.png)

The input is fed to a pre-trained encoder, the output of the encoder is used to select the nearest keys, and the corresponding values are fed to the decoder to solve the task.

### Detailed description

**GOAL:** To learn a model $f_\theta : \mathcal X \rightarrow \mathcal Y$ from training data $S = ((x_i, y_i)_{i=1}^n)$ that is robust to *strong* input distribution changes.

Let the model be formulated as $$f_\theta = d_\delta \circ v_\gamma \circ k_\beta \circ z_\alpha$$

In the first step an input is fed to the encoder $z_\alpha : \mathcal X \rightarrow \mathcal Z \in \mathbb R^m$ extracts a representational embedding from the high-dimensional observation $x$. We further project this representation into C lower-dimensional feature heads, each of them being passed as input into a **separate head-specific learnable** key-value codebook. This projection is done using C **fixed** Gaussian Random projection matrices. If $x$ is sufficiently low-dimensional then we can skip encoding an partition $x$ directly.

A **KEY-VALUE CODEBOOK** is a bijection that maps each code vector to a different value vector which is learnable. Within each codebook, a quantization process $k_\beta$ selects the closest key to its head-specific input.

For the purpose of classification the suthors propose a simple non-parametric decoder function which uses average-pooling to calculate the element-wise average of all the fetched value codes and then applies a softmax on top of it.

## A SIMPLE LEARNING SETTING
We perform a simple eight-class classification task in a class-incremental manner to show the efficacy of the bottleneck. In each stage, we sample 100 examples of two classes for 1000 training steps, using gradient descent to update the weights, then move on to two new classes for the next 1000 steps. The input features of each class follow spatially separated normal distributions:
![](https://i.imgur.com/fVhROlp.png)

The results are clear: the naive models based on linear probes and a simple multi-layer perceptron simply overfit on the most recent training data, thus forgetting the previously learned information. However, at each step the Discrete Key Value Bottleneck holds on to the previous information while also learning new ones.
 
![](https://i.imgur.com/VXKxuaG.png)
