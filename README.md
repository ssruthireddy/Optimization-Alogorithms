# Optimization-Algorithm’s
Optimization Algorithm’s to reduce Error: How you should change your weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use. Optimization algorithms or strategies are responsible for reducing the losses and to provide the most accurate results possible.

Loss function: Error function used for back propagation
Activation function: A typical (non-linear) function used for output neuron
Iteration: when a weight update is done, it could be after every training example, or after the entire training set, or after batch of training example.
Epoch: training set has been passed through the network once to update weights (ex: we have to run training for many such epochs to train deep networks) 
Learning state (α): size of the step in the direction of negative gradient

Different types of optimizers and their advantages:

Algorithmic Approaches:
3 variants of using gradient descent to update network weights: Batch, Stochastic, Mini-Batch
Ex: 10000 observations
  Batch Gradient descent (BGD): updating weights just once per epoch, after all samples are evaluated. BGD is an offline algorithm. It strictly works from information that is stored. 1 epoch has 1 Iteration.

Gradient Descent is the most basic optimization algorithm. It’s used heavily in linear regression and classification algorithms. Back propagation in neural networks also uses a gradient descent algorithm.

Gradient descent is a first-order optimization algorithm which is dependent on the first order derivative of a loss function. It calculates that which way the weights should be altered so that the function can reach a minimal. Through back propagation, the loss is transferred from one layer to another and the model’s parameters also known as weights are modified depending on the losses so that the loss can be minimized.

  Stochastic Gradient descent (SGD): update the weights after every sample. Stochastic (Random) shuffle the training set. Online algorithm,  because it doesn’t require sample to be stored. 1 epoch has 10000 iterations
Advantages:
Frequent updates of model parameters hence, converges in less time.
Requires less memory as no need to store values of loss functions.
May get new minima’s.
Disadvantages:
High variance in model parameters.
May shoot even after achieving global minima.
To get the same convergence as gradient descent needs to slowly reduce the value of learning rate. 

  Min-batch (SGD): update weights after some fixed number of samples have been evaluated. Mini batch size is always power of 2. Within the mini-batch randomly chooses the values.  Batch size is 1000, 1 epoch has 10 iterations. Total no of observations/batch size10000/1000

Min-batch (SGD) challenges: one rule applies to all weights
-	Need to mention the value of learning rate (η): too low learning rate can result in long learning times and can get stuck in shallow local minima.
-	Learning rate (η): too high can cause us to overshoot deep local minima, and get stuck at bouncing around inside a minimum when we find it.
-	Avoid the problem using decay schedule (Exponential, Step…) to change learning rate (η) over time we still have to pick that schedule and its parameters.
-	Until now whatever technique it is like decay schedules it applies to all the weights.
-	Update weights with a “one update rate fits all” approach.

Advantages:
1.	Frequently updates the model parameters and also has less variance.
2.	Requires medium amount of memory.
All types of Gradient Descent have some challenges:
1.	Choosing an optimum value of the learning rate. If the learning rate is too small than 
gradient descent may take ages to converge.
2.	Have a constant learning rate for all the parameters. There may be some parameters which we may not want to change at the same rate.
3.	May get trapped at local minima.

To overcome challenges there are variants of Min-batch (SGD): Every weight there is change
Decay Parameter - Shrinking Learning rate.
Shrinking learning rate : After certain number of epochs model will stop learning. Because model assume to stage where there is no change. This is called as stuck at plateau, gradient vanish and tangent line is flat -------,slope = 0, to give that slight push mathematically it’s called as momentum.

  Momentum: Get over flat plateaus. To update weight it looks at previous rate of change and multiply with factor called ϒgamma = 0.9. Since you have applied previous ROC(rate of change) movement is more.
 Momentum: Momentum was invented for reducing high variance in SGD and softens the convergence. It accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction. One more hyperparameter is used in this method known as momentum symbolized by ‘γ’. Now, the weights are updated by θ=θ−V(t).
The momentum term γ is usually set to 0.9 or a similar value.
Advantages:
1.	Reduces the oscillations and high variance of the parameters.
2.	Converges faster than gradient descent.
Disadvantages:
1.	One more hyper-parameter is added which needs to be selected manually and accurately.

  Nestrov Momentum: 
Momentum may be a good method but if the momentum is too high the algorithm may miss the local minima and may continue to rise up. So, to resolve this issue the NAG algorithm was developed. It is a look ahead method. We know we’ll be using γV(t−1) for modifying the weights so, θ−γV(t−1) approximately tells us the future location. Now, we’ll calculate the cost based on this future parameter rather than the current one.
V(t)=γV(t−1)+α. ∇J( θ−γV(t−1) ) and then update the parameters using θ=θ−V(t).

Advantages:
1.	Does not miss the local minima.
2.	Slows if minima’s are occurring.
Disadvantages:
1.	Still, the hyper parameter needs to be selected manually.

Adaptive Techniques: AdaGrad  ,AdaDelta, RMSProp,  Adam – each weight has different learning rate.

  Adagrad: This optimizer changes the learning rate. It changes the learning rate ‘η’ for each parameter and at every time step ‘t’. It’s a type second order optimization algorithm. It works on the derivative of an error function.
Advantages:
1.	Learning rate changes for each training parameter.
2.	Don’t need to manually tune the learning rate.
3.	Able to train on sparse data.
Disadvantages:
1.	Computationally expensive as a need to calculate the second order derivative.
2.	The learning rate is always decreasing results in slow training.

  AdaDelta: It is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of accumulating all previously squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size w. In this exponentially moving average is used rather than the sum of all the gradients.
Advantages:
1.	Now the learning rate does not decay and the training does not stop.
Disadvantages:
2.	Computationally expensive.

  RMSProp
  Adam: Adam (Adaptive Moment Estimation) works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search. In addition to storing an exponentially decaying average of past squared gradients like AdaDelta, Adam also keeps an exponentially decaying average of past gradients M(t).
M(t) and V(t) are values of the first moment which is the Mean and the second moment which is the uncentered variance of the gradients respectively.

Advantages:
1.	The method is too fast and converges rapidly.
2.	Rectifies vanishing learning rate, high variance.
Disadvantages:
1.	Computationally costly.

Conclusions
Adam is the best optimizers. If one wants to train the neural network in less time and more efficiently than Adam is the optimizer.
For sparse data use the optimizers with dynamic learning rate.
If, want to use gradient descent algorithm than min-batch gradient descent is the best option.







  
