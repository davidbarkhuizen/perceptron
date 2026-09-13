# Perceptron Theory

## External References

Shree Nayar

Computer Science Dept, School of End & Applied Sciences, Columbia Universitry 
https://fpcv.cs.columbia.edu/

First Principles of Computer Vision Course
- https://fpcv.cs.columbia.edu/
- https://www.youtube.com/@firstprinciplesofcomputerv3258 

Perceptron | Neural Networks
- https://www.youtube.com/watch?v=OFbnpY_k7js

## Configuration

nand gate
w = [-2, 2]
q = 3

# Summary of Perceptron Theory as per Rosenblatt, 1958

given

input x_  
- vector   
- n inputs  
- x1 .. xn  

input weights w_  
- vector  
- 1 weight for each input  
- w1 .. wn  
- w_  

bias b (activation threshold)
- scalar  
- (aggregate) activation threshold  

activation function, f
= | 0     IFF   w_.x_ <= -b 
  | 1     IFF   w_.x_ > -b

then, if we define z = w_.x_ + b 

then we can write the activation, a, as a function of z:

activation, a
= f(z)
= | 0   IFF     z <= 0
  | 1   IFF     z > 0

i.e the activaton function of the Perceptron neuron is step function

For a single neuron, Rosenblatt's perceptron convergence theorem guarantees the update rule
above finds a separating hyperplane in a finite number of steps, provided the training data
is linearly separable.

## Multi-unit (cardinality > 1) networks

`LinearClassifierNetwork` combines `cardinality` perceptron neurons (the hidden layer) via a
fixed output neuron that ANDs their activations, so its decision region is the *intersection*
of `cardinality` half-planes - a convex polytope, not necessarily a single hyperplane.

Training this can't just apply the single-neuron update rule to every hidden neuron with the
same target label: that gives the neurons no way to specialise into different half-planes, so
they'd all just redundantly converge toward the same one. Instead, on a misclassification,
only the single hidden neuron closest to flipping (smallest |z|) among those responsible for
the error is updated - a minimum-disturbance rule in the spirit of Widrow's MADALINE
("Multiple ADALINE").

Unlike the single-neuron case, this has no convergence guarantee analogous to Rosenblatt's
theorem - there's no proof it finds a matching set of hyperplanes in finite steps, or at all,
for an arbitrary target polytope. In practice it converges well for modest cardinality (see
`tests/test_networks.py`'s cardinality=2 case), but that's an empirical observation, not a
theorem.


