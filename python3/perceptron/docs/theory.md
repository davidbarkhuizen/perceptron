# Theory

## Perceptron

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

## Theory

(Rosenblatt, 1958)

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




