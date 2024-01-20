# Perceptron

## installation

poetry add --dev pytest@latest 

## configuration

nand gate
w = [-2, 2]
q = 3

## theory

input x_  
- vector   
- n inputs  
- x1 .. xn  

input weights w_  
- vector  
- 1 weight for each input  
- w1 .. wn  
- w_  

bias b
- scalar  
- (aggregate) activation threshold  

activation function, f
= 0     if  w_.x_ <= -b 
= 1     if  w_.x_ > -b


output