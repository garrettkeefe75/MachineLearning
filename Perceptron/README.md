The HW3-Perceptron.py program allows you to set which version of Perceptron you want to use, the amount of epochs you want to calculated, and even the learning rate you wish to use.
An example run would look like
- python .\HW3-Perceptron.py Standard 10 0.1

Where we would use the Standard perceptron with 10 epochs and a learning rate of 0.1.

To use the other types of perceptron, you just change the first passed argument in the command line.
- python .\HW3-Perceptron.py Voted
- python .\HW3-Perceptron.py Averaged

The last two parameters are optional and do not have to be specified. The default epoch value is 10 and the default learning rate is 0.1.