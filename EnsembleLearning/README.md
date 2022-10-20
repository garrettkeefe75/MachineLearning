To run the Adaboost algo run the below in the commandline
- python ./HW2-Adaboost.py
You can alter the number of iterations by appending a number to the commandline like so
- python ./HW2-Adaboost.py 10
Bagged Learning and Random forests can be run the same way
- python .\HW2-RandForests.py 10
- python .\HW2-Baggedlearning.py 10
Random Forest has an extra feature where you can also pass along the number of features 
want to pick at each node
- python .\HW2-RandForests.py 10 3
Where 3 is the number of features in each subset and 10 is the number of iterations