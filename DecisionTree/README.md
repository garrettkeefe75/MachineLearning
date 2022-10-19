To use the decision tree algorithm, run the following command in the command line
 - python ./HW1-ID3.py
To change what is being tested, you can alter the code to use attributes set 1, 2, 3, 4.
When using the ID3 call, you can specify which of the variations you want to use by passing "ME", "GI", or "Entropy" into the last parameter.
an example ID3 call would look like ID3(set, attributes, label, depth, variation, weights) or ID3(S, A, L, 10, "GI", [1/len(S)]*len(S))