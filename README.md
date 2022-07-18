# PrefixSpanPar8

This is a parallel version of frequent sequence mining algorithm based on PrefixSpan. 
Algorithm uses GPU hardware acceleration in order to scan thousands of database rows at once, which allows for fast extraction of all frequent sequences.
Program was written using CUDA Toolkit and will only work with CUDA capable devices.

Algorithm can be launched from command line and requires the following three inputs: "name of database file", "name of output file" and "minimum support value". 
For example:
PrzefixSpanPar8 test3.txt out.txt 0.01

Name of database file – name of text file in which input database is stored. Each sequence in database must be placed in separate line. 
Each item in a sequence must be represented with single char from "a" to "z". Each element in a sequence must be separated with comma. 
Algorithm assumes that all items in single element are sorted alphabetically. Each sequence must end with a dot. 
Example sequence:
cfh,c,af,g,b,abdf,h,h.

Some example databases are also included in repository.

Name of output file – name of text file to which all frequent sequences will be written. Each line in this file contains frequent sequence and
value representing the number of database rows in which given sequence appears.

Minimum support value – value between 0 and 1. All sequences with support value greater or equal to minimum support are considered frequent. 
Support for given sequence is defined as a/b, where:
a - The number of rows in database containing given sequence
b – The total number of rows in database.
