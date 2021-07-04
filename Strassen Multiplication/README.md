# Strassen Multiplication in Java

Matrix Multiplication of A*B using Strassen's Method with Time-Complexity of θ(n<sup>log<sub>2</sub>(7)</sup>)

Time-Complexity: T(n) = 7T(n/2) + θ(n<sup>2</sup>)

The algorithm written here runs pretty fast, because no sub-matrices are copied. Instead, sub-matrices are represented with original matrices by varying the indices of row and column and supplying dimension size.

# Naive matrix multiplication

Naive matrix multiplication has run time of θ(n<sup>3</sup>), and is implemented with 3 for-loops. The Strassen algorithm written will default to run the naive matrix multiplication when n <= 64. The reason is that for small n, naive method will outperform Strassen method because of the overhead incurred. Feel free to experiment with different n.



## How to use
To run the `test` method which randomly generates 2 matrices of dimension 512x512, and compares the run-time of naive matrix multiplication vs Strassen multiplication:
1. `javac Strassen.java`
2. `java Strassen`

You can modify the parameter of `test(n, numberOfTimes)` to run your desired matrix dimension for size n, and how many times to re-run the matrix multiplication to get an average time spent.


To supply your own matrices:
1. Comment out line 318.
2. Uncomment line 319.
3. `javac Strassen.java`
4. `java Strassen < 1.input`

    
     The run() method takes in the following format:
     
     The first line consists of a single integer n, indicating that the matrices are n x n size.
     
     After which, n lines follow which represents the entries of the first matrix.
     
     Then, n lines follow which represents the entries of the second matrix.
     
     E.g. Input:
     ```
     2
     1 4
     3 2
     7 5
     9 8
     ```
     Output:
     ```
     43 37
     39 31
     ```
     
     
# Note
Only works with matrix of size power of 2. E.g. 256, 512, 1024 etc. Only works with matrix on INT value (does not check for INT overflow after multiplication). Single-threaded. 
       
