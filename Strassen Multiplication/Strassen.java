/*
    Matrix Multiplication of A*B using Strassen's Method with Time-Complexity of Theta(n^log2(7))
    Time-Complexity: T(n) = 7 T(n/2) + Theta(n^2)

    This program runs pretty fast as sub-Matrices are not copied, but rather, represented as the Matrix Class by varying the indices.
    Note: This program only works with square Matrices of dimension power of 2. E.g. 64, 128, 512, 1024, etc. Single-threaded. Works only for INT value.

    Some time test according to my CPU:
    Naive method:
        - Matrix n=512: 324ms
        - Matrix n=1024: 4092ms
        - Matrix n=2048: 83400ms
    Strassen method:
        - Matrix n=512: 170ms
        - Matrix n=1024 1050ms
        - Matrix n=2048: 8320ms

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
    OF THE POSSIBILITY OF SUCH DAMAGE.

    Author: Gan Chin Yao
 */

import java.time.Duration;
import java.time.Instant;
import java.util.Random;
import java.util.Scanner;

public class Strassen {
    // A Matrix is represented with matr[][] int-array, with legal values ranging from matr[mRow TO mRow + dimension] and matr[mCol TO mCol + dimension].
    // This is so that we can represent sub-Matrices using the same original Matrix without copying the values by simply changing mRow and mCol index, hence this runs way faster.
    class Matrix {
        private int[][] matr;
        private int mRow; // Start index of the Row
        private int mCol; // Start index of the Column
        private int dimension; // Size dimension by dimension of the matrix representation

        public Matrix(int matr[][], int mRow, int mCol, int dimension) {
            this.matr = matr;
            this.mRow = mRow;
            this.mCol = mCol;
            this.dimension = dimension;
        }

        // Dimension is the actual dimension of the Matrix represented. It need not be the size of the matr[][] stored.
        // For example, if the int[][] stores 512 x 512, but you only want to represent the first quarter of the matrix, the dimension would be 128 x 128.
        public int getDimension() {
            return this.dimension;
        }

        // row and col starts from 0 to dimension.
        // This method takes care of the offset.
        public int getElement(int row, int col) {
            return matr[mRow + row][mCol + col];
        }

        public int[][] getMatr() {
            return this.matr;
        }

        public int getmRow() {
            return this.mRow;
        }

        public int getmCol() {
            return this.mCol;
        }
    }


    /**
     * Multiply two matrices by Strassen's Method.
     *
     * @param m1 A 2D integer array of size n x n.
     * @param m2 A 2D integer array of size n x n.
     * @return
     */
    public int[][] strassen(int m1[][], int m2[][]) {
        return strassenHelper(m1, m2, 0, 0, 0, 0, m1.length).getMatr();
    }

    /**
     * Helper method for strassen method.
     *
     * @param m1        The first Matrix of size dimension x dimension
     * @param m2        The second Matrix of size dimension x dimension
     * @param m1Row     Integer index to start reading the row of m1.
     * @param m1Col     Integer index to start reading the column of m1.
     * @param m2Row     Integer index to start reading the row of m2.
     * @param m2Col     Integer index to start reading the column of m2.
     * @param dimension The size of the square Matrix m1 and m2.
     * @return A Matrix representation of the product of m1 and m2.
     */
    private Matrix strassenHelper(int m1[][], int m2[][], int m1Row, int m1Col, int m2Row, int m2Col, int dimension) {
        // Use the naive method when size is less than 64, because the naive method will be faster for small n.
        // You can experiment with different size value and observe the speed difference.
        if (dimension <= 64) {
            return naiveMatrixMult(m1, m2, m1Row, m1Col, m2Row, m2Col, dimension);
        }

        int halfN = dimension / 2;

        /*
         * Matrices are labelled in this order:
         *             m1:         m2:
         *          | a   b |   | e   f |
         *          | c   d |   | g   h |
         */

        // Create sub-Matrix representation by simply changing index value of original Matrix. No Matrix copying takes place.
        Matrix a = new Matrix(m1, m1Row, m1Col, halfN);
        Matrix b = new Matrix(m1, m1Row, m1Col + halfN, halfN);
        Matrix c = new Matrix(m1, m1Row + halfN, m1Col, halfN);
        Matrix d = new Matrix(m1, m1Row + halfN, m1Col + halfN, halfN);
        Matrix e = new Matrix(m2, m2Row, m2Col, halfN);
        Matrix f = new Matrix(m2, m2Row, m2Col + halfN, halfN);
        Matrix g = new Matrix(m2, m2Row + halfN, m2Col, halfN);
        Matrix h = new Matrix(m2, m2Row + halfN, m2Col + halfN, halfN);

        Matrix f_minus_h = matrixMinus(f, h);
        Matrix a_plus_b = matrixAdd(a, b);
        Matrix c_plus_d = matrixAdd(c, d);
        Matrix g_minus_e = matrixMinus(g, e);
        Matrix a_plus_d = matrixAdd(a, d);
        Matrix e_plus_h = matrixAdd(e, h);
        Matrix b_minus_d = matrixMinus(b, d);
        Matrix g_plus_h = matrixAdd(g, h);
        Matrix a_minus_c = matrixMinus(a, c);
        Matrix e_plus_f = matrixAdd(e, f);
        Matrix P1 = strassenHelper(a.getMatr(), f_minus_h.getMatr(), a.getmRow(), a.getmCol(), f_minus_h.getmRow(), f_minus_h.getmCol(), halfN);
        Matrix P2 = strassenHelper(a_plus_b.getMatr(), h.getMatr(), a_plus_b.getmRow(), a_plus_b.getmCol(), h.getmRow(), h.getmCol(), halfN);
        Matrix P3 = strassenHelper(c_plus_d.getMatr(), e.getMatr(), c_plus_d.getmRow(), c_plus_d.getmCol(), e.getmRow(), e.getmCol(), halfN);
        Matrix P4 = strassenHelper(d.getMatr(), g_minus_e.getMatr(), d.getmRow(), d.getmCol(), g_minus_e.getmRow(), g_minus_e.getmCol(), halfN);
        Matrix P5 = strassenHelper(a_plus_d.getMatr(), e_plus_h.getMatr(), a_plus_d.getmRow(), a_plus_d.getmCol(), e_plus_h.getmRow(), e_plus_h.getmCol(), halfN);
        Matrix P6 = strassenHelper(b_minus_d.getMatr(), g_plus_h.getMatr(), b_minus_d.getmRow(), b_minus_d.getmCol(), g_plus_h.getmRow(), g_plus_h.getmCol(), halfN);
        Matrix P7 = strassenHelper(a_minus_c.getMatr(), e_plus_f.getMatr(), a_minus_c.getmRow(), a_minus_c.getmCol(), e_plus_f.getmRow(), e_plus_f.getmCol(), halfN);

        Matrix r = matrixAdd(matrixMinus(matrixAdd(P5, P4), P2), P6);
        Matrix s = matrixAdd(P1, P2);
        Matrix t = matrixAdd(P3, P4);
        Matrix u = matrixMinus(matrixMinus(matrixAdd(P5, P1), P3), P7);

        int[][] result = new int[dimension][dimension];

        // Assign values into result array
        for (int i = 0; i < halfN; i++) {
            for (int j = 0; j < halfN; j++) {
                result[i][j] = r.getElement(i, j);
            }
        }

        for (int i = 0; i < halfN; i++) {
            for (int j = halfN; j < dimension; j++) {
                result[i][j] = s.getElement(i, j - halfN); // Minus halfN because .getElement() already took care of the offset.
            }
        }

        for (int i = halfN; i < dimension; i++) {
            for (int j = 0; j < halfN; j++) {
                result[i][j] = t.getElement(i - halfN, j);
            }
        }

        for (int i = halfN; i < dimension; i++) {
            for (int j = halfN; j < dimension; j++) {
                result[i][j] = u.getElement(i - halfN, j - halfN);
            }
        }

        return new Matrix(result, 0, 0, dimension);
    }

    /**
     * O(n^3) Naive Matrix Multiplication by using 3 For-loops.
     *
     * @param m1        The first Matrix of size dimension x dimension
     * @param m2        The second Matrix of size dimension x dimension
     * @param m1Row     Integer index to start reading the row of m1.
     * @param m1Col     Integer index to start reading the column of m1.
     * @param m2Row     Integer index to start reading the row of m2.
     * @param m2Col     Integer index to start reading the column of m2.
     * @param dimension The size of the square Matrix m1 and m2.
     * @return A matrix representation of the product of m1 and m2.
     */
    private Matrix naiveMatrixMult(int m1[][], int m2[][], int m1Row, int m1Col, int m2Row, int m2Col, int dimension) {
        int[][] product = new int[dimension][dimension];
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                product[row][col] = 0;
                for (int k = 0; k < dimension; k++) {
                    product[row][col] += m1[row + m1Row][k + m1Col] * m2[k + m2Row][col + m2Col];
                }
            }
        }
        return new Matrix(product, 0, 0, dimension);
    }

    // m1 - m2
    private Matrix matrixMinus(Matrix m1, Matrix m2) {
        int dimension = m1.getDimension();
        int[][] result = new int[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                result[i][j] = m1.getElement(i, j) - m2.getElement(i, j);
            }
        }
        return new Matrix(result, 0, 0, dimension);
    }

    // m1 + m2
    private Matrix matrixAdd(Matrix m1, Matrix m2) {
        int dimension = m1.getDimension();
        int[][] result = new int[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                result[i][j] = m1.getElement(i, j) + m2.getElement(i, j);
            }
        }
        return new Matrix(result, 0, 0, dimension);
    }

    /**
     * The run() method takes in the following format:
     * The first line consists of a single integer n, indicating that the matrix are n x n size.
     * After which, n lines follow which represents the entries of the first matrix.
     * Then, n ines follow which represents the entries of the second matrix.
     * <p>
     * E.g. Input:
     * 2
     * 1 4
     * 3 2
     * 7 5
     * 9 8
     * <p>
     * Output:
     * 43 37
     * 39 31
     */
    void run() {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int a[][] = new int[n][n];
        int b[][] = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = sc.nextInt();
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = sc.nextInt();
            }
        }

        int[][] product = strassen(a, b);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%d ", product[i][j]);
            }
            System.out.printf("\n");
        }
    }

    /**
     * Test the strassen method vs the naive methods with random values.
     *
     * @param n          The dimension of the matrix, in power of 2. E.g. n=512 represents matrix A=512x512 multiply by matrix B=512x512
     * @param numOfTimes How many times to run each of the naive and Strassen method, so we can get the average run time for better comparison. Higher number means longer waiting for program to complete.
     */
    private void test(int n, int numOfTimes) {
        int[][] m1 = new int[n][n];
        int[][] m2 = new int[n][n];
        Random ran = new Random();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Populate m1 and m2 with random values from 0-100.
                m1[i][j] = ran.nextInt(100);
                m2[i][j] = ran.nextInt(100);
            }
        }

        double naiveTime = 0; // Time taken for the Naive algorithm
        for (int i = 0; i < numOfTimes; i++) {
            Instant start = Instant.now();
            naiveMatrixMult(m1, m2, 0, 0, 0, 0, n);
            Instant stop = Instant.now();
            naiveTime += Duration.between(start, stop).toMillis();
        }
        naiveTime = naiveTime / numOfTimes; // Get the average time per run.

        double stressonTime = 0; // Time taken for using stresson method.
        for (int i = 0; i < numOfTimes; i++) {
            Instant start = Instant.now();
            strassen(m1, m2);
            Instant stop = Instant.now();
            stressonTime += Duration.between(start, stop).toMillis();
        }
        stressonTime = stressonTime / numOfTimes;

        System.out.println("For multiplying Matrices of size " + n + "x" + n);
        System.out.println("Naive method takes: " + naiveTime + "ms");
        System.out.println("Stresson method takes: " + stressonTime + "ms");
    }

    public static void main(String[] args) {
        Strassen strassen = new Strassen();
        strassen.test(512, 3); // Runs naive vs strassen test with random values.
        // strassen.run(); // Uncomment this out to run strassen with your own values. You can comment out the .test() above.
    }
}
