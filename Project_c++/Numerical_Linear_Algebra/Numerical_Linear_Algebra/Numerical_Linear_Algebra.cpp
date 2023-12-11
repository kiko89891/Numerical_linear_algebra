#include <iostream>
#include <vector>
#include <functional>
#include <stdio.h>
#include <numeric>
#include <string>
#include <sstream>
#include <cctype>  // For isdigit function
#include <stdexcept>

#include <Eigen/Dense>



// ROWS CANNOT BE LESS THAN COLUMNS ELSE DOESN NOT MAKE SENSE

// Inspiration for PCA 
// https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9


// we need the matrix to be templated so that we can Initilize several matrices at once
// with all methods avaliable to all matrices. 
template <typename T>
class Matrix {

private:
    // vector of vectors where the rows are vectors aand each vector contain N number of elements(columns)
    std::vector<std::vector<double>> matrix;

public:
    // initialize a matrix with a set of rows and column filled with a default value of 2 
    Matrix(double rows, double cols, double defaultValue = 2)
        :matrix(rows, std::vector<double>(cols, defaultValue)) {}

    double rows = matrix.size();
    double cols = matrix[0].size();

    
    friend class PCA ;  // THIS IS NEEDED TO ACCESS VARIABLES and METHODS FROM class Matrix to class PCA
    
 

    template <typename T>
    T setValue(double row, double col, T value) {
        // check for out of range values

        try {
            
            matrix.at(row).at(col) = value;
        }

        catch (const std::out_of_range& e) {
            std::cout << "Error: Attempted to access an element outside the bounds of the matrix.";
        }
        return 0;
    };

    // Function to display the matrix
    template <typename T>
    T display() const {
        for (const auto& row : matrix) {
            for (T element : row) {
                std::cout << element << "\t";

            }
            std::cout << "\n";

        }
        return 0;
    };

    
    template <typename T>
    T transpose() {

        // Create a new matrix for the transpose
        std::vector<std::vector<T>> Mat_Transpose(cols, std::vector<T>(rows, 0));

        // Find the transpose using std::swap
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::swap(Mat_Transpose[j][i], matrix[i][j]);
            }
        }

        matrix = Mat_Transpose; // Overwrite the matrix with the transpose
        return 0;
    }

    
   // Function to calculate the mean, variance, and center the data of each column in a matrix
    template <typename T>
    T Center_data() {

        // Calculate means
        std::vector<T> columnMeans(cols, 0.0);
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                columnMeans[j] += matrix[i][j];
            }
            columnMeans[j] /= static_cast<T>(rows);
        }

        // Calculate E(X^2)
        std::vector<T> columnSquaredMeans(cols, 0.0);
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                columnSquaredMeans[j] += std::pow(matrix[i][j], 2);
            }
            columnSquaredMeans[j] /= static_cast<T>(rows);
        }

        // Calculate variances
        std::vector<T> columnVariances(cols, 0.0);
        for (size_t j = 0; j < cols; ++j) {
            columnVariances[j] = columnSquaredMeans[j] - std::pow(columnMeans[j], 2);
        }

        // Center the data
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i][j] = (matrix[i][j] - columnMeans[j]) / std::sqrt(columnVariances[j]);
            }
        }
        return 0;
    }

    // Takes Martrix and takes an operation and matrix 
    template <typename T>
    T operate(const Matrix& Matrix2, char operation) {

        /////////////////////////////
        if (cols != Matrix2.matrix.size() && operation == '*') {
            std::cerr << "Error: ROWS on matrix1 and COLMUNS on matrix2 are not the same size." << std::endl;
            return 0;
        }
        else if (operation == '*') {
            std::vector<std::vector<T>> result(rows, std::vector<T>(Matrix2.matrix[0].size(), 0));

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < Matrix2.matrix[0].size(); ++j) {
                    for (size_t k = 0; k < cols; ++k) {
                        result[i][j] += matrix[i][k] * Matrix2.matrix[k][j];
                    }
                }

            }
            matrix = result; // Update the matrix with the multiplication result
        }


        // Check for dimensionality check
        if (rows != Matrix2.matrix.size() || cols != Matrix2.matrix[0].size()) {
            std::cerr << "Error: Matrices must have the same dimensions for operation." << std::endl;
            return 0;
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (operation == '+') {
                    matrix[i][j] += Matrix2.matrix[i][j];
                }
                else if (operation == '-') {
                    matrix[i][j] -= Matrix2.matrix[i][j];
                }
            }
        }
    };


    // eigenvalues , eigenvectors , inverse of a matrix

    //https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm //
    // 
    // Function to calculate the covariance matrix ( FOR A SAMPLE NOT A POPULATION!!!!!!!!) N-1  is divided not N

    template <typename T>
    std::vector<std::vector<T>> calculateCovarianceMatrix() {
        // Get the number of rows and columns in the matrix
     
        // Calculate the mean for each column
        std::vector<T> mean(cols, 0.0);
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                mean[j] += matrix[i][j];
            }
            mean[j] /= rows;
        }

        // Calculate the covariance matrix
        std::vector<std::vector<T>> covarianceMatrix(cols, std::vector<T>(cols, 0.0));

        // THIS IS COLUMS AS THE COVRAIANCE MATRIX IS SYMETRIC DEFINED BY THE SIZE OF THE COLUMNS 
        // BY DEFINITION OF EIGENVALUES IT NEEDS TO BE SYMETRIC
        for (size_t i = 0; i < cols; ++i) { // it should be cols not rows 

            for (size_t j = 0; j < cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {

                   covarianceMatrix[j][k] += (matrix[i][j] - mean[j]) * (matrix[i][k] - mean[k]);

                }
            }
        }

        // Divide the Covariance matrix by the number of observations
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                covarianceMatrix[j][k] /= (rows - 1);
            }
        }
        matrix = covarianceMatrix;  // The matrix becomes the covariance matrix 

        return covarianceMatrix;
    }
    // Function to convert a matrix from vector of vectors to Eigen::MatrixXd
    template <typename T>
    Eigen::MatrixXd convertToEigenMatrix() {
        if (matrix.empty() || matrix[0].empty()) {
            throw std::invalid_argument("Input matrix is empty");
        }

        Eigen::MatrixXd eigenMatrix(matrix.size(), matrix[0].size());

        for (size_t i = 0; i < matrix.size(); ++i) {
            if (matrix[i].size() != matrix[0].size()) {
                throw std::invalid_argument("Inconsistent row sizes in input matrix");
            }

            for (size_t j = 0; j < matrix[0].size(); ++j) {
                eigenMatrix(i, j) = matrix[i][j];
            }
        }

        return eigenMatrix;


    }


    // Function to calculate eigenvalues and eigenvectors using the QR method
    template <typename T>
    T calculateEigenvaluesAndEigenvectors(const Eigen::MatrixXd& inputMatrix,
        Eigen::MatrixXd& eigenvalues,
        Eigen::MatrixXd& eigenvectors,
        size_t maxIterations = 100) 
    
    {
        size_t n = inputMatrix.rows();eigenvalues = inputMatrix;eigenvectors = Eigen::MatrixXd::Identity(n, n);

        for (size_t i = 0; i < maxIterations; ++i) {
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(eigenvalues);
            Eigen::MatrixXd Q = qr.householderQ();
            Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
            eigenvalues = R * Q;
            eigenvectors = eigenvectors * Q;

            // Check for convergence of the matrices
            T offDiagonalSum = eigenvalues.cwiseAbs().sum() - eigenvalues.diagonal().cwiseAbs().sum();


            if (offDiagonalSum < 0.000001) {
                std::cout << "Converged after " << i + 1 << " iterations.\n";
                break;
            }
        }
        return 0;
    }

};

// it is a new class since the above are properties of a matrix, whereas this is not
class PCA {
          
      // Step 1: Standardize the dataset. IN CLASS int main it happens

      //  Step 2 : Calculate the covariance matrix for the features in the dataset.

      //  Step 3 : Calculate the eigenvalues and eigenvectors for the covariance matrix.
    
      //  Step 4 : Transform the original matrix.

   
public:

    template <typename T>
    T Call_PCA(Matrix<T>& pca)
    {
       
        // STEP 2 

        std::cout << "Display before :\n" << std::endl;

        pca.display<T>();
        pca.Center_data<T>();
       
        // Display the matrix
       
        std::cout << "Display after centering:\n" << std::endl;
        pca.display<T>();
        
        // Calculate Covariance matrix

        pca.calculateCovarianceMatrix<T>();

        // STEP 3 
        // 
        // 
         // Convert the matrix to Eigen::MatrixXd
        Eigen::MatrixXd eigenMatrix = pca.convertToEigenMatrix<T>();

        // Display the resulting Eigen::MatrixXd
        std::cout << "Eigen::MatrixXd:\n" << eigenMatrix << std::endl;

        // Eigenvectors 
        Eigen::MatrixXd eigenvalues;
        Eigen::MatrixXd eigenvectors;

        pca.calculateEigenvaluesAndEigenvectors<T>(eigenMatrix, eigenvalues, eigenvectors);

        // Display the results
        std::cout << "Eigenvalues:\n" << std::endl << eigenvalues << std::endl;
        std::cout << "Eigenvectors:\n" << std::endl << eigenvectors << std::endl;


        // step 4 Calculate PCA

        Eigen::MatrixXd PCA_matrix = eigenMatrix * eigenvectors;


        std::cout << "PCA MATRIX:\n" << std::endl << PCA_matrix << std::endl;
        return 0;
    }


};




int main() {


    // STEP 1   

    // Define user input
    std::string userInput;
    std::string userInput2;
    double row;
    double col;


    std::cout << "Enter rows size of the matrix: ";
    std::cin >> userInput;
    std::cout << "Enter  columns size of the matrix: ";
    std::cin >> userInput2;
    try {
        // Attempt to convert userInput to an integer
        row = std::stoi(userInput);
        col = std::stoi(userInput2);
     
    }
    catch (const std::invalid_argument& e) {
        // Catch exception for invalid input (non integer) 
        std::cout << "Invalid input. Please enter an interger. Line 275" << std::endl;
    };



    // Create an instance of the Matrix class

    // Initialize matrix
    Matrix<double> myMatrix(row, col);

    

    std::cout << "fill data" << std::endl;

    // use random numbers to fill the matrix call it N*M times
    myMatrix.setValue<double>(0, 0, 5.234f);
    myMatrix.setValue<double>(0, 1, 2.123f);

    myMatrix.setValue<double>(1, 0, 4.345f);
    myMatrix.setValue<double>(1, 1, 3.1234f);
    myMatrix.setValue<double>(0, 2, 52.2134f);
    myMatrix.setValue<double>(1, 2, 43.234f);
    myMatrix.setValue<double>(2, 2, 34.75f);

    myMatrix.setValue<double>(0, 3, 454.4f);
    myMatrix.setValue<double>(1, 3, 12.34f);
    myMatrix.setValue<double>(2, 3, 567.4f);

  

    PCA init_PCA;

    init_PCA.Call_PCA<double>(myMatrix); // call function 


};
