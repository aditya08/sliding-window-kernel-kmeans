#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <fstream>
#include <iostream>
#include <sstream>

#include "matrix.hpp"

template <typename T>
class LIBSVMReader {
  public:
    LIBSVMReader(const std::string filename, int max_rows = -1, int max_cols = -1): filename(filename) {
        readMetadata(max_rows, max_cols);
    }
    void loadData(Matrix<float>& dataset, Matrix<int>& labels, int max_rows = -1, int max_cols = -1) {
        std::string line;
        int row = 0, nnz = 0;
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            return;
        }
        // const int* labels_dataptr = labels.getDataPtr();
        // const float* data_dataptr = dataset.getDataPtr();
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            labels.setValue(row, 0, std::stoi(token));
            // labels_dataptr[row++] = std::stoi(token);
            int col = 0;
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    col = std::stoi(token.substr(0, pos)) - 1; // Convert to 0-based index
                    if (max_cols > 0 && col >= max_cols) continue;
                    T value = static_cast<T>(std::stof(token.substr(pos + 1)));
                    dataset.setValue(row, col, value);
                    nnz++;
                    // data_dataptr[nnz++] = value;
                }
            }
            row++;
            if (max_rows > 0 && row >= max_rows) break;
        }
        file.close();
    }

    void loadData(Matrix<double>& dataset, Matrix<int>& labels, int max_rows = -1, int max_cols = -1) {
        std::string line;
        int row = 0, nnz = 0;
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            return;
        }

        // const int* labels_dataptr = labels.getDataPtr();
        const double* data_dataptr = dataset.getDataPtr();
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            labels.setValue(row, 0, std::stoi(token));
            // labels_dataptr[row++] = std::stoi(token);
            int col = 0;
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    col = std::stoi(token.substr(0, pos)) - 1; // Convert to 0-based index
                    if (max_cols > 0 && col >= max_cols) continue;
                    T value = static_cast<T>(std::stod(token.substr(pos + 1)));
                    dataset.setValue(row, col, value);
                    nnz++;
                    // data_dataptr[nnz++] = value;
                }
            }
            row++;
            if (max_rows > 0 && row >= max_rows) break;
        }
        file.close();
    }

    int getNumRows() const { return nrows; }
    int getNumCols() const { return ncols; }
    int getNumNonZero() const { return nnz; }
    int getSize() const { return size; }

    ~LIBSVMReader(){
        if (file.is_open()) {
            file.close();
        }
    }

    private:
    std::string filename;
    std::ifstream file;
    int nrows;
    int ncols;
    int nnz; // number of non-zero entries
    int size; // total size of the data array
    void readMetadata(int max_rows = -1, int max_cols = -1) {
        nrows = 0;
        ncols = 0;
        nnz = 0;
        file.open(filename); // Reopen the file for reading metadata
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            // First token is the label
            iss >> token;
            // Remaining tokens are feature_index:feature_value
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    int index = std::stoi(token.substr(0, pos));
                    if (max_cols > 0 && index > max_cols) continue;
                    else if (index > ncols) {
                        ncols = index;
                    }
                    nnz++;
                }
            }
            nrows++;
            if (max_rows > 0 && nrows >= max_rows) break;
        }
        size = nrows * ncols;
        // Reset file stream to beginning for data reading
        file.close(); // Close the file after reading metadata
    }
};

template class LIBSVMReader<float>;
template class LIBSVMReader<double>;
#endif // __DATALOADER_HPP__