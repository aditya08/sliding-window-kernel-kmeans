#ifndef __DATALOADER_HPP__
#define __DATALOADER_HPP__

#include <iostream>

#include "matrix.hpp"

template <typename T>
class LIBSVMReader {
  public:
    LIBSVMReader(const std::string filename): filename(filename) {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            nrows = 0;
            ncols = 0;
            nnz = 0;
            size = 0;
            return;
        }
        readMetadata();
    }
    void loadData(Matrix<float>& dataset, Matrix<int>& labels) {
        std::string line;
        int row = 0, nnz = 0;
        const int* labels_dataptr = labels.getDataPtr();
        const float* data_dataptr = dataset.getDataPtr();
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            labels_dataptr[row++] = std::stoi(token);
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    T value = static_cast<T>(std::stof(token.substr(pos + 1)));
                    data_dataptr[nnz++] = value;
                }
            }
        }
    }

    void loadData(Matrix<double>& dataset, Matrix<int>& labels) {
        std::string line;
        int row = 0, nnz = 0;
        const int* labels_dataptr = labels.getDataPtr();
        const double* data_dataptr = dataset.getDataPtr();
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            labels_dataptr[row++] = std::stoi(token);
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    T value = static_cast<T>(std::stod(token.substr(pos + 1)));
                    data_dataptr[nnz++] = value;
                }
            }
        }
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
    void readMetadata(){
        nrows = 0;
        ncols = 0;
        nnz = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            nrows++;
            std::istringstream iss(line);
            std::string token;
            // First token is the label
            iss >> token;
            // Remaining tokens are feature_index:feature_value
            while (iss >> token) {
                size_t pos = token.find(':');
                if (pos != std::string::npos) {
                    int index = std::stoi(token.substr(0, pos));
                    if (index > ncols) {
                        ncols = index;
                    }
                    nnz++;
                }
            }
        }
        size = nrows * ncols;
        // Reset file stream to beginning for data reading
        file.clear();
        file.seekg(0, std::ios::beg);
    }
};

template class LIBSVMReader<float>;
template class LIBSVMReader<double>;
#endif // __DATALOADER_HPP__