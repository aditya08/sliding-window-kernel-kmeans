#include "dataloader.hpp"

int main(int argc, char** argv) {
    const std::string filename = "./data/acoustic_small";
    LIBSVMReader<float> reader(filename);
    int nrows = reader.getNumRows();
    int ncols = reader.getNumCols();
    int nnz = reader.getNumNonZero();
    int size = reader.getSize();
    if (nrows != 10 || ncols != 50 || nnz <= 0) {
        std::cerr << "No data to load from file: " << filename << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "Dataset: " << filename << " | Rows: " << nrows << " | Cols: " << ncols << " | Non-zero entries: " << nnz << " | Size: " << size << "\n";
    Matrix<float> dataset(nrows, ncols);
    Matrix<int> labels(nrows, 1);

    reader.loadData(dataset, labels);

    // Print first 10 rows of dataset and labels
    std::cout << "First 10 rows of dataset:\n";
    for (int i = 0; i < std::min(10, nrows); ++i) {
        std::cout << "Label: " << labels.getValue(i, 0) << " | Data: ";
        for (int j = 0; j < ncols; ++j) {
            std::cout << dataset.getValue(i, j) << " ";
        }
        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}
