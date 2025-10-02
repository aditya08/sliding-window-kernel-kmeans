#ifndef MATRIX_CUH
#define MATRIX_CUH

template <typename T>
class Matrix {
  public:
    Matrix(int rows, int cols, bool allocateOnDevice = false);
    Matrix(int rows, int cols, T* data, bool allocateOnDevice = false);
    ~Matrix();

    void setValue(int row, int col, T value);
    T getValue(int row, int col) const;
    int getRows() const;
    int getCols() const;
    void copy(int rows, int cols, const T* data, bool allocateOnDevice = false);
    const T* getDataPtr(int offset = 0);

    private:
    int nrows = 0;
    int ncols = 0;
    int size = 0;
    bool isOnDevice = false;
    T* data_ptr = nullptr;

    void freeMemory();
    void allocateMemory(int len, bool allocateOnDevice = false);
};

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;
#endif // MATRIX_CUH