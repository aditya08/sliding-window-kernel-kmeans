#ifndef MATRIX_CUH
#define MATRIX_CUH

template <typename T>
class Matrix {
  public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, T* data);
    ~Matrix();

    void setValue(int row, int col, T value);
    T getValue(int row, int col) const;
    int getRows() const;
    int getCols() const;
    void copy(int rows, int cols, const T* data);
    const T* getDataPtr(int offset = 0);

    private:
    int nrows;
    int ncols;
    int size;
    T* data_ptr;

    void freeMemory();
    void allocateMemory(int len);
};

template class Matrix<float>;
template class Matrix<double>;
#endif // MATRIX_CUH