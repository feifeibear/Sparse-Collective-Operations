#include <iostream>
#include "sparsedata.hpp"

int main() {
  std::vector<float> denseData;
  const int denseSize = 20;
  //denseData = new float [denseSize];
  const float sparseRatio = 0.2;
  GenerateDenseArray(denseSize, sparseRatio, denseData);
  for(int i = 0; i < denseSize; ++i) {
    std::cout << denseData[i] << " ";
  }
  std::cout << std::endl;

  //test Dense2Sparse
  std::vector<float> sparseData;
  std::vector<int> index;
  int sparseSize;
  Dense2Sparse(denseData, sparseData, index);

  std::cout << sparseSize << std::endl;
  for(int i = 0; i < sparseSize; ++i) {
    std::cout << index[i] << " : " << sparseData[i] << ",";
  }
  std::cout << std::endl;

  //test sparse2Dense
  std::vector<float> DenseOut (denseSize, 0.0);
  Sparse2Dense(DenseOut, sparseData, index);
  for(int i = 0; i < denseSize; ++i) {
    std::cout << DenseOut[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
