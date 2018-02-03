#include "sparsedata.hpp"

void GenerateDenseArray(const int size, const float sparse_ratio,
    std::vector<float>& data, int rank = 0) {
  srand((unsigned)time(NULL) + rank);
  for(int i = 0; i < size; ++i) {
    float p = rand()/float(RAND_MAX);
    if(p > sparse_ratio) {
      data.push_back(0);
    } else {
      data.push_back(p);
    }
  }
  return;
}

void Dense2Sparse(const std::vector<float>& denseData,
    std::vector<float>& sparseData, std::vector<int>& index) {
  int denseSize = denseData.size();
  for(int i = 0; i < denseSize; ++i) {
    if(fabs(denseData[i] - 0.0) < 1e-6) {
      continue;
    } else {
      //std::cout << i << ", " << denseData[i] << ' ';
      sparseData.push_back(denseData[i]);
      index.push_back(i);
    }
  }
  return;
}

/* ***
 * dense data should already be allocated
 * */
void Sparse2Dense(std::vector<float>& denseData,
    const std::vector<float>& sparseData, const std::vector<int>& sparseIdx) {
  int size = sparseIdx.size();
  for(int i = 0; i < size; ++i) {
    denseData[sparseIdx[i]] = sparseData[i];
  }
  return;
}

