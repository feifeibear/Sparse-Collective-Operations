#ifndef _SPARSEDATA_H_
#define _SPARSEDATA_H_
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

void GenerateDenseArray(const int size, const float sparse_ratio,
    std::vector<float>& data, int rank);

void Dense2Sparse(const std::vector<float>& denseData,
    std::vector<float>& sparseData, std::vector<int>& index);

void Sparse2Dense(std::vector<float>& denseData,
    const std::vector<float>& sparseData, const std::vector<int>& sparseIdx);
#endif
