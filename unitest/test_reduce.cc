#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

void GenerateDenseArray(const int size, const float sparse_ratio,
    std::vector<float>& data) {
  srand((unsigned)time(0) + size);
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
      sparseData.push_back(denseData[i]);
      index.push_back(i);
    }
  }
  return;
}

void Sparse2Dense(std::vector<float>& denseData, const int denseSize,
    const std::vector<float>& sparseData, const std::vector<int>& index, const int sparseSize) {
  int size = index.size();
  for(int i = 0; i < size; ++i) {
    denseData[index[i]] = sparseData[i];
  }
  return;
}

//add src to dst, original dst will be covered
void sparseReduce(vector<float>& dst_data, vector<int>& dst_idx, const vector<float>& src_data, const vector<int>& src_idx) {
    // Accumulate values from `src` into `dst` on the CPU.
    int i = 0, j = 0;
    int dst_size = dst_data.size();
    int src_size = src_data.size();
    vector<float> res_data(dst_size + src_size);
    vector<int> res_idx(dst_size + src_size);
    int pos = 0;
    //vector<float>::iterator data_iter = res_data.begin();
    //vector<int>::iterator idx_iter = res_idx.begin();
    while(i < dst_size && j < src_size) {
      if(dst_idx[i] < src_idx[j]) {
        //res_data.push_back(dst_data[i]);
        //res_idx.push_back(dst_idx[i]);
        res_data[pos] = dst_data[i];
        res_idx[pos] = dst_idx[i];
        pos++;
        i++;
      } else if (dst_idx[i] == src_idx[j]) {
        //res_data.push_back(dst_data[i] + src_data[j]);
        //res_idx.push_back(dst_idx[i]);
        res_data[pos] = dst_data[i] + src_data[j];
        res_idx[pos] = dst_idx[i];
        pos++;
        i++;
        j++;
      } else {
        //res_data.push_back(src_data[j]);
        //res_idx.push_back(src_idx[j]);
        res_data[pos] = src_data[j];
        res_idx[pos] = src_idx[j];
        pos++;
        j++;
      }
    }
    while(i < dst_size) {
      //res_idx.push_back(dst_idx[i]);
      //res_data.push_back(dst_data[i]);
      res_idx[pos] = (dst_idx[i]);
      res_data[pos] = (dst_data[i]);
      pos++;
      i++;
    }
    while(j < src_size) {
      //res_idx.push_back(src_idx[j]);
      //res_data.push_back(src_data[j]);
      res_idx[pos] = (src_idx[j]);
      res_data[pos] = (src_data[j]);
      pos++;
      j++;
    }
    //dst_data.assign(res_data.begin(), res_data.end());
    //dst_idx.assign(res_idx.begin(), res_idx.end());
    dst_data.assign(res_data.begin(), res_data.begin() + pos);
    dst_idx.assign(res_idx.begin(), res_idx.begin() + pos);

    return;
}

void printSparse(vector<float>& d, vector<int>& idx) {
  int l = d.size();
  cout << "this vector is " <<  endl;
  for(int i = 0; i < l; ++i) {
    std::cout << idx[i] << " : " << d[i] << " , ";
  }
  cout << std::endl;
}

int main() {
  vector<float> dst_data;
  vector<int> dst_idx;
  vector<float> src_data;
  vector<int> src_idx;

  vector<float> raw_data1;
  GenerateDenseArray(10, 1., raw_data1);
  Dense2Sparse(raw_data1, dst_data, dst_idx);

  vector<float> raw_data2;
  GenerateDenseArray(15, 1., raw_data2);
  Dense2Sparse(raw_data2, src_data, src_idx);

  printSparse(dst_data, dst_idx);
  printSparse(src_data, src_idx);
  sparseReduce(dst_data, dst_idx, src_data, src_idx);
  printSparse(dst_data, dst_idx);

  return 0;

}
