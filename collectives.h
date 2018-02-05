#ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_
#define BAIDU_ALLREDUCE_COLLECTIVES_H_ value

#include <cstddef>
#include <vector>
#include <mpi.h>
using namespace std;

#define NO_DEVICE -1

/*
 * This file contains the implementation of the baidu-allreduce communication
 * collectives, and provides the following functions:
 *
 *    void InitCollectives(int device);
 *    void RingAllreduce(float* data, size_t length, float** output);
 *    void RingAllgather(float* data, size_t length, float** output);
 *
 */

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
void InitCollectives(int device);

void TreeAllreduce(float* data, size_t length, float* output_ptr);

void SparseTreeAllreduce(const vector<float>& data, const vector<int> &index, vector<float>& output_data, vector<int>& output_idx);
#endif /* ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_ */
