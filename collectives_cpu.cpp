#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>
#include <mpi.h>

#include "collectives.h"

struct MPIGlobalState {
    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
};

static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void InitCollectives(int device) {
    // CPU-only initialization.
    int mpi_error = MPI_Init(NULL, NULL);
    if(mpi_error != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Init failed with an error");
    }
    global_state.initialized = true;
}

// Allocate a new memory buffer on CPU or GPU.
float* alloc(size_t size) {
    // CPU memory allocation through standard allocator.
    return new float[size];
}

// Deallocate an allocated memory buffer.
void dealloc(float* buffer) {
    // CPU memory deallocation through standard allocator.
    delete[] buffer;
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size) {
    // CPU memory allocation through standard allocator.
    std::memcpy((void*) dst, (void*) src, size * sizeof(float));
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size) {
    // Accumulate values from `src` into `dst` on the CPU.
    for(size_t i = 0; i < size; i++) {
        dst[i] += src[i];
    }
}


// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}

/***
  * AllReduce with latency efficient way
  * T = logP(alpha + n*beta) 
  ***/
void TreeAllreduce(float* data, size_t length, float** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for(size_t other_length : lengths) {
        if(length != other_length) {
            throw std::runtime_error("RingAllreduce received different lengths");
        }
    }
    //TODO only work for log2
    // Allocate the output buffer.
    float* buffer = alloc(length);
    float* output = alloc(length);
    *output_ptr =  output;

    MPI_Status recv_status;
    MPI_Request recv_req;
    // What type of data is being sent
    MPI_Datatype datatype = MPI_FLOAT;
    copy(output, data, length);

    //reduce
    for(int step = size/2; step >= 1; step /= 2) {
      //send process
      if(rank >= step && rank < step*2) {
        int send_to = rank - step;
        MPI_Send(output, length,
                datatype, send_to, 0, MPI_COMM_WORLD);

      }
      //recv process
      if(rank < step) {
        int recv_from = rank + step;
        MPI_Recv(buffer, length,
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
        reduce(output, buffer, length);
      }
    }
    //gather
    for(int step = 1; step <= size/2; step *= 2) {
      //recv
      if(rank >= step && rank < step*2) {
        int recv_from = rank - step;
        MPI_Recv(output, length,
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
      }
      //send
      if(rank < step) {
        int send_to = rank + step;
        MPI_Send(output, length,
                MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
      }
    }
    dealloc(buffer);
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.

//add src to dst, original dst will be covered
void sparseReduce(vector<float>& dst_data, vector<int>& dst_idx, const vector<float>& src_data, const vector<int>& src_idx) {
    // Accumulate values from `src` into `dst` on the CPU.
    int i = 0, j = 0;
    vector<float> res_data;
    vector<int> res_idx;
    int dst_size = dst_data.size();
    int src_size = src_data.size();
    while(i < dst_size && j < src_size) {
      if(dst_idx[i] < src_idx[j]) {
        res_data.push_back(dst_data[i]);
        res_idx.push_back(dst_idx[i]);
        i++;
      } else if (dst_idx[i] == src_idx[j]) {
        res_data.push_back(dst_data[i] + src_data[j]);
        res_idx.push_back(dst_idx[i]);
        i++;
        j++;
      } else {
        res_data.push_back(src_data[j]);
        res_idx.push_back(src_idx[j]);
        j++;
      }
    }
    while(i < dst_size) {
      res_idx.push_back(dst_idx[i]);
      res_data.push_back(dst_data[i]);
      i++;
    }
    while(j < src_size) {
      res_idx.push_back(src_idx[j]);
      res_data.push_back(src_data[j]);
      j++;
    }
    dst_data.assign(res_data.begin(), res_data.end());
    dst_idx.assign(res_idx.begin(), res_idx.end());

    return;
}

void SparseTreeAllreduce(const vector<float>& input_data, const vector<int> &input_index, vector<float>& output_data, vector<int>& output_idx) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    //TODO only work for log2
    // Allocate the output buffer.
    //float* buffer = alloc(length);
    //float* output = alloc(length);
    //*output_ptr =  output;

    MPI_Status recv_status;
    MPI_Request recv_req;

    //init copy : input -> buffer
    vector<float> buffer_data(input_data);
    vector<int> buffer_idx(input_index);

    //reduce
    for(int step = size/2; step >= 1; step /= 2) {
      //sending processes : send buffer
      if(rank >= step && rank < step*2) {
        int send_to = rank - step;
        int send_length = buffer_data.size();
        MPI_Send(&send_length, 1, MPI_INT, send_to, 0, MPI_COMM_WORLD);
        MPI_Send(&buffer_data[0], send_length,
                MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
        MPI_Send(&buffer_idx[0], send_length,
                MPI_INT, send_to, 0, MPI_COMM_WORLD);

      }
      //recving processes : recv to recv_buff, reduce to buffer
      if(rank < step) {
        int recv_from = rank + step;
        int send_length = 0;
        MPI_Recv(&send_length, 1, MPI_INT, recv_from, 0, MPI_COMM_WORLD, &recv_status);
        vector<float> recv_buffer_data(send_length, 0);
        vector<int> recv_buffer_idx(send_length, 0);
        MPI_Recv(&recv_buffer_data[0], send_length,
                MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &recv_status);
        MPI_Recv(&recv_buffer_idx[0], send_length,
                MPI_INT, recv_from, 0, MPI_COMM_WORLD, &recv_status);
        sparseReduce(buffer_data, buffer_idx, recv_buffer_data, recv_buffer_idx);
      }
    }
    //Proc 0 : store buffer -> output_data
    //bcast output from Proc 0
    int finial_length = buffer_data.size();
    MPI_Bcast(&finial_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
      output_data.assign(buffer_data.begin(), buffer_data.end());
      output_idx.assign(buffer_idx.begin(), buffer_idx.end());
    } else {
      output_data.resize(finial_length);
      output_idx.resize(finial_length);
    }

    MPI_Bcast(&output_data[0], finial_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&output_idx[0], finial_length, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //gather
    //for(int step = 1; step <= size/2; step *= 2) {
    //  //recv
    //  if(rank >= step && rank < step*2) {
    //    int recv_from = rank - step;
    //    MPI_Recv(output, length,
    //            datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
    //  }
    //  //send
    //  if(rank < step) {
    //    int send_to = rank + step;
    //    MPI_Send(output, length,
    //            datatype, send_to, 0, MPI_COMM_WORLD);
    //  }
    //}
}
