#include "collectives.h"
#include "sparseTest/sparsedata.hpp"
#include "timer.h"

#include <mpi.h>

#include <stdexcept>
#include <iostream>
#include <vector>

void TestMPIAllreduceCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations, const float sparseRatio) {
    // Initialize on CPU (no GPU device ID).
    InitCollectives(NO_DEVICE);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");
    timer::Timer timer;
    timer::Timer timer2;

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        std::vector<float> denseData;
        GenerateDenseArray(size, sparseRatio, denseData, mpi_rank);
        float sparse_seconds = 0.0f, sparse_comm_seconds = 0.0f, dense_seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            timer.start();

            //transfer dense to sparse
            std::vector<float> sparseData;
            std::vector<int> sparseIdx;
            Dense2Sparse(denseData, sparseData, sparseIdx);

            //declear output
            std::vector<float> sparseOutputData;
            std::vector<int> sparseOutputIdx;

            timer2.start();
            if(!sparseData.empty())
              SparseTreeAllreduce(sparseData, sparseIdx, sparseOutputData, sparseOutputIdx);
            else
              std::cout << "data is empty, no all reduce" << std::endl;
            sparse_comm_seconds += timer2.seconds();


            //transfer sparse to dense
            std::vector<float> denseOutput(denseData.size(), 0);
            Sparse2Dense(denseOutput, sparseOutputData, sparseOutputIdx);

            sparse_seconds += timer.seconds();


            //Check results
            //int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
            //                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
            timer.start();
            std::vector<float> denseOutputRef(denseData.size(), 0);
            //MPI_Allreduce(&denseData[0], &denseOutputRef[0], denseData.size(),
            //    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            TreeAllreduce(&denseData[0], denseData.size(), &denseOutputRef[0]);
            dense_seconds += timer.seconds();
            // Check that we get the expected result.
            //if(1 || mpi_rank == 0) {
            //  std::cout<<"----output result------"<<std::endl;
            //  for(size_t j = 0; j < denseOutput.size(); j++) {
            //    std::cout << j << " : " << denseData[j] << " : " << denseOutput[j] << " " << denseOutputRef[j] << std::endl;
            //  }
            //  std::cout<<"----output result------"<<std::endl;
            //}
            for(size_t j = 0; j < denseOutput.size(); j++) {
                if(fabs(denseOutput[j] - denseOutputRef[j]) > 1e-6 && mpi_rank == 0) {
                    std::cerr << "Unexpected result from allreduce: " << denseOutput[j] << " vs " << denseOutputRef[j] << std::endl;
                }
            }
        }
        if(mpi_rank == 0) {
            std::cout << "Verified sparse MPI allreduce for size "
                << size
                << " ( sparse : "
                << sparse_seconds / iters
                << " , sparse comm. : "
                << sparse_comm_seconds / iters
                << " , dense : "
                << dense_seconds / iters
                << " per iteration)" << std::endl;
        }

        //delete[] data;
    }

}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./allreduce-test (cpu|gpu) sparseRatio" << std::endl;
        return 1;
    }
    std::string input(argv[1]);
    float sparseRatio = stof(argv[2]);

    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes = {
        65536, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
    };

    // Number of iterations to run for each buffer size.
    std::vector<size_t> iterations = {
        //100000, 100000, 100000, 100000,
        //1000, 1000, 1000, 1000,
        //100, 50, 10, 1
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    // Test on either CPU and GPU.
    if(input == "cpu") {
        TestMPIAllreduceCPU(buffer_sizes, iterations, sparseRatio);
        //TestCollectivesCPU(buffer_sizes, iterations);
    } else if(input == "gpu") {
    } else {
        std::cerr << "Unknown device type: " << input << std::endl
                  << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
