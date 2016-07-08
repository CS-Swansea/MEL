/*
The MIT License(MIT)

Copyright(c) 2016 Joss Whittle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#define  MEL_IMPLEMENTATION
#include "MEL_omp.hpp"

#include <iomanip>

//-----------------------------------------------------------------//
// Example Usage: mpirun --pernode --hostfile [path] ./ompExample  //
//-----------------------------------------------------------------//
int main(int argc, char *argv []) {
    MEL::Init(argc, argv);
    
    // Who are we?
    MEL::Comm comm = MEL::Comm::WORLD;
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);

    double start, end;

    // Allocate buffers
    const int LEN = 1e8;
    int *src = MEL::MemAlloc<int>(LEN, 1), // Initilize to 1
        *dst = (rank == 0) ? MEL::MemAlloc<int>(LEN) : nullptr;
    
    // Perform the Reduction on 1 thread with MPI_SUM
    start = MEL::Wtime();
    //MPI_Reduce(src, dst, LEN, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MEL::Reduce(src, dst, LEN, MEL::Op::SUM, 0, comm); 
    end = MEL::Wtime();

    if (rank == 0)
        std::cout << "Reduced " << LEN << " elements in " << std::setw(10) << (end - start) 
        << " seconds on 1 thread with MEL::Op::SUM == MPI_SUM." << std::endl;

    // Create a MEL User Defined Operation using a functor wrapped in a map function
    auto SUM = MEL::OpCreate<int, MEL::Functor::SUM>();

    // Perform the Reduction on 1 thread
    start = MEL::Wtime();
    MEL::Reduce(src, dst, LEN, SUM, 0, comm);
    end   = MEL::Wtime();

    if (rank == 0) 
        std::cout << "Reduced " << LEN << " elements in " << std::setw(10) << (end - start)
        << " seconds on 1 thread with mapped MEL::FUNCTOR::SUM." << std::endl;

    // Create a MEL User Defined Operation using a functor wrapped in a parallel map function
    auto ompSUM = MEL::OMP::OpCreate<int, MEL::Functor::SUM>();
    
    // For 2, 4, 8, 16 threads
    for (int n = 2; n <= 16; n <<= 1) {
        omp_set_num_threads(n);
        omp_set_schedule(omp_sched_static, 0);

        // Perform the Reduction on n threads
        start = MEL::Wtime();
        MEL::Reduce(src, dst, LEN, ompSUM, 0, comm);
        end   = MEL::Wtime();
    
        if (rank == 0)
            std::cout << "Reduced " << LEN << " elements in " << std::setw(10) << (end - start)
            << " seconds on " << n << " threads with parallel mapped MEL::FUNCTOR::SUM." << std::endl;
    }

    // Clean up the operations when we are done with them
    MEL::OpFree(SUM, ompSUM);
    // Clean up buffers
    MEL::MemFree(src, dst);

    MEL::Finalize();
    return 0;
}