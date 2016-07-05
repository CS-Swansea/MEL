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
#pragma once

#include <omp.h>
#include "MEL.hpp"

namespace MEL {
    namespace OMP {

		/**
 		 * \defgroup OMP OpenMP Extensions
		 * Extensions to MEL that leverage OpenMP for within node paralellism
		 */

		/**
		 * \ingroup  OMP
		 */
        enum Schedule : int {
            STATIC  = omp_sched_static,
            DYNAMIC = omp_sched_dynamic,
            GUIDED  = omp_sched_guided,
            AUTO    = omp_sched_auto
        };

        namespace Functor {

			/**
			 * \ingroup  OMP
			 * Maps the given binary functor to the local array of a reduction / accumulate operation, using OpenMP for parallelism
			 *
			 * \param[in] in		The left hand array for the reduction
			 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
			 * \param[in] len		Pointer to a single int representing the number of elements to be processed
			 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
			 */
            template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&)>
            void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
                omp_set_schedule((omp_sched_t) SCHEDULE, CHUNK);
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
                for (int i = 0; i < *len; ++i) 
                    inout[i] = F(in[i], inout[i]);
            };
    
			/**
			 * \ingroup  OMP
			 * Maps the given binary functor to the local array of a reduction / accumulate operation, using OpenMP for parallelism
			 *
			 * \param[in] in		The left hand array for the reduction
			 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
			 * \param[in] len		Pointer to a single int representing the number of elements to be processed
			 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
			 */
            template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&, MEL::Datatype)>
            void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
                omp_set_schedule((omp_sched_t) SCHEDULE, CHUNK);
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
                for (int i = 0; i < *len; ++i) 
                    inout[i] = F(in[i], inout[i], dt);
            };
        };

        /**
		 * \ingroup  OMP
		 * Maps the given binary functor to the local array of a reduction / accumulate operation, using OpenMP for parallelism
		 *
		 * \param[in] commute		Is the operation commutative
		 * \return					Returns a handle to a new Op
		 */
		template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&)>
		inline MEL::Op OpCreate(bool commute = true) {
            MPI_Op op;
            MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) MEL::OMP::Functor::ARRAY_OP_FUNC<NUM_THREADS, CHUNK, SCHEDULE, T, F>, commute, (MPI_Op*) &op), "OMP::Op::CreatOp" );
            return MEL::Op(op);
        };

		/**
		 * \ingroup  OMP
		 * Maps the given binary functor to the local array of a reduction / accumulate operation, using OpenMP for parallelism
		 *
		 * \param[in] commute		Is the operation commutative
		 * \return					Returns a handle to a new Op
		 */
        template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&, MEL::Datatype)>
		inline MEL::Op OpCreate(bool commute = true) {
            MPI_Op op;
            MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) MEL::OMP::Functor::ARRAY_OP_FUNC<NUM_THREADS, CHUNK, SCHEDULE, T, F>, commute, (MPI_Op*) &op), "OMP::Op::CreatOp" );
            return MEL::Op(op);
        };

    };
};