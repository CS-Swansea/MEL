#pragma once

#include <omp.h>
#include "MEL.hpp"

namespace MEL {
	namespace OMP {

		enum Schedule : int {
			STATIC		= omp_sched_static,
			DYNAMIC		= omp_sched_dynamic,
			GUIDED		= omp_sched_guided,
			AUTO		= omp_sched_auto
		};

		namespace Functor {
			template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&)>
			void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
				omp_set_schedule((omp_sched_t) SCHEDULE, CHUNK);
				#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
				for (int i = 0; i < *len; ++i) 
					inout[i] = F(in[i], inout[i]);
			};
	
			template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&, MEL::Datatype)>
			void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
				omp_set_schedule((omp_sched_t) SCHEDULE, CHUNK);
				#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
				for (int i = 0; i < *len; ++i) 
					inout[i] = F(in[i], inout[i], dt);
			};
		};

		/// MEL-style mapped functor
		template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&)>
		inline MEL::Op CreateOp(bool commute = true) {
			MPI_Op op;
			MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) MEL::OMP::Functor::ARRAY_OP_FUNC<NUM_THREADS, CHUNK, SCHEDULE, T, F>, commute, (MPI_Op*) &op), "OMP::Op::CreatOp" );
			return MEL::Op(op);
		};
		template<int NUM_THREADS, int CHUNK, Schedule SCHEDULE, typename T, T(*F)(T&, T&, MEL::Datatype)>
		inline MEL::Op CreateOp(bool commute = true) {
			MPI_Op op;
			MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) MEL::OMP::Functor::ARRAY_OP_FUNC<NUM_THREADS, CHUNK, SCHEDULE, T, F>, commute, (MPI_Op*) &op), "OMP::Op::CreatOp" );
			return MEL::Op(op);
		};

	};
};