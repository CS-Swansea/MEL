#define  MEL_IMPLEMENTATION
#include "MEL.hpp"

#include "MEL_stl.hpp"

int main(int argc, char *argv[]) {
	MEL::Init(argc, argv);

	MEL::Comm comm = MEL::Comm::WORLD;
	const int rank = MEL::CommRank(comm),
			  size = MEL::CommSize(comm);
	
	srand(12345);

	std::vector<int> a(10);
	if (rank == 0) {
		for (auto it = a.begin(); it != a.end(); ++it) {
			*it = rand() % 10;
		}

		for (int i = 1; i < size; ++i) 
			MEL::Send(a.begin(), a.end(), MEL::Datatype::INT, i, 0, comm);
	}
	else {
		MEL::Recv(a.begin(), a.end(), MEL::Datatype::INT, 0, 0, comm);
	}

	auto mutex = MEL::MutexCreate(0, comm);

	MEL::MutexLock(mutex);

	std::cout << "Rank: " << rank << " | ";
	for (auto it = a.begin(); it != a.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	
	MEL::MutexUnlock(mutex);

	MEL::MutexFree(mutex);

	MEL::Finalize();
	return 0;
};