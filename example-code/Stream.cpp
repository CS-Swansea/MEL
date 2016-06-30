#define  MEL_IMPLEMENTATION
#include "MEL.hpp"
#include "MEL_stream.hpp"

int main(int argc, char *argv[]) {
	MEL::Init(argc, argv);

	MEL::Comm comm = MEL::Comm::WORLD;
	const int rank = MEL::CommRank(comm),
			  size = MEL::CommSize(comm);

	/*
	if (rank == 0) {
		MEL::send_stream sstream(1, 0, comm, 32);

		for (int i = 0; i < 50; ++i) {
			//sendStream.write((const char*) &i, sizeof(int));
			sstream << i;
		}

	}
	else if (rank == 1) {
		MEL::recv_stream rstream(0, 0, comm, 32);

		for (int i = 0; i < 50; ++i) {
			int j;
			//recvStream.read((char*) &j, sizeof(int));
			rstream >> j;

			std::cout << "Received j = " << j << std::endl;
		}
	}
	*/

	if (rank == 0) {
		MEL::bcast_stream bstream(0, comm);

		for (int i = 0; i < 10; ++i) {
			//sendStream.write((const char*) &i, sizeof(int));
			bstream << i;
		}

	}
	else {
		MEL::bcast_stream bstream(0, comm);

		for (int i = 0; i < 10; ++i) {
			int j;
			//recvStream.read((char*) &j, sizeof(int));
			bstream >> j;

			std::cout << "Rank " << rank << " Received j = " << j << std::endl;
		}
	}

	MEL::Finalize();
	return 0;
};