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

#include <streambuf>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "MEL.hpp"

namespace MEL {

	class send_stream {
	private:
		/// Members		
		const MEL::Comm comm;
		const int dst, tag;
		const int blockSize;
		int index;
		std::vector<char> buffer;
		MEL::Request rq;

		inline void put_block() {
			MEL::Wait(rq);
			rq = MEL::Isend(&buffer[0], blockSize, dst, tag, comm);
			index = 0;
		};

	public:
		send_stream(const int _dst, const int _tag, const MEL::Comm &_comm, const int _blockSize = 256)
			: dst(_dst), tag(_tag), comm(_comm), blockSize(_blockSize), buffer(_blockSize, 0), index(0), rq(MEL::Request::REQUEST_NULL) {
		};
		~send_stream() {
			put_block();
		};

		send_stream(const send_stream &old) = delete;
		inline send_stream& operator=(const send_stream &old) = delete;

		inline void write(const char *ptr, const int length) {
			int len = length;
			char *cptr = (char*) ptr;

			while (len > 0) {
				const int insertLen = std::min(len, (blockSize - index));

				if (insertLen > 0) {
					//std::cout << "Inserting " << insertLen << " elements to buffer..." << std::endl;
					std::memcpy(&buffer[index], cptr, insertLen);
					cptr  += insertLen;
					index += insertLen;
					len   -= insertLen;	
				}
				
				if (index >= blockSize) put_block();
			}
		};

		template<typename T>
		inline send_stream& operator<<(const T &val) {
			write((const char*) &val, sizeof(T));
			return *this;
		};

	};

	class recv_stream {
	private:
		/// Members		
		const MEL::Comm comm;
		const int src, tag;
		const int blockSize;
		int index;
		std::vector<char> buffer;
		MEL::Request rq;

		inline void get_block() {
			MEL::Recv(&buffer[0], blockSize, src, tag, comm);
			index = 0;
		};

	public:
		recv_stream(const int _src, const int _tag, const MEL::Comm &_comm, const int _blockSize = 256)
			: src(_src), tag(_tag), comm(_comm), blockSize(_blockSize), buffer(_blockSize, 0), index(_blockSize), rq(MEL::Request::REQUEST_NULL) {
		};

		recv_stream(const recv_stream &old) = delete;
		inline recv_stream& operator=(const recv_stream &old) = delete;

		inline void read(char *ptr, const int length) {
			int len = length;
			char *cptr = (char*) ptr;

			while (len > 0) {
				const int retrieveLen = std::min(len, (blockSize - index));

				if (retrieveLen > 0) {
					//std::cout << "Retrieving " << retrieveLen << " elements from buffer..." << std::endl;
					std::memcpy(ptr, &buffer[index], retrieveLen);
					cptr  += retrieveLen;
					index += retrieveLen;
					len	  -= retrieveLen;
				}

				if (index >= blockSize) get_block();
			}
		};

		template<typename T>
		inline recv_stream& operator>>(T &val) {
			read((char*) &val, sizeof(T));
			return *this;
		};
	};

	class bcast_stream {
	private:
		/// Members		
		const MEL::Comm comm;
		const int src, rank;
		const int blockSize;
		int index;
		std::vector<char> buffer;
		MEL::Request rq;

		inline void sync_block() {
			if (rank == src) MEL::Wait(rq);
			rq = MEL::Ibcast(&buffer[0], blockSize, src, comm);
			if (rank != src) MEL::Wait(rq);
			index = 0;
		};

	public:
		bcast_stream(const int _src, const MEL::Comm &_comm, const int _blockSize = 256)
			: src(_src), rank(MEL::CommRank(_comm)), comm(_comm), blockSize(_blockSize), buffer(_blockSize, 0), rq(MEL::Request::REQUEST_NULL) {
			index = (src == rank) ? 0 : blockSize;
		};

		~bcast_stream() {
			if (src == rank) sync_block();
		};

		bcast_stream(const bcast_stream &old) = delete;
		inline bcast_stream& operator=(const bcast_stream &old) = delete;

		inline void write(const char *ptr, const int length) {
			if (src != rank) MEL::Abort(-1, "Attempting to write to read-only bcast_stream.");

			int len = length;
			char *cptr = (char*) ptr;

			while (len > 0) {
				const int insertLen = std::min(len, (blockSize - index));

				if (insertLen > 0) {
					//std::cout << "Inserting " << insertLen << " elements to buffer..." << std::endl;
					std::memcpy(&buffer[index], cptr, insertLen);
					cptr += insertLen;
					index += insertLen;
					len -= insertLen;
				}

				if (index >= blockSize) sync_block();
			}
		};

		inline void read(char *ptr, const int length) {
			if (src == rank) MEL::Abort(-1, "Attempting to read from write-only bcast_stream.");
			
			int len = length;
			char *cptr = (char*) ptr;

			while (len > 0) {
				const int retrieveLen = std::min(len, (blockSize - index));

				if (retrieveLen > 0) {
					//std::cout << "Retrieving " << retrieveLen << " elements from buffer..." << std::endl;
					std::memcpy(ptr, &buffer[index], retrieveLen);
					cptr  += retrieveLen;
					index += retrieveLen;
					len	  -= retrieveLen;
				}

				if (index >= blockSize) sync_block();
			}
		};

		template<typename T>
		inline bcast_stream& operator<<(const T &val) {
			write((const char*) &val, sizeof(T));
			return *this;
		};

		template<typename T>
		inline bcast_stream& operator>>(T &val) {
			read((char*) &val, sizeof(T));
			return *this;
		};

		template<typename T>
		inline bcast_stream& operator&(T &val) {
			if (src == rank) {
				write((const char*) &val, sizeof(T));
			}
			else {
				read((char*) &val, sizeof(T));
			}
			return *this;
		};
	};
};