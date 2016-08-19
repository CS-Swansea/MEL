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

    class Send_stream {
    private:
        /// Members		
        const MEL::Comm comm;
        const int tag, blockSize;
        int dst, index, block;
        std::vector<char> buffer;

        inline void put_block() {
            if (dst < 0) MEL::Abort(-1, "Attempting to put to closed stream.");

            std::cout << "SEND Put: " << std::endl;
            MEL::Ssend(&buffer[block], blockSize, dst, tag, comm);
            index = 0;
            block = (block == 0) ? blockSize : 0;
            std::cout << "SEND POT: " << std::endl;
        };

    public:
        Send_stream(const int _dst, const int _tag, const MEL::Comm &_comm, const int _blockSize = 256)
            : dst(_dst), tag(_tag), comm(_comm), blockSize(_blockSize),
            buffer(_blockSize * 2, 0), index(0), block(0) {
        };
        ~Send_stream() {
            if (dst >= 0) put_block();
        };

        Send_stream(const Send_stream &old) = delete;
        inline Send_stream operator=(const Send_stream &old) = delete;
        Send_stream(Send_stream &&old) = delete;
        inline Send_stream operator=(Send_stream &&old) = delete;

        inline void close() {
            if (dst >= 0) put_block();
            dst = -1;
        };

        template<typename T>
        inline void write(const T *ptr, const int length = 1) {
            int len = length * sizeof(T);
            char *cptr = (char*) ptr;

            std::cout << "SEND Write: " << (sizeof(T) * length) << std::endl;

            while (len > 0) {
                const int insertLen = std::min(len, (blockSize - index));

                std::cout << "SEND Insert: " << insertLen << std::endl;

                if (insertLen > 0) {
                    std::memcpy(&buffer[block + index], cptr, insertLen);
                    cptr  += insertLen;
                    index += insertLen;
                    len   -= insertLen;
                }

                if (index >= blockSize) put_block();
            }
        };

        template<typename T>
        inline Send_stream& operator<<(const T &val) {
            write(&val);
            return *this;
        };

    };

    class Recv_stream {
    private:
        /// Members		
        const MEL::Comm comm;
        const int tag, blockSize;
        int src, index;
        std::vector<char> buffer;

        inline void get_block() {
            if (src < 0) MEL::Abort(-1, "Attempting to get from closed stream.");

            std::cout << "RECV Get: src = " << src << std::endl;
            MEL::Recv(&buffer[0], blockSize, src, tag, comm);
            index = 0;
            std::cout << "RECV GOT: " << std::endl;
        };

    public:
        Recv_stream(const int _src, const int _tag, const MEL::Comm &_comm, const int _blockSize = 256)
            : src(_src), tag(_tag), comm(_comm), blockSize(_blockSize), buffer(_blockSize, 0), index(_blockSize) {
        };

        Recv_stream(const Recv_stream &old) = delete;
        inline Recv_stream operator=(const Recv_stream &old) = delete;
        Recv_stream(Recv_stream &&old) = delete;
        inline Recv_stream operator=(Recv_stream &&old) = delete;

        inline void close() {
            src = -1;
        };

        template<typename T>
        inline void read(T *ptr, const int length = 1) {
            int len = length * sizeof(T);
            char *cptr = (char*) ptr;

            std::cout << "RECV Read: " << (sizeof(T) * length) << std::endl;

            while (len > 0) {
                const int retrieveLen = std::min(len, (blockSize - index));

                std::cout << "RECV Retrieve: " << retrieveLen << std::endl;

                if (retrieveLen > 0) {
                    std::memcpy(cptr, &buffer[index], retrieveLen);
                    cptr  += retrieveLen;
                    index += retrieveLen;
                    len   -= retrieveLen;
                }

                if (index >= blockSize) get_block();
            }
        };

        template<typename T>
        inline Recv_stream& operator>>(T &val) {
            read(&val);
            return *this;
        };
    };

    class Bcast_stream {
    private:
        /// Members		
        const MEL::Comm comm;
        const int rank, blockSize;
        int src, index, block;
        std::vector<char> buffer;
        
        inline void sync_block() {
            if (src < 0) MEL::Abort(-1, "Attempting to sync closed stream.");

            if (rank == src) {
                MEL::Bcast(&buffer[block], blockSize, src, comm);
                block = (block == 0) ? blockSize : 0;
            }
            else {
                MEL::Bcast(&buffer[0], blockSize, src, comm);
            }
            index = 0;
        };

    public:
        Bcast_stream(const int _src, const MEL::Comm &_comm, const int _blockSize = 256)
            : src(_src), rank(MEL::CommRank(_comm)), comm(_comm), blockSize(_blockSize),
            block(0) {
            
            if (src == rank) {
                index = 0;
                buffer.resize(blockSize * 2, 0);
            }
            else {
                index = blockSize;
                buffer.resize(blockSize, 0);
            }
        };

        ~Bcast_stream() {
            if (src == rank) sync_block();
        };

        Bcast_stream(const Bcast_stream &old) = delete;
        inline Bcast_stream operator=(const Bcast_stream &old) = delete;
        Bcast_stream(Bcast_stream &&old) = delete;
        inline Bcast_stream operator=(Bcast_stream &&old) = delete;

        inline void close() {
            if (src == rank && index > 0) sync_block();
            src = -1;
        };

        template<typename T>
        inline void write(const T *ptr, const int length = 1) {
            if (src != rank) MEL::Abort(-1, "Attempting to write to read-only bcast_stream.");

            int len = length * sizeof(T);
            char *cptr = (char*) ptr;

            while (len > 0) {
                const int insertLen = std::min(len, (blockSize - index));

                if (insertLen > 0) {
                    std::memcpy(&buffer[block + index], cptr, insertLen);
                    cptr  += insertLen;
                    index += insertLen;
                    len   -= insertLen;
                }

                if (index >= blockSize) sync_block();
            }
        };

        template<typename T>
        inline void read(T *ptr, const int length = 1) {
            if (src == rank) MEL::Abort(-1, "Attempting to read from write-only bcast_stream.");

            int len = length * sizeof(T);
            char *cptr = (char*) ptr;

            while (len > 0) {
                const int retrieveLen = std::min(len, (blockSize - index));

                if (retrieveLen > 0) {
                    std::memcpy(cptr, &buffer[index], retrieveLen);
                    cptr  += retrieveLen;
                    index += retrieveLen;
                    len   -= retrieveLen;
                }

                if (index >= blockSize) sync_block();
            }
        };

        template<typename T>
        inline Bcast_stream& operator<<(const T &val) {
            write(&val);
            return *this;
        };

        template<typename T>
        inline Bcast_stream& operator>>(T &val) {
            read(&val);
            return *this;
        };

        template<typename T>
        inline Bcast_stream& operator&(T &val) {
            if (src == rank) {
                write(&val);
            }
            else {
                read(&val));
            }
            return *this;
        };
    };
};