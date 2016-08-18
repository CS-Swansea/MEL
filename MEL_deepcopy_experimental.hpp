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

#include "MEL.hpp"
#include "MEL_stream.hpp"

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <unordered_map>

namespace MEL {
    namespace Deep {

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class TransportSendStream {
        private:
            /// Members
            MEL::Send_stream stream;

        public:
            static constexpr bool SOURCE = true;
            
            TransportSendStream(const int _pid, const int _tag, const MEL::Comm &_comm, const int _blockSize = 512) 
                                : stream(_pid, _tag, _comm, _blockSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                stream.write(ptr, len);
                offset += len * sizeof(T);
            };
        };

        class TransportRecvStream {
        private:
            /// Members
            MEL::Send_stream stream;

        public:
            static constexpr bool SOURCE = false;

            TransportRecvStream(const int _pid, const int _tag, const MEL::Comm &_comm, const int _blockSize = 512) 
                                : stream(_pid, _tag, _comm, _blockSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                stream.read(ptr, len);
            };
        };

        class TransportBcastStreamRoot {
        private:
            /// Members
            MEL::Bcast_stream stream;

        public:
            static constexpr bool SOURCE = true;

            TransportBcastStreamRoot(const int _root, const MEL::Comm &_comm, const int _blockSize) 
                                    : stream(_root, _comm, _blockSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                stream.write(ptr, len);
            };
        };

        class TransportBcastStream {
        private:
            /// Members
            MEL::Bcast_stream stream;

        public:
            static constexpr bool SOURCE = false;

            TransportBcastStream(const int _root, const MEL::Comm &_comm, const int _blockSize) 
                                    : stream(_root, _comm, _blockSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                stream.read(ptr, len);
            };
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class TransportSend {
        private:
            /// Members
            const int pid, tag;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = true;

            TransportSend(const int _pid, const int _tag, const MEL::Comm &_comm) : pid(_pid), tag(_tag), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Send(ptr, len, pid, tag, comm);
            };
        };

        class TransportRecv {
        private:
            /// Members
            const int pid, tag;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = false;

            TransportRecv(const int _pid, const int _tag, const MEL::Comm &_comm) : pid(_pid), tag(_tag), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Recv(ptr, len, pid, tag, comm);
            };
        };

        class TransportBcastRoot {
        private:
            /// Members
            const int root;
            const MEL::Comm comm;
            
        public:
            static constexpr bool SOURCE = true;

            TransportBcastRoot(const int _root, const MEL::Comm &_comm) : root(_root), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Bcast(ptr, len, root, comm);
            };
        };

        class TransportBcast {
        private:
            /// Members
            const int root;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = false;

            TransportBcast(const int _root, const MEL::Comm &_comm) : root(_root), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Bcast(ptr, len, root, comm);
            };
        };

        class TransportFileWrite {
        private:
            /// Members
            const MEL::File file;

        public:
            static constexpr bool SOURCE = true;

            TransportFileWrite(const MEL::File &_file) : file(_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::FileWrite(file, ptr, len);
            };
        };

        class TransportFileRead {
        private:
            /// Members
            const MEL::File file;

        public:
            static constexpr bool SOURCE = false;

            TransportFileRead(const MEL::File &_file) : file(_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::FileRead(file, ptr, len);
            };
        };

        class TransportSTLFileWrite {
        private:
            /// Members
            std::ofstream *file;

        public:
            static constexpr bool SOURCE = true;
            
            TransportSTLFileWrite(std::ofstream &_file) : file(&_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);
                file->write((char*) ptr, num);
            };
        };

        class TransportSTLFileRead {
        private:
            /// Members
            std::ifstream *file;
               
        public:
            static constexpr bool SOURCE = false;

            TransportSTLFileRead(std::ifstream &_file) : file(&_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);
                file->read((char *) ptr, num);
            };
        };

        class TransportBufferWrite {
        private:
            /// Members
            int offset, bufferSize;
            char *buffer;

        public:
            static constexpr bool SOURCE = true;

            TransportBufferWrite(char *_buffer, const int _bufferSize) : offset(0), buffer(_buffer), bufferSize(_bufferSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);

                if ((offset + num) <= bufferSize) {
                    memcpy((void*) &buffer[offset], ptr, num);
                    offset += num;
                }
                else {
                    MEL::Abort(-1, "TransportBufferWrite : Offset longer than buffer...");
                }
            };
        };

        class TransportBufferRead {
        private:
            /// Members
            int offset, bufferSize;
            char *buffer;

        public:
            static constexpr bool SOURCE = false;

            TransportBufferRead(char *_buffer, const int _bufferSize) : offset(0), buffer(_buffer), bufferSize(_bufferSize) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);

                if ((offset + num) <= bufferSize) {
                    memcpy((void*) ptr, &buffer[offset], num);
                    offset += num;
                }
                else {
                    MEL::Abort(-1, "TransportBufferRead : Offset longer than buffer...");
                }
            };
        };

        class NoTransport {
        public:
            static constexpr bool SOURCE = true; 
        
            explicit NoTransport(const int dummy) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {};
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class PointerHashMap {
        private:
            struct PassThroughHash {
                size_t operator()(const size_t val) const {
                    return val;
                }
            };

            template<typename T>
            inline size_t HashPtr(const T *val) const {
                static const size_t shift = (size_t) log2(1 + sizeof(T));
                return (size_t) (val) >> shift;
            };

            /// Members
            std::unordered_map<size_t, void*, PassThroughHash> pointerMap;

        public:
            template<typename T>
            inline bool checkPointerCache(T* &ptr) {
                return pointerMap.find(HashPtr(ptr)) != pointerMap.end();
            };

            template<typename T>
            inline void cachePointer(T* oldPtr, T* ptr) {
                pointerMap.insert(std::make_pair(HashPtr(oldPtr), (void*) ptr));
            };

            template<typename T>
            inline void getCachedPointer(T* &ptr) {
                auto it = pointerMap.find(HashPtr(ptr));
                if (it != pointerMap.end()) {
                    ptr = (T*) it->second;
                }
            };
        };
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename TRANSPORT_METHOD, typename HASH_MAP = MEL::Deep::PointerHashMap>
        class Message;

        template<typename T>
        struct HasDeepCopyMethod {
            template<typename U, void(U::*)(MEL::Deep::Message<NoTransport>&)> struct SFINAE {};
            template<typename U> static char Test(SFINAE<U, &U::DeepCopy>*);
            template<typename U> static int Test(...);
            static const bool Has = sizeof(Test<T>(0)) == sizeof(char);
        };

        template<typename T, typename R = void>
        using enable_if_deep = typename std::enable_if<HasDeepCopyMethod<T>::Has, R>::type;
        template<typename T, typename R = void>
        using enable_if_not_deep = typename std::enable_if<!(HasDeepCopyMethod<T>::Has), R>::type;
        template<typename T, typename R = void>
        using enable_if_pointer = typename std::enable_if<std::is_pointer<T>::value, R>::type;
        template<typename T, typename R = void>
        using enable_if_not_pointer = typename std::enable_if<!std::is_pointer<T>::value, R>::type;

        template<typename T, typename R = void>
        using enable_if_deep_not_pointer = typename std::enable_if<HasDeepCopyMethod<T>::Has && !std::is_pointer<T>::value, R>::type;
        template<typename T, typename R = void>
        using enable_if_not_deep_not_pointer = typename std::enable_if<!HasDeepCopyMethod<T>::Has && !std::is_pointer<T>::value, R>::type;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename TRANSPORT_METHOD, typename HASH_MAP = MEL::Deep::PointerHashMap>
        class Message {
        private:
            /// Members
            int              offset;
            TRANSPORT_METHOD transporter;
            HASH_MAP         pointerMap;
            
            template<typename P>
            inline enable_if_pointer<P> transport(P &ptr, const int len) {
                offset += len * sizeof(std::remove_pointer<P>::type); // where P == T*, find T
                transporter.transport(ptr, len);
            };

            template<typename T>
            inline enable_if_not_pointer<T> transport(T &obj) {
                T *ptr = &obj;
                transport(ptr, 1);
            };

            template<typename P>
            inline enable_if_pointer<P> transportAlloc(P &ptr, const int len) {
                if (!TRANSPORT_METHOD::SOURCE) {
                    ptr = (len > 0 && ptr != nullptr) ? MEL::MemAlloc<std::remove_pointer<P>::type>(len) : nullptr; // where P == T*, find T
                }
                transport(ptr, len);
            };

        public:
            
            template<typename ...Args>
            Message(Args &&...args) : offset(0), transporter(std::forward<Args>(args)...) {};

            Message()                           = delete;
            Message(const Message &)            = delete;
            Message& operator=(const Message &) = delete;
            Message(Message &&)                 = delete;
            Message& operator=(Message &&)      = delete;

            inline int getOffset() const {
                return offset;
            };

            template<typename T>
            inline enable_if_not_deep<T> packVarFootprint(T &obj) {
                transport(obj);
            };

            template<typename D>
            inline enable_if_deep<D> packVarFootprint(D &obj) {
                transport(obj);
                obj.DeepCopy(*this);
            };

            template<typename D>
            inline enable_if_deep<D> packVar(D &obj) {
                obj.DeepCopy(*this);
            };

            template<typename T>
            inline enable_if_not_deep<T> packPtr(T* &ptr, int len = 1) {
                transportAlloc(ptr, len);
            };

            template<typename D>
            inline enable_if_deep<D> packPtr(D* &ptr, int len = 1) {
                transportAlloc(ptr, len);
                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            template<typename T>
            inline enable_if_not_deep<T> packSharedPtr(T* &ptr, int len = 1) {
                T *oldPtr = ptr;
                if (pointerMap.checkPointerCache(ptr)) {
                    if (!TRANSPORT_METHOD::SOURCE) {
                        pointerMap.getCachedPointer(ptr);
                    }
                    return;
                }

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);
            };

            template<typename D>
            inline enable_if_deep<D> packSharedPtr(D* &ptr, int len = 1) {
                D *oldPtr = ptr;
                if (pointerMap.checkPointerCache(ptr)) {
                    if (!TRANSPORT_METHOD::SOURCE) {
                        pointerMap.getCachedPointer(ptr);
                    }
                    return;
                }

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);

                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            inline void packSTL(std::string &obj) {
                int len = obj.size();
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::string(len, ' ');

                char *p = &obj[0];
                if (len > 0) transport(p, len);
            };

            template<typename T>
            inline enable_if_not_deep<T> packSTL(std::vector<T> &obj) {
                int len = obj.size();
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::vector<T>(len, T());

                T *p = &obj[0];
                if (len > 0) transport(p, len);
            };

            template<typename D>
            inline enable_if_deep<D> packSTL(std::vector<D> &obj) {
                int len = obj.size();
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::vector<D>(len, D());

                D *p = &obj[0];
                if (len > 0) transport(p, len);
                /// Copy content
                for (int i = 0; i < len; ++i) {
                    obj[i].DeepCopy(*this);
                }
            };

            template<typename T>
            inline enable_if_not_deep<T> packSTL(std::list<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) len = obj.size();
                transport(len);
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::list<T>(len, T());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                }
            };

            template<typename D>
            inline enable_if_deep<D> packSTL(std::list<D> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) len = obj.size();
                transport(len);
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::list<D>(len, D());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    it->DeepCopy(*this);
                }
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD, HASH_MAP>& operator&(std::vector<T> &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD, HASH_MAP>& operator&(std::list<T> &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD, HASH_MAP>& operator&(T &obj) {
                packVar(obj);
                return *this;
            };
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T, int> BufferSize(T &obj) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packVarFootprint(obj);
            return msg.getOffset();
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P, int> BufferSize(P &ptr) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packPtr(ptr);
            return msg.getOffset();
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P, int> BufferSize(P &ptr, const int len) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Send(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Send(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(obj, dst, tag, comm, MEL::Deep::BufferSize(obj));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packPtr(ptr);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, dst, tag, comm, MEL::Deep::BufferSize(ptr));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, len, dst, tag, comm, MEL::Deep::BufferSize(ptr, len));
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> SendStream(T &obj, const int dst, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportSendStream, HASH_MAP> msg(dst, tag, comm, blockSize);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> SendStream(P &ptr, const int dst, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportSendStream, HASH_MAP> msg(dst, tag, comm, blockSize);
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> SendStream(P &ptr, int const &len, const int dst, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportSendStream, HASH_MAP> msg(dst, tag, comm, blockSize);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Recv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Recv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Recv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::Recv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedRecv(T &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedRecv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> RecvStream(T &obj, const int src, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportRecvStream, HASH_MAP> msg(src, tag, comm, blockSize);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> RecvStream(P &ptr, const int src, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportRecvStream, HASH_MAP> msg(src, tag, comm, blockSize);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> RecvStream(P &ptr, int &len, const int src, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportRecvStream, HASH_MAP> msg(src, tag, comm, blockSize);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> RecvStream(P &ptr, int const &len, const int src, const int tag, const Comm &comm, const int blockSize = 512) {
            Message<TransportRecvStream, HASH_MAP> msg(src, tag, comm, blockSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::RecvStream(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> Bcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packVarFootprint(obj);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                msg.packVarFootprint(obj);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Bcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packPtr(ptr);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packPtr(ptr);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Bcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                int _len = len;
                msg.packVarFootprint(_len);
                msg.packPtr(ptr, _len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                int _len = len;
                msg.packVarFootprint(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::Bcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr, _len);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> Bcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packVarFootprint(len);
                msg.packPtr(ptr, len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packVarFootprint(len);
                msg.packPtr(ptr, len);
            }
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packVarFootprint(obj);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                msg.packVarFootprint(obj);

                MEL::MemFree(buffer);
            }
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast(obj, root, comm, 0);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, root, comm, MEL::Deep::BufferSize(ptr));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, root, comm, 0);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packVarFootprint(len); 
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packVarFootprint(len); 
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packVarFootprint(len);
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                int _len = len;
                msg.packVarFootprint(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedBcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> BcastStream(T &obj, const int root, const Comm &comm, const int blockSize = 512) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastStreamRoot, HASH_MAP> msg(root, comm, blockSize);
                msg.packVarFootprint(obj);
            }
            else {
                Message<TransportBcastStream, HASH_MAP> msg(root, comm, blockSize);
                msg.packVarFootprint(obj);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BcastStream(P &ptr, const int root, const Comm &comm, const int blockSize = 512) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastStreamRoot, HASH_MAP> msg(root, comm, blockSize);
                msg.packPtr(ptr);
            }
            else {
                Message<TransportBcastStream, HASH_MAP> msg(root, comm, blockSize);
                ptr = (P) 0x1;
                msg.packPtr(ptr);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BcastStream(P &ptr, int const &len, const int root, const Comm &comm, const int blockSize = 512) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastStreamRoot, HASH_MAP> msg(root, comm, blockSize);
                msg.packVarFootprint(len);
                msg.packPtr(ptr, len);
            }
            else {
                Message<TransportBcastStream, HASH_MAP> msg(root, comm, blockSize);
                ptr = (P) 0x1;
                int _len = len;
                msg.packVarFootprint(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BcastStream(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr, _len);
            }
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BcastStream(P &ptr, int &len, const int root, const Comm &comm, const int blockSize = 512) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastStreamRoot, HASH_MAP> msg(root, comm, blockSize);
                msg.packVarFootprint(len);
                msg.packPtr(ptr, len);
            }
            else {
                Message<TransportBcastStream, HASH_MAP> msg(root, comm, blockSize);
                ptr = (P) 0x1;
                msg.packVarFootprint(len);
                msg.packPtr(ptr, len);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> FileWrite(T &obj, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileWrite(P &ptr, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileWrite(P &ptr, int const &len, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> FileRead(T &obj, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, int const &len, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int const &len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> FileWrite(T &obj, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileWrite(P &ptr, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileWrite(P &ptr, int const &len, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_not_pointer<T> FileRead(T &obj, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            msg.packVarFootprint(obj);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, int const &len, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);
        };

        template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packVarFootprint(obj);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packVarFootprint(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVarFootprint(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };
    };
};