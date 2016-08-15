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

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <unordered_map>

namespace MEL {
    namespace Deep {

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class TransportSend {
        private:
            /// Members
            int offset;
            const int pid, tag;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = true;
            
            TransportSend(const int _pid, const int _tag, const MEL::Comm &_comm) : offset(0), pid(_pid), tag(_tag), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Send(ptr, len, pid, tag, comm);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportRecv {
        private:
            /// Members
            int offset;
            const int pid, tag;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = false;

            TransportRecv(const int _pid, const int _tag, const MEL::Comm &_comm) : offset(0), pid(_pid), tag(_tag), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Recv(ptr, len, pid, tag, comm);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                ptr = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportBcastRoot {
        private:
            /// Members
            int offset;
            const int root;
            const MEL::Comm comm;
            
        public:
            static constexpr bool SOURCE = true;

            TransportBcastRoot(const int _root, const MEL::Comm &_comm) : offset(0), root(_root), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Bcast(ptr, len, root, comm);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportBcast {
        private:
            /// Members
            int offset;
            const int root;
            const MEL::Comm comm;

        public:
            static constexpr bool SOURCE = false;

            TransportBcast(const int _root, const MEL::Comm &_comm) : offset(0), root(_root), comm(_comm) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::Bcast(ptr, len, root, comm);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                ptr = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportFileWrite {
        private:
            /// Members
            int offset;
            const MEL::File file;

        public:
            static constexpr bool SOURCE = true;

            TransportFileWrite(const MEL::File &_file) : offset(0), file(_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::FileWrite(file, ptr, len);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportFileRead {
        private:
            /// Members
            int offset;
            const MEL::File file;

        public:
            static constexpr bool SOURCE = false;

            TransportFileRead(const MEL::File &_file) : offset(0), file(_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                MEL::FileRead(file, ptr, len);
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                ptr = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportSTLFileWrite {
        private:
            /// Members
            int offset;
            std::ofstream *file;

        public:
            static constexpr bool SOURCE = true;
            
            TransportSTLFileWrite(std::ofstream &_file) : offset(0), file(&_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);
                file->write((char*) ptr, num);
                offset += num;
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportSTLFileRead {
        private:
            /// Members
            int offset;
            std::ifstream *file;
               
        public:
            static constexpr bool SOURCE = false;

            TransportSTLFileRead(std::ifstream &_file) : offset(0), file(&_file) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);
                file->read((char *) ptr, num);
                offset += num;
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                ptr = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportBufferWrite {
        private:
            /// Members
            int offset;
            const char *buffer;

        public:
            static constexpr bool SOURCE = true;

            TransportBufferWrite(const char *_buffer, const int bufferSize) : offset(0), buffer(_buffer) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);

                memcpy(&buffer[offset], &obj, num);
                offset += num;
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class TransportBufferRead {
        private:
            /// Members
            int offset;
            const char *buffer;

        public:
            static constexpr bool SOURCE = false;

            TransportBufferRead(const char *_buffer, const int bufferSize) : offset(0), buffer(_buffer) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                const int num = len * sizeof(T);

                memcpy(&obj, &buffer[offset], num);
                offset += num;
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                ptr = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        class NoTransport {
        private:
            /// Members
            int offset;

        public:
            static constexpr bool SOURCE = true; 
        
            NoTransport() : offset(0) {};

            template<typename T>
            inline void transport(T *&ptr, const int len) {
                offset += len * sizeof(T);
            };

            template<typename T>
            inline void transportAlloc(T *&ptr, const int len) {
                transport(ptr, len);
            };

            inline int getOffset() const {
                return offset;
            };
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename TRANSPORT_METHOD>
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


        template<typename TRANSPORT_METHOD>
        class Message {
        private:
            /// Members
            TRANSPORT_METHOD transporter;

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

            std::unordered_map<size_t, void*, PassThroughHash> pointerMap;

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

            template<typename T>
            inline enable_if_not_pointer<T> transport(T &obj) {
                transporter.transport(ptr, 1);
            };

            template<typename P>
            inline enable_if_pointer<P> transport(P &ptr, const int len) {
                transporter.transport(ptr, len);
            };

            template<typename P>
            inline enable_if_pointer<P> transportAlloc(P &ptr, const int len) {
                transporter.transportAlloc(ptr, len);
            };

        public:
            
            Message(TRANSPORT_METHOD _transporter) : transporter(_transporter) {};

            Message()                           = delete;
            Message(const Message &)            = delete;
            Message& operator=(const Message &) = delete;
            Message(Message &&)                 = delete;
            Message& operator=(Message &&)      = delete;

            inline int getOffset() const {
                return transporter.getOffset();
            };

            template<typename T>
            inline enable_if_not_deep<T> packVar(T &obj) {
                transport(obj);
            };

            template<typename D>
            inline enable_if_deep<D> packVar(D &obj) {
                transport(obj);
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
                if (checkPointerCache(ptr)) {
                    if (!TRANSPORT_METHOD::SOURCE) {
                        getCachedPointer(ptr);
                    }
                    return;
                }

                transportAlloc(ptr, len);
                cachePointer(oldPtr, ptr);
            };

            template<typename D>
            inline enable_if_deep<D> packSharedPtr(D* &ptr, int len = 1) {
                D *oldPtr = ptr;
                if (checkPointerCache(ptr)) {
                    if (!TRANSPORT_METHOD::SOURCE) {
                        getCachedPointer(ptr);
                    }
                    return;
                }

                transportAlloc(ptr, len);
                cachePointer(oldPtr, ptr);

                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            inline void packSTL(std::string &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE)  len = obj.size();
                transport(len);
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::string(len, ' ');

                char *p = &obj[0];
                if (len > 0) Transport(p, len);
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
                Transport(len);
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
                Transport(len);
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::list<D>(len, D());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    it->DeepCopy(*this);
                }
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD>& operator&(std::vector<T> &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD>& operator&(std::list<T> &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message<TRANSPORT_METHOD>& operator&(T &obj) {
                packVar(obj);
                return *this;
            };
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_deep_not_pointer<T, int> BufferSize(T &obj) {
            Message<NoTransport> msg(NoTransport::NoTransport());
            msg & obj;
            return msg.getOffset();
        };

        template<typename P>
        inline enable_if_pointer<P, int> BufferSize(P &ptr) {
            Message<NoTransport> msg(NoTransport::NoTransport());
            msg.packPtr(ptr);
            return msg.getOffset();
        };

        template<typename P>
        inline enable_if_pointer<P, int> BufferSize(P &ptr, const int len) {
            Message<NoTransport> msg(NoTransport::NoTransport());
            msg.packVar(len);
            msg.packPtr(ptr, len);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend> msg(TransportSend::TransportSend(dst, tag, comm));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> Send(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend> msg(TransportSend::TransportSend(dst, tag, comm));
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> Send(P &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend> msg(TransportSend::TransportSend(dst, tag, comm));
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg & obj;

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(obj, dst, tag, comm, MEL::Deep::BufferSize(obj));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packPtr(ptr);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, dst, tag, comm, MEL::Deep::BufferSize(ptr));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, len, dst, tag, comm, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv> msg(TransportRecv::TransportRecv(src, tag, comm));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv> msg(TransportRecv::TransportRecv(src, tag, comm));
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv> msg(TransportRecv::TransportRecv(src, tag, comm));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, const int len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv> msg(TransportRecv::TransportRecv(src, tag, comm));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::Recv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedRecv(T &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            msg & obj;

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedRecv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> Bcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot> msg(TransportBcastRoot::TransportBcastRoot(root, comm));
                msg & obj;
            }
            else {
                Message<TransportBcast> msg(TransportBcast::TransportBcast(root, comm));
                msg & obj;
            }
        };

        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot> msg(TransportBcastRoot::TransportBcastRoot(root, comm));
                msg.packPtr(ptr);
            }
            else {
                Message<TransportBcast> msg(TransportBcast::TransportBcast(root, comm));
                ptr = (P) 0x1;
                msg.packPtr(ptr);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, const int len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot> msg(TransportBcastRoot::TransportBcastRoot(root, comm));
                msg.packVar(len);
                msg.packPtr(ptr, len);
            }
            else {
                Message<TransportBcast> msg(TransportBcast::TransportBcast(root, comm));
                ptr = (P) 0x1;
                int _len = len;
                msg.packVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::Bcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr, _len);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot> msg(TransportBcastRoot::TransportBcastRoot(root, comm));
                msg.packVar(len);
                msg.packPtr(ptr, len);
            }
            else {
                Message<TransportBcast> msg(TransportBcast::TransportBcast(root, comm));
                ptr = (P) 0x1;
                msg.packVar(len);
                msg.packPtr(ptr, len);
            }
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
                msg & obj;

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, _bufferSize));
                msg & obj;

                MEL::MemFree(buffer);
            }
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast(obj, root, comm, 0);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, _bufferSize));
                ptr = (P) 0x1;
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, root, comm, MEL::Deep::BufferSize(ptr));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, root, comm, 0);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
                msg.packVar(len); 
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, _bufferSize));
                ptr = (P) 0x1;
                msg.packVar(len); 
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
                msg.packVar(len);
                msg.packPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, _bufferSize));
                ptr = (P) 0x1;
                int _len = len;
                msg.packVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedBcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> FileWrite(T &obj, MEL::File &file) {
            Message<TransportFileWrite> msg(TransportFileWrite::TransportFileWrite(file));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, MEL::File &file) {
            Message<TransportFileWrite> msg(TransportFileWrite::TransportFileWrite(file));
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, const int len, MEL::File &file) {
            Message<TransportFileWrite> msg(TransportFileWrite::TransportFileWrite(file));
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg & obj;

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> FileRead(T &obj, MEL::File &file) {
            Message<TransportFileRead> msg(TransportFileRead::TransportFileRead(file));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, MEL::File &file) {
            Message<TransportFileRead> msg(TransportFileRead::TransportFileRead(file));
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, const int len, MEL::File &file) {
            Message<TransportFileRead> msg(TransportFileRead::TransportFileRead(file));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, MEL::File &file) {
            Message<TransportFileRead> msg(TransportFileRead::TransportFileRead(file));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            msg & obj;

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> FileWrite(T &obj, std::ofstream &file) {
            Message<TransportSTLFileWrite> msg(TransportSTLFileWrite::TransportSTLFileWrite(file));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, std::ofstream &file) {
            Message<TransportSTLFileWrite> msg(TransportSTLFileWrite::TransportSTLFileWrite(file));
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, const int len, std::ofstream &file) {
            Message<TransportSTLFileWrite> msg(TransportSTLFileWrite::TransportSTLFileWrite(file));
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg & obj;

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite> msg(TransportBufferWrite::TransportBufferWrite(buffer, bufferSize));
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename T>
        inline enable_if_not_pointer<T> FileRead(T &obj, std::ifstream &file) {
            Message<TransportSTLFileRead> msg(TransportSTLFileRead::TransportSTLFileRead(file));
            msg & obj;
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, std::ifstream &file) {
            Message<TransportSTLFileRead> msg(TransportSTLFileRead::TransportSTLFileRead(file));
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, const int len, std::ifstream &file) {
            Message<TransportSTLFileRead> msg(TransportSTLFileRead::TransportSTLFileRead(file));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, std::ifstream &file) {
            Message<TransportSTLFileRead> msg(TransportSTLFileRead::TransportSTLFileRead(file));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            msg & obj;

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packPtr(ptr);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead> msg(TransportBufferRead::TransportBufferRead(buffer, bufferSize));
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);

            MEL::MemFree(buffer);
        };
    };
};