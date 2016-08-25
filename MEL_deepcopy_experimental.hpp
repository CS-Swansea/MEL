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
            };
        };

        class TransportRecvStream {
        private:
            /// Members
            MEL::Recv_stream stream;

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
            }
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
            // Maximum number of hashmaps is the bit size of a pointer
            static constexpr size_t NUM_HASH_MAPS = sizeof(void*) * 8; // == 32 or 64 based on system

            // We hash pointers explicitly before giving them to the hashmaps, so the hash function
            // for the std::unordered_map becomes just returning the already hashed value.
            struct PassThroughHash {
                inline size_t operator() (const size_t val) const { return val; }
            };

            // Array of hashmaps for storing pointers to types of any size
            std::unordered_map<size_t, void*, PassThroughHash> pointerMap[NUM_HASH_MAPS];

        public:
            // Pointer hashmap public interface

            // Returns true if oldPtr is found in the hash-map and sets ptr equal to the stored value
            // Otherwise returns false and ptr is unaltered
            template<typename T>
            inline bool checkPointerCache(T* oldPtr, T* &ptr) {
                // The shift value to use for a type T
                static const size_t shift = log2(1 + sizeof(T));

                // Is oldPtr already in the hashmap for T?
                const auto it = pointerMap[shift - 1].find(((size_t) oldPtr >> shift));
                if (it != pointerMap[shift - 1].end()) {
                    // If so set ptr equal to the value stored in the hashmap
                    ptr = (T*) it->second;
                    return true;
                }
                return false;
            };

            // Insert ptr into the hashmap using oldptr as the key
            template<typename T>
            inline void cachePointer(T* oldPtr, T* ptr) {
                // The shift value to use for a type T
                static const size_t shift = log2(1 + sizeof(T));
                pointerMap[shift - 1].insert(std::make_pair(((size_t) oldPtr >> shift), (void*) ptr));
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

        template <typename T>
        using is_vector = typename std::is_same<T, std::vector<typename T::value_type, typename T::allocator_type>>;
        template <typename T>
        using is_list =   typename std::is_same<T, std::list<typename T::value_type, typename T::allocator_type>>;
        template <typename T>
        using is_string = typename std::is_same<T, std::string>;
        
        template<typename T, typename R = void>
        using enable_if_stl = typename std::enable_if<is_vector<T>::value || is_list<T>::value || is_string<T>::value, R>::type;
        template<typename T, typename R = void>
        using enable_if_not_pointer_not_stl = typename std::enable_if<(!is_vector<T>::value || is_list<T>::value || is_string<T>::value) && !std::is_pointer<T>::value, R>::type;
        template<typename T, typename R = void>
        using enable_if_deep_not_pointer_not_stl = typename std::enable_if<HasDeepCopyMethod<T>::Has && (!is_vector<T>::value || is_list<T>::value || is_string<T>::value) && !std::is_pointer<T>::value, R>::type;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename TRANSPORT_METHOD, typename HASH_MAP>
        class Message {
        private:
            /// Members
            int              offset;
            TRANSPORT_METHOD transporter;
            HASH_MAP         pointerMap;
            
            template<typename P>
            inline enable_if_pointer<P> transport(P &ptr, const int len) {
                typedef typename std::remove_pointer<P>::type T; // where P == T*, find T

                offset += len * sizeof(T);
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
                    typedef typename std::remove_pointer<P>::type T; // where P == T*, find T
                    ptr = (len > 0 && ptr != nullptr) ? MEL::MemAlloc<T>(len) : nullptr; 
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

            template<typename D>
            inline enable_if_deep<D> packVar(D &obj) {
                obj.DeepCopy(*this);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packVar(T &obj) {
                F(obj, *this);
            };

            template<typename T>
            inline enable_if_not_deep<T> packRootVar(T &obj) {
                transport(obj);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packRootVar(T &obj) {
                transport(obj);
                F(obj, *this);
            };

            template<typename D>
            inline enable_if_deep<D> packRootVar(D &obj) {
                transport(obj);
                obj.DeepCopy(*this);
            };

            template<typename T>
            inline enable_if_not_deep<T> packPtr(T* &ptr, int len = 1) {
                transportAlloc(ptr, len);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packPtr(T* &ptr, int len = 1) {
                transportAlloc(ptr, len);
                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        F(ptr[i], *this);
                    }
                }
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
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packSharedPtr(T* &ptr, int len = 1) {
                T *oldPtr = ptr;
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);

                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        F(ptr[i], *this);
                    }
                }
            };

            template<typename D>
            inline enable_if_deep<D> packSharedPtr(D* &ptr, int len = 1) {
                D *oldPtr = ptr;
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);

                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            template<typename T>
            inline enable_if_not_deep<T> packRootPtr(T* &ptr, int len = 1) {
                // Explicitly transport the pointer value for the root node
                size_t addr = (size_t) ptr;
                transport(addr);
                ptr = (T*) addr;

                T *oldPtr = ptr;
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packRootPtr(T* &ptr, int len = 1) {
                // Explicitly transport the pointer value for the root node
                size_t addr = (size_t) ptr;
                transport(addr);
                ptr = (T*) addr;

                T *oldPtr = ptr;
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

                transportAlloc(ptr, len);
                pointerMap.cachePointer(oldPtr, ptr);
                
                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        F(ptr[i], *this);
                    }
                }
            };

            template<typename D>
            inline enable_if_deep<D> packRootPtr(D* &ptr, int len = 1) {
                // Explicitly transport the pointer value for the root node
                size_t addr = (size_t) ptr;
                transport(addr);
                ptr = (D*) addr;
                
                D *oldPtr = ptr;
                if (pointerMap.checkPointerCache(oldPtr, ptr)) return;

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

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packSTL(std::vector<T> &obj) {
                int len = obj.size();
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::vector<T>(len, T());

                T *p = &obj[0];
                if (len > 0) transport(p, len);

                /// Copy content
                for (int i = 0; i < len; ++i) {
                    F(obj[i], *this);
                }
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

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packSTL(std::list<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) len = obj.size();
                transport(len);
                if (!TRANSPORT_METHOD::SOURCE) new (&obj) std::list<T>(len, T());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    F(*it, *this);
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

            inline void packRootSTL(std::string &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len);
                }

                char *p = &obj[0];
                if (len > 0) transport(p, len);
            };

            template<typename T>
            inline enable_if_not_deep<T> packRootSTL(std::vector<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, T());
                }

                T *p = &obj[0];
                if (len > 0) transport(p, len);
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packRootSTL(std::vector<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, T());
                }

                T *p = &obj[0];
                if (len > 0) transport(p, len);
                /// Copy content
                for (int i = 0; i < len; ++i) {
                    F(obj[i], *this);
                }
            };

            template<typename D>
            inline enable_if_deep<D> packRootSTL(std::vector<D> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, D());
                }

                D *p = &obj[0];
                if (len > 0) transport(p, len);
                /// Copy content
                for (int i = 0; i < len; ++i) {
                    obj[i].DeepCopy(*this);
                }
            };

            template<typename T>
            inline enable_if_not_deep<T> packRootSTL(std::list<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, T());
                }

                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                }
            };

            template<typename T, void(*F)(T&, MEL::Deep::Message<TRANSPORT_METHOD, HASH_MAP>&)>
            inline void packRootSTL(std::list<T> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, T());
                }

                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    F(*it, *this);
                }
            };

            template<typename D>
            inline enable_if_deep<D> packRootSTL(std::list<D> &obj) {
                int len;
                if (TRANSPORT_METHOD::SOURCE) {
                    len = obj.size(); transport(len);
                }
                else {
                    transport(len); obj.resize(len, D());
                }

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

#define TEMPLATE_STL template<typename S, typename HASH_MAP = MEL::Deep::PointerHashMap>
#define TEMPLATE_T   template<typename T, typename HASH_MAP = MEL::Deep::PointerHashMap>
#define TEMPLATE_P   template<typename P, typename HASH_MAP = MEL::Deep::PointerHashMap>

#define TEMPLATE_STL_F(transport_method) template<typename S, typename HASH_MAP, void(*F)(typename S::value_type &, MEL::Deep::Message<transport_method, HASH_MAP>&)>
#define TEMPLATE_T_F(transport_method)   template<typename T, typename HASH_MAP, void(*F)(T&, MEL::Deep::Message<transport_method, HASH_MAP>&)>
#define TEMPLATE_P_F(transport_method)   template<typename P, typename HASH_MAP, void(*F)(typename std::remove_pointer<P>::type&, MEL::Deep::Message<transport_method, HASH_MAP>&)>

#define TEMPLATE_STL_F2(transport_method1, transport_method2) template<typename S, typename HASH_MAP, void(*F1)(typename S::value_type &, MEL::Deep::Message<transport_method1, HASH_MAP>&),               void(*F2)(typename S::value_type &, MEL::Deep::Message<transport_method2, HASH_MAP>&)>
#define TEMPLATE_T_F2(transport_method1, transport_method2)   template<typename T, typename HASH_MAP, void(*F1)(T&, MEL::Deep::Message<transport_method1, HASH_MAP>&),                                     void(*F2)(T&, MEL::Deep::Message<transport_method2, HASH_MAP>&)>
#define TEMPLATE_P_F2(transport_method1, transport_method2)   template<typename P, typename HASH_MAP, void(*F1)(typename std::remove_pointer<P>::type&, MEL::Deep::Message<transport_method1, HASH_MAP>&), void(*F2)(typename std::remove_pointer<P>::type&, MEL::Deep::Message<transport_method2, HASH_MAP>&)>

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Buffer Size
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S, int> BufferSize(S &obj) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootSTL(obj);
            return msg.getOffset();
        };

        TEMPLATE_STL_F(NoTransport)
        inline enable_if_stl<S, int> BufferSize(S &obj) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootSTL<S, F>(obj);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T, int> BufferSize(T &obj) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootVar(obj);
            return msg.getOffset();
        };

        TEMPLATE_T_F(NoTransport)
        inline enable_if_not_pointer_not_stl<T, int> BufferSize(T &obj) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootVar<T, F>(obj);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P, int> BufferSize(P &ptr) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootPtr(ptr);
            return msg.getOffset();
        };

        TEMPLATE_P_F(NoTransport)
        inline enable_if_pointer<P, int> BufferSize(P &ptr) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootPtr<typename std::remove_pointer<P>::type, F>(ptr);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P, int> BufferSize(P &ptr, const int len) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
            return msg.getOffset();
        };

        TEMPLATE_P_F(NoTransport)
        inline enable_if_pointer<P, int> BufferSize(P &ptr, const int len) {
            Message<NoTransport, HASH_MAP> msg(0);
            msg.packRootVar(len);
            msg.packRootPtr<typename std::remove_pointer<P>::type, F>(ptr, len);
            return msg.getOffset();
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Send
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL 

        TEMPLATE_STL
        inline enable_if_stl<S> Send(S &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL_F(TransportSend)
        inline enable_if_stl<S> Send(S &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootSTL<S, F>(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedSend(S &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL_F(TransportBufferWrite)
        inline enable_if_stl<S> BufferedSend(S &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL<S, F>(obj);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedSend(S &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(obj, dst, tag, comm, MEL::Deep::BufferSize(obj));
        };

        TEMPLATE_STL_F(TransportBufferWrite)
        inline enable_if_stl<S> BufferedSend(S &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend<S, HASH_MAP, F>(obj, dst, tag, comm, MEL::Deep::BufferSize<S, HASH_MAP, F>(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootVar(obj);
        };

        TEMPLATE_T_F(TransportSend)
        inline enable_if_not_pointer_not_stl<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootVar<T, F>(obj);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(obj);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_T_F(TransportBufferWrite)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar<T, F>(obj);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(obj, dst, tag, comm, MEL::Deep::BufferSize(obj));
        };

        TEMPLATE_T_F(TransportBufferWrite)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend<T, HASH_MAP, F>(obj, dst, tag, comm, MEL::Deep::BufferSize<T, HASH_MAP, F>(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> Send(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P_F(TransportSend)
        inline enable_if_pointer<P> Send(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootPtr<P, F>(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootPtr(ptr);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootPtr<P, F>(ptr);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, dst, tag, comm, MEL::Deep::BufferSize(ptr));
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend<P, HASH_MAP, F>(ptr, dst, tag, comm, MEL::Deep::BufferSize<P, HASH_MAP, F>(ptr));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> Send(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P_F(TransportSend)
        inline enable_if_pointer<P> Send(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            Message<TransportSend, HASH_MAP> msg(dst, tag, comm);
            msg.packRootVar(len);
            msg.packRootPtr<P, F>(ptr, len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(len);
            msg.packRootPtr<P, F>(ptr, len);

            MEL::Deep::Send(buffer, bufferSize, dst, tag, comm);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend(ptr, len, dst, tag, comm, MEL::Deep::BufferSize(ptr, len));
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedSend(P &ptr, int const &len, const int dst, const int tag, const Comm &comm) {
            MEL::Deep::BufferedSend<P, HASH_MAP, F>(ptr, len, dst, tag, comm, MEL::Deep::BufferSize<P, HASH_MAP, F>(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Recv
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S> Recv(S &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL_F(TransportRecv)
        inline enable_if_stl<S> Recv(S &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            msg.packRootSTL<S, F>(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedRecv(S &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL_F(TransportBufferRead)
        inline enable_if_stl<S> BufferedRecv(S &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL<S, F>(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            msg.packRootVar(obj);
        };

        TEMPLATE_T_F(TransportRecv)
        inline enable_if_not_pointer_not_stl<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            msg.packRootVar<T, F>(obj);
        };
        
        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedRecv(T &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(obj);

            MEL::MemFree(buffer);
        };

        TEMPLATE_T_F(TransportBufferRead)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedRecv(T &obj, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar<T, F>(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> Recv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P_F(TransportRecv)
        inline enable_if_pointer<P> Recv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packRootPtr<P, F>(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferRead)
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootPtr<P, F>(ptr);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> Recv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P_F(TransportRecv)
        inline enable_if_pointer<P> Recv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr<P, F>(ptr, len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> Recv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::Recv(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);
        };

        TEMPLATE_P_F(TransportRecv)
        inline enable_if_pointer<P> Recv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            Message<TransportRecv, HASH_MAP> msg(src, tag, comm);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::Recv(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr<P, F>(ptr, _len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferRead)
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr<P, F>(ptr, len);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedRecv(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferRead)
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int const &len, const int src, const int tag, const Comm &comm) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::Recv(buffer, bufferSize, src, tag, comm);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedRecv(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr<P, F>(ptr, _len);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Bcast
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S> Bcast(S &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootSTL(obj);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                msg.packRootSTL(obj);
            }
        };

        TEMPLATE_STL_F2(TransportBcastRoot, TransportBcast)
        inline enable_if_stl<S> Bcast(S &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootSTL<S, F1>(obj);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                msg.packRootSTL<S, F2>(obj);
            }
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedBcast(S &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootSTL(obj);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                msg.packRootSTL(obj);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_STL_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_stl<S> BufferedBcast(S &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootSTL<S, F1>(obj);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                msg.packRootSTL<S, F2>(obj);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedBcast(S &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast(obj, root, comm, 0);
            }
        };

        TEMPLATE_STL_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_stl<S> BufferedBcast(S &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast<S, HASH_MAP, F1>(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast<S, HASH_MAP, F2>(obj, root, comm, 0);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> Bcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootVar(obj);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                msg.packRootVar(obj);
            }
        };

        TEMPLATE_T_F2(TransportBcastRoot, TransportBcast)
        inline enable_if_not_pointer_not_stl<T> Bcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootVar<T, F1>(obj);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                msg.packRootVar<T, F2>(obj);
            }
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedBcast(T &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar(obj);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                msg.packRootVar(obj);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_T_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedBcast(T &obj, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar<T, F1>(obj);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                msg.packRootVar<T, F2>(obj);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedBcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast(obj, root, comm, 0);
            }
        };

        TEMPLATE_T_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedBcast(T &obj, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast<T, HASH_MAP, F1>(obj, root, comm, MEL::Deep::BufferSize(obj));
            }
            else {
                MEL::Deep::BufferedBcast<T, HASH_MAP, F2>(obj, root, comm, 0);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> Bcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootPtr(ptr);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packRootPtr(ptr);
            }
        };

        TEMPLATE_P_F2(TransportBcastRoot, TransportBcast)
        inline enable_if_pointer<P> Bcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootPtr<P, F1>(ptr);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packRootPtr<P, F2>(ptr);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packRootPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootPtr<P, F1>(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packRootPtr<P, F2>(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, root, comm, MEL::Deep::BufferSize(ptr));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, root, comm, 0);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast<P, HASH_MAP, F1>(ptr, root, comm, MEL::Deep::BufferSize(ptr));
            }
            else {
                MEL::Deep::BufferedBcast<P, HASH_MAP, F2>(ptr, root, comm, 0);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> Bcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                int _len = len;
                msg.packRootVar(_len);
                msg.packRootPtr(ptr, _len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                int _len = len;
                msg.packRootVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::Bcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packRootPtr(ptr, _len);
            }
        };

        TEMPLATE_P_F2(TransportBcastRoot, TransportBcast)
        inline enable_if_pointer<P> Bcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                int _len = len;
                msg.packRootVar(_len);
                msg.packRootPtr<P, F1>(ptr, _len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                int _len = len;
                msg.packRootVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::Bcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packRootPtr<P, F2>(ptr, _len);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> Bcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootVar(len);
                msg.packRootPtr(ptr, len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packRootVar(len);
                msg.packRootPtr(ptr, len);
            }
        };

        TEMPLATE_P_F2(TransportBcastRoot, TransportBcast)
        inline enable_if_pointer<P> Bcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                Message<TransportBcastRoot, HASH_MAP> msg(root, comm);
                msg.packRootVar(len);
                msg.packRootPtr<P, F1>(ptr, len);
            }
            else {
                Message<TransportBcast, HASH_MAP> msg(root, comm);
                ptr = (P) 0x1;
                msg.packRootVar(len);
                msg.packRootPtr<P, F2>(ptr, len);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar(len); 
                msg.packRootPtr(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packRootVar(len); 
                msg.packRootPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar(len); 
                msg.packRootPtr<P, F1>(ptr);

                MEL::Deep::Bcast(buffer, bufferSize, root, comm);

                MEL::MemFree(buffer);
            }
            else {
                int _bufferSize;
                char *buffer = nullptr;
                MEL::Deep::Bcast(buffer, _bufferSize, root, comm);

                Message<TransportBufferRead, HASH_MAP> msg(buffer, _bufferSize);
                ptr = (P) 0x1;
                msg.packRootVar(len); 
                msg.packRootPtr<P, F2>(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast<p, HASH_MAP, F1>(ptr, len, root, comm, MEL::Deep::BufferSize<P, HASH_MAP, F1>(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast<p, HASH_MAP, F2>(ptr, len, root, comm, 0);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar(len);
                msg.packRootPtr(ptr);

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
                msg.packRootVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedBcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packRootPtr(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm, const int bufferSize) {
            if (MEL::CommRank(comm) == root) {
                char *buffer = MEL::MemAlloc<char>(bufferSize);
                Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
                msg.packRootVar(len);
                msg.packRootPtr<P, F1>(ptr);

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
                msg.packRootVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedBcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packRootPtr<P, F2>(ptr);

                MEL::MemFree(buffer);
            }
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, MEL::Deep::BufferSize(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast(ptr, len, root, comm, 0);
            }
        };

        TEMPLATE_P_F2(TransportBufferWrite, TransportBufferRead)
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int const &len, const int root, const Comm &comm) {
            if (MEL::CommRank(comm) == root) {
                MEL::Deep::BufferedBcast<P, HASH_MAP, F1>(ptr, len, root, comm, MEL::Deep::BufferSize<P, HASH_MAP, F1>(ptr, len));
            }
            else {
                MEL::Deep::BufferedBcast<P, HASH_MAP, F2>(ptr, len, root, comm, 0);
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MPI_File Write
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S> FileWrite(S &obj, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL_F(TransportFileWrite)
        inline enable_if_stl<S> FileWrite(S &obj, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootSTL<S, F>(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileWrite(S &obj, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL_F(TransportBufferWrite)
        inline enable_if_stl<S> BufferedFileWrite(S &obj, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL<S, F>(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileWrite(S &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        TEMPLATE_STL_F(TransportBufferWrite)
        inline enable_if_stl<S> BufferedFileWrite(S &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite<S, HASH_MAP, F>(obj, file, MEL::Deep::BufferSize<S, HASH_MAP, F>(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> FileWrite(T &obj, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootVar(obj);
        };

        TEMPLATE_T_F(TransportFileWrite)
        inline enable_if_not_pointer_not_stl<T> FileWrite(T &obj, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootVar<T, F>(obj);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileWrite(T &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        TEMPLATE_T_F(TransportBufferWrite)
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileWrite(T &obj, MEL::File &file) {
            MEL::Deep::BufferedFileWrite<T, HASH_MAP, F>(obj, file, MEL::Deep::BufferSize<T, HASH_MAP, F>(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> FileWrite(P &ptr, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P_F(TransportFileWrite)
        inline enable_if_pointer<P> FileWrite(P &ptr, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootPtr<T, F>(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootPtr<T, F>(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file) {
            MEL::Deep::BufferedFileWrite<T, HASH_MAP, F>(ptr, file, MEL::Deep::BufferSize<T, HASH_MAP, F>(ptr));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> FileWrite(P &ptr, int const &len, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P_F(TransportFileWrite)
        inline enable_if_pointer<P> FileWrite(P &ptr, int const &len, MEL::File &file) {
            Message<TransportFileWrite, HASH_MAP> msg(file);
            msg.packRootVar(len);
            msg.packRootPtr<T, F>(ptr, len);
        };

        /// !!!

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(len);
            msg.packRootPtr<P, F>(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        TEMPLATE_P_F(TransportBufferWrite)
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, MEL::File &file) {
            MEL::Deep::BufferedFileWrite<P, HASH_MAP, F>(ptr, len, file, MEL::Deep::BufferSize<P, HASH_MAP, F>(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MPI_File Read
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S> FileRead(S &obj, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileRead(S &obj, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> FileRead(T &obj, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            msg.packRootVar(obj);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileRead(T &obj, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, int const &len, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, MEL::File &file) {
            Message<TransportFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int const &len, MEL::File &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL File Write
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL 

        TEMPLATE_STL
        inline enable_if_stl<S> FileWrite(S &obj, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileWrite(S &obj, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileWrite(S &obj, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> FileWrite(T &obj, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packRootVar(obj);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileWrite(T &obj, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(obj);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileWrite(T &obj, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(obj, file, MEL::Deep::BufferSize(obj));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> FileWrite(P &ptr, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootPtr(ptr);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, file, MEL::Deep::BufferSize(ptr));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> FileWrite(P &ptr, int const &len, std::ofstream &file) {
            Message<TransportSTLFileWrite, HASH_MAP> msg(file);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, std::ofstream &file, const int bufferSize) {
            char *buffer = MEL::MemAlloc<char>(bufferSize);
            Message<TransportBufferWrite, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::Deep::FileWrite(buffer, bufferSize, file);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, int const &len, std::ofstream &file) {
            MEL::Deep::BufferedFileWrite(ptr, len, file, MEL::Deep::BufferSize(ptr, len));
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL File Read
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // STL

        TEMPLATE_STL
        inline enable_if_stl<S> FileRead(S &obj, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            msg.packRootSTL(obj);
        };

        TEMPLATE_STL
        inline enable_if_stl<S> BufferedFileRead(S &obj, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootSTL(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Object

        TEMPLATE_T
        inline enable_if_not_pointer_not_stl<T> FileRead(T &obj, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            msg.packRootVar(obj);
        };

        TEMPLATE_T
        inline enable_if_deep_not_pointer_not_stl<T> BufferedFileRead(T &obj, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            msg.packRootVar(obj);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootPtr(ptr);

            MEL::MemFree(buffer);
        };

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointer / Length

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, int const &len, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, std::ifstream &file) {
            Message<TransportSTLFileRead, HASH_MAP> msg(file);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            ptr = (P) 0x1;
            msg.packRootVar(len);
            msg.packRootPtr(ptr, len);

            MEL::MemFree(buffer);
        };

        TEMPLATE_P
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, std::ifstream &file) {
            int bufferSize;
            char *buffer = nullptr;
            MEL::Deep::FileRead(buffer, bufferSize, file);

            Message<TransportBufferRead, HASH_MAP> msg(buffer, bufferSize);
            int _len = len;
            ptr = (P) 0x1;
            msg.packRootVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packRootPtr(ptr, _len);

            MEL::MemFree(buffer);
        };

#undef TEMPLATE_STL
#undef TEMPLATE_T
#undef TEMPLATE_P

#undef TEMPLATE_STL_F
#undef TEMPLATE_T_F
#undef TEMPLATE_P_F
    };
};