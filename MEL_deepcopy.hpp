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

/**
* \file MEL_deepcopy.hpp
*/

namespace MEL {
    namespace Deep {

        /**
         * \defgroup Deep Deep Copy Extensions
         * Extensions to MEL that leverage template meta programming to implement efficient Deep Copy Semantics. Algorithm discussed in the paper "Implementing Generalized Deep Copy in MPI," Joss Whittle, Rita Borgo, Mark Jones (Currently under review).
         *
         * \defgroup DeepMsg Deep Copy Message Transport API
         *
         * \defgroup DeepUtils Utilities for Deep Copy
         *
         * \defgroup DeepP2P Send/Recv Deep Copy
         *
         * \defgroup DeepCOL Bcast Deep Copy
         *
         * \defgroup DeepFile File-IO Deep Copy
         *
         */
        
        /// \cond HIDE
        class Message;
        template<typename T>
        struct HasDeepCopyMethod {
            template<typename U, void(U::*)(MEL::Deep::Message&)> struct SFINAE {};
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
        /// \endcond

        class Message {
        public:
            
            /// \cond HIDE
            enum class Mode {
                P2P        = 0x1,
                Collective = 0x2,
                MEL_File   = 0x4,
                STL_File   = 0x8
            };
            /// \endcond

        private:
            /// \cond HIDE
            /// Members
            const int pid, tag;
            const Mode mode;
            const Comm comm;
            const bool source, buffered;

            char *buffer;
            int offset, bufferSize;
            void *filePtr;
            
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
            /// \endcond

            /**
             * \ingroup  Deep
             * Is the message representing a buffered operation? 
             *
             * \return		Returns true if the message object represents a buffered operation
             */
            inline bool isBuffered() const {
                return buffered;
            };

            /**
             * \ingroup  Deep
             * Is the message representing a collective operation? 
             *
             * \return		Returns true if the message object represents a collective operation
             */
            inline bool isCollective() const {
                return mode == Message::Mode::Collective;
            };

            /**
             * \ingroup  Deep
             * Is the message representing a point-2-point operation? 
             *
             * \return		Returns true if the message object represents a point-2-point operation
             */
            inline bool isP2P() const {
                return mode == Message::Mode::P2P;
            };

            /**
             * \ingroup  Deep
             * Is the message representing a file access operation? 
             *
             * \return		Returns true if the message object represents a file access operation
             */
            inline bool isMELFile() const {
                return mode == Message::Mode::MEL_File;
            };

            /**
             * \ingroup  Deep
             * Is the message representing a stl file access operation? 
             *
             * \return		Returns true if the message object represents an stl file access operation
             */
            inline bool isSTLFile() const {
                return mode == Message::Mode::STL_File;
            };

            /**
             * \ingroup  Deep
             * Is the local process the sender? 
             *
             * \return		Returns true if the local process is the sender
             */
            inline bool isSource() const {
                return source;
            };

            /**
             * \ingroup  Deep
             * Does the message currently have a buffer attached
             *
             * \return		Returns true if the message has an allocated internal buffer
             */
            inline bool hasBuffer() const {
                return buffer != nullptr;
            };

            /**
             * \ingroup  Deep
             * Add a non-deep object reference to the the messages buffer
             *
             * \param[in,out] obj	The object reference to put/get from the buffer
             */
            template<typename T>
            inline void BufferVar(T &obj) {
                const int num = sizeof(T);
                if (hasBuffer()) {
                    if (isSource()) {
                        memcpy(&buffer[offset], &obj, num);
                    }
                    else {
                        memcpy(&obj, &buffer[offset], num);
                    }
                }
                offset += num;
            };
            
            /**
             * \ingroup  Deep
             * Add a non-deep array to the the messages buffer
             *
             * \param[in,out] ptr	The pointer to the array to put/get from the buffer
             * \param[in]	  len	The number of elements to put/get
             */
            template<typename T>
            inline void BufferPtr(T *&ptr, int len) {
                const int num = sizeof(T) * len;
                if (hasBuffer() && len > 0) {
                    if (isSource()) {
                        memcpy(&buffer[offset], ptr, num);
                    }
                    else {
                        memcpy(ptr, &buffer[offset], num);
                    }
                }
                offset += num;
            };

            /**
             * \ingroup  Deep
             * Check if a pointer is currently in the map of shared pointer
             *
             * \param[in] ptr	The pointer to the array to check
             * \return				Returns true if the pointer is already in the map
             */
            template<typename T>
            inline bool CheckPointerCache(T* &ptr) {
                return pointerMap.find(HashPtr(ptr)) != pointerMap.end();
            };

            /**
             * \ingroup  Deep
             * Insert a pointer into map of shared pointers
             *
             * \param[in] oldPtr	The old "dangling" pointer
             * \param[in] ptr		The newly allocated pointer
             */
            template<typename T>
            inline void CachePointer(T* oldPtr, T* ptr) {
                pointerMap.insert(std::make_pair(HashPtr(oldPtr), (void*) ptr));
            };

            /**
             * \ingroup  Deep
             * Replace the value of the given pointer with the value mapped to it in the shared pointer map
             *
             * \param[in,out] ptr	The old "dangling" pointer to be modified
             */
            template<typename T>
            inline void GetCachedPointer(T* &ptr) {
                auto it = pointerMap.find(HashPtr(ptr));
                if (it != pointerMap.end()) {
                    ptr = (T*) it->second;
                }
            };

            /**
             * \ingroup  Deep
             * Send an array to the receiving process where it will need to be allocated, or if the message is buffered pack the array into the buffer
             *
             * \param[in] src		The pointer to the array to send
             * \param[in] len		The number of elements to send
             */
            template<typename T>
            inline void SendAlloc(T* &src, int len) {
                if (src != nullptr) {
                    if (isBuffered()) {
                        BufferPtr(src, len);
                    }
                    else {
                        MEL::Send<T>(src, len, pid, tag, comm);
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Receive an array from the sending process, or if the message is buffered unpack the array from the buffer. An allocation will be made before receiving the data
             *
             * \param[out] dst		The pointer to the array to receive
             * \param[in] len		The number of elements to receive
             */
            template<typename T>
            inline void RecvAlloc(T* &dst, int len) {
                if (dst != nullptr) {
                    dst = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                        
                    if (isBuffered()) {
                        BufferPtr(dst, len);
                    }
                    else {
                        MEL::Recv<T>(dst, len, pid, tag, comm);
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Write an array to file, or if the message is buffered pack the array into the buffer
             *
             * \param[in] src		The pointer to the array to write
             * \param[in] len		The number of elements to write
             */
            template<typename T>
            inline void MELWriteAlloc(T* &src, int len) {
                if (src != nullptr) {
                    if (isBuffered()) {
                        BufferPtr(src, len);
                    }
                    else {
                        MEL::FileWrite<T>(*((MEL::File*) filePtr), src, len);
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Read an array from file, or if the message is buffered unpack the array from the buffer. An allocation will be made before receiving the data
             *
             * \param[out] dst		The pointer to the array to read
             * \param[in] len		The number of elements to read
             */
            template<typename T>
            inline void MELReadAlloc(T* &dst, int len) {
                if (dst != nullptr) {					
                    dst = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                    
                    if (isBuffered()) {
                        BufferPtr(dst, len);
                    }
                    else {
                        MEL::FileRead<T>(*((MEL::File*) filePtr), dst, len);
                    }
                    
                }
            };

            /**
             * \ingroup  Deep
             * Write an array to an stl file, or if the message is buffered pack the array into the buffer
             *
             * \param[in] src		The pointer to the array to write
             * \param[in] len		The number of elements to write
             */
            template<typename T>
            inline void STLWriteAlloc(T* &src, int len) {
                if (src != nullptr) {
                    if (isBuffered()) {
                        BufferPtr(src, len);
                    }
                    else {
                        ((std::ofstream*) filePtr)->write((const char*) src, len * sizeof(T));
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Read an array from an stl file, or if the message is buffered unpack the array from the buffer. An allocation will be made before receiving the data
             *
             * \param[out] dst		The pointer to the array to read
             * \param[in] len		The number of elements to read
             */
            template<typename T>
            inline void STLReadAlloc(T* &dst, int len) {
                if (dst != nullptr) {					
                    dst = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                    
                    if (isBuffered()) {
                        BufferPtr(dst, len);
                    }
                    else {  
                        ((std::ifstream*) filePtr)->read((char*) dst, len * sizeof(T));
                    }
                    
                }
            };

            /**
             * \ingroup  Deep
             * Broadcast an array, or if the message is buffered pack/unpack the array to/from the buffer. An allocation will be made before receiving the data
             *
             * \param[in,out] buf		The pointer to the array to broadcast
             * \param[in] len			The number of elements to broadcast
             */
            template<typename T>
            inline void BcastAlloc(T* &buf, int len) {
                if (buf != nullptr) {
                    if (isSource()) {
                        if (isBuffered()) {
                            BufferPtr(buf, len);
                        }
                        else {
                            MEL::Bcast<T>(buf, len, pid, comm);
                        }
                    }
                    else {
                        buf = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;
                            
                        if (isBuffered()) {
                            BufferPtr(buf, len);
                        }
                        else {
                            MEL::Bcast<T>(buf, len, pid, comm);
                        }
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Abstract the transportation of a non-deep object reference by whatever means the message is using
             *
             * \param[in,out] obj	The object to be transported
             */
            template<typename T>
            inline void Transport(T &obj) {
                if (isBuffered()) {
                    BufferVar(obj);
                }
                else if (isCollective()) {
                    MEL::Bcast<T>(&obj, 1, pid, comm); 
                }
                else if (isP2P()) {
                    if (isSource()) {
                        MEL::Send<T>(&obj, 1, pid, tag, comm); 
                    }
                    else {
                        MEL::Recv<T>(&obj, 1, pid, tag, comm); 
                    }
                }
                else if (isMELFile()) {
                    if (isSource()) {
                        MEL::FileWrite<T>(*((MEL::File*) filePtr), &obj, 1);
                    }
                    else {
                        MEL::FileRead<T>(*((MEL::File*) filePtr), &obj, 1);
                    }
                }
                else if (isSTLFile()) {
                    if (isSource()) {
                        ((std::ofstream*) filePtr)->write((const char*) &obj, sizeof(T));
                    }
                    else {
                        ((std::ifstream*) filePtr)->read((char*) &obj, sizeof(T));
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Abstract the transportation of a non-deep object array by whatever means the message is using
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename T>
            inline void Transport(T *ptr, int len) {
                if (isBuffered()) {
                    BufferPtr(ptr, len);
                }
                else if (isCollective()) {
                    MEL::Bcast<T>(ptr, len, pid, comm); 
                }
                else if (isP2P()) {
                    if (isSource()) {
                        MEL::Send<T>(ptr, len, pid, tag, comm); 
                    }
                    else {
                        MEL::Recv<T>(ptr, len, pid, tag, comm); 
                    }
                }
                else if (isMELFile()) {
                    if (isSource()) { 
                        MEL::FileWrite<T>(*((MEL::File*) filePtr), ptr, len);
                    }
                    else {
                        MEL::FileRead<T>(*((MEL::File*) filePtr), ptr, len);
                    }
                }
                else if (isSTLFile()) {
                    if (isSource()) {
                        ((std::ofstream*) filePtr)->write((const char*) ptr, len * sizeof(T));
                    }
                    else {
                        ((std::ifstream*) filePtr)->read((char*) ptr, len * sizeof(T));
                    }
                }
            };

            /**
             * \ingroup  Deep
             * Abstract the transportation of a non-deep object array by whatever means the message is using. Allocations will be made before receiving the data
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename T>
            inline void TransportAlloc(T *&ptr, int len) {
                if (isCollective()) {
                    BcastAlloc(ptr, len);
                }
                else if (isP2P()) {
                    if (isSource()) {
                        SendAlloc(ptr, len);
                    }
                    else {
                        RecvAlloc(ptr, len);
                    }
                }
                else if (isMELFile()) {
                    if (isSource()) {
                        MELWriteAlloc(ptr, len);
                    }
                    else {
                        MELReadAlloc(ptr, len);
                    }
                }
                else if (isSTLFile()) {
                    if (isSource()) {
                        STLWriteAlloc(ptr, len);
                    }
                    else {
                        STLReadAlloc(ptr, len);
                    }
                }
            };

        public:
            
            /// \cond HIDE
            /// Internal helper - Don't call this yourself!
            inline int _GetOffset() const {
                return offset;
            };

            /// Internal helper - Don't call this yourself!
            inline void _BufferAlloc(const int size) {
                _BufferFree();
                bufferSize = size;
                buffer = MEL::MemAlloc<char>(bufferSize);
            };

            /// Internal helper - Don't call this yourself!
            inline void _BufferProbeAlloc() {
                _BufferFree();
                bufferSize = MEL::ProbeGetCount<char>(pid, tag, comm);
                buffer = MEL::MemAlloc<char>(bufferSize);
            };

            /// Internal helper - Don't call this yourself!
            inline void _BufferFree() {
                MEL::MemFree(buffer);
                offset = 0;
                bufferSize = 0;
            };

            /// Internal helper - Don't call this yourself!
            inline void _BufferTransport() {
                if (isCollective()) {
                    MEL::Bcast(buffer, bufferSize, pid, comm); 
                }
                else if (isP2P()) {
                    if (isSource()) {
                        MEL::Send(buffer, bufferSize, pid, tag, comm); 
                    }
                    else {
                        MEL::Recv(buffer, bufferSize, pid, tag, comm); 
                    }
                }
                else if (isMELFile()) {
                    if (isSource()) {
                        MEL::FileWrite(*((MEL::File*) filePtr), buffer, bufferSize);
                    }
                    else {
                        MEL::FileRead(*((MEL::File*) filePtr), buffer, bufferSize);
                    }
                }
                else if (isSTLFile()) {
                    if (isSource()) {
                        ((std::ofstream*) filePtr)->write((const char*) buffer, bufferSize);
                    }
                    else {
                        ((std::ifstream*) filePtr)->read((char*) buffer, bufferSize);  
                    }
                }
            };

            /// Internal helper - Don't call this yourself!
            inline void _FileAttach(MEL::File *ptr) {
                filePtr = ptr;
            };
            inline void _FileAttach(std::ofstream *ptr) {
                filePtr = ptr;
            };
            inline void _FileAttach(std::ifstream *ptr) {
                filePtr = ptr;
            };
            inline void _FileDetach() {
                filePtr = nullptr;
            };

            /// Constructor
            Message() = delete;
            Message(const Message &) = delete;
            Message& operator=(const Message &) = delete;
            Message(Message &&) = delete;
            Message& operator=(Message &&) = delete;

            /// Message is immutable as far as the user is concerned
            Message(const int _pid, const int _tag, const Comm _comm, const bool _src, const Mode _mode, const bool _buf)
                : pid(_pid), tag(_tag), comm(_comm), source(_src), mode(_mode), buffered(_buf), buffer(nullptr), offset(0) {};
            
            /// \endcond

            /**
             * \ingroup  DeepMsg
             *
             * Copies the footprint of an object as is
             *
             * \param[in,out] obj	The object to be transported
             */
            template<typename T>
            inline enable_if_not_deep<T> packVar(T &obj) {
                Transport(obj);
            };

            /**
             * \ingroup  DeepMsg
             *
             * Transport a deep object reference
             *
             * \param[in,out] obj	The object to be transported
             */
            template<typename D>
            inline enable_if_deep<D> packVar(D &obj) {
                Transport(obj);
                obj.DeepCopy(*this);
            };

            /**
             * \ingroup  DeepMsg
             * Transport an array by its contiguous footprint in memory
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename T>
            inline enable_if_not_deep<T> packPtr(T* &ptr, int len = 1) {
                TransportAlloc(ptr, len);
            };

            /**
             * \ingroup  DeepMsg
             * Transport a deep array
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename D>
            inline enable_if_deep<D> packPtr(D* &ptr, int len = 1) {
                TransportAlloc(ptr, len);
                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            /**
             * \ingroup  DeepMsg
             * Transport a (potentially shared) array by its contiguous footprint in memory
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename T>
            inline enable_if_not_deep<T> packSharedPtr(T* &ptr, int len = 1) {
                T *oldPtr = ptr;
                if (CheckPointerCache(ptr)) {
                    if (!isSource()) {
                        GetCachedPointer(ptr);
                    }
                    return;
                }

                TransportAlloc(ptr, len);

                CachePointer(oldPtr, ptr);
            };

            /**
             * \ingroup  DeepMsg
             * Transport a (potentially shared) deep array
             *
             * \param[in,out] ptr	Pointer to the array to be transported
             * \param[in] len		The number of elements to transport
             */
            template<typename D>
            inline enable_if_deep<D> packSharedPtr(D* &ptr, int len = 1) {
                D *oldPtr = ptr;
                if (CheckPointerCache(ptr)) {
                    if (!isSource()) {
                        GetCachedPointer(ptr);
                    }
                    return;
                }

                TransportAlloc(ptr, len);

                CachePointer(oldPtr, ptr);

                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::string
             *
             * \param[in,out] obj	The std::string to transport
             */
            inline void packSTL(std::string &obj) {
                int len;
                if (isSource()) len = obj.size();
                Transport(len);
                if (!isSource()) new (&obj) std::string(len, ' ');

                char *p = &obj[0];
                if (len > 0) Transport(p, len);
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::vector
             *
             * \param[in,out] obj	The std::vector to transport
             */
            template<typename T>
            inline enable_if_not_deep<T> packSTL(std::vector<T> &obj) {
                //int len;
                //if (isSource()) len = obj.size();
                //Transport(len);

                int len = obj.size();
                if (!isSource()) new (&obj) std::vector<T>(len, T());
                
                T *p = &obj[0];
                if (len > 0) Transport(p, len); 
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::vector of deep objects
             *
             * \param[in,out] obj	The std::vector to transport
             */
            template<typename D>
            inline enable_if_deep<D> packSTL(std::vector<D> &obj) {
                //int len;
                //if (isSource()) len = obj.size();
                //Transport(len);

                int len = obj.size();
                if (!isSource()) new (&obj) std::vector<D>(len, D());
                
                D *p = &obj[0];
                if (len > 0) Transport(p, len);
                /// Copy content
                for (int i = 0; i < len; ++i) {
                    obj[i].DeepCopy(*this);
                }
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::list
             *
             * \param[in,out] obj	The std::list to transport
             */
            template<typename T>
            inline enable_if_not_deep<T> packSTL(std::list<T> &obj) {
                int len;
                if (isSource()) len = obj.size();
                Transport(len);
                if (!isSource()) new (&obj) std::list<T>(len, T());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                }
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::list of deep objects
             *
             * \param[in,out] obj	The std::list to transport
             */
            template<typename D>
            inline enable_if_deep<D> packSTL(std::list<D> &obj) {
                int len;
                if (isSource()) len = obj.size();
                Transport(len);
                if (!isSource()) new (&obj) std::list<D>(len, D());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    it->DeepCopy(*this);
                }
            };

            /**
             * \ingroup  DeepMsg
             * Transport a std::string
             *
             * \param[in,out] obj	The std::string to transport
             */
            inline Message& operator&(std::string &obj) {
                packSTL(obj);
                return *this;
            };

            /**
             * \ingroup  DeepMsg
             * Transport a deep/non-deep std::vector
             *
             * \param[in,out] obj	The std::vector to transport
             */
            template<typename T>
            inline Message& operator&(std::vector<T> &obj) {
                packSTL(obj);
                return *this;
            };

            /**
             * \ingroup  DeepMsg
             * Transport a deep/non-deep std::list
             *
             * \param[in,out] obj	The std::list to transport
             */
            template<typename T>
            inline Message& operator&(std::list<T> &obj) {
                packSTL(obj);
                return *this;
            };

            /**
             * \ingroup  DeepMsg
             * Transport a deep/non-deep object reference
             *
             * \param[in,out] obj	The object to transport
             */
            template<typename T>
            inline Message& operator&(T &obj) {
                packVar(obj);
                return *this;
            };
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        /**
         * \ingroup  DeepUtils
         * Compute the buffer size needed to transport a deep object reference
         *
         * \param[in] obj	The deep object to transport
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T, int> BufferSize(T &obj) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg & obj;
            return msg._GetOffset();
        };

        /**
         * \ingroup  DeepUtils
         * Compute the buffer size needed to transport a pointer to a single deep/non-deep variable
         *
         * \param[in] ptr	Pointer to the object to be transported
         */
        template<typename P>
        inline enable_if_pointer<P, int> BufferSize(P &ptr) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            return msg._GetOffset();
        };

        /**
         * \ingroup  DeepUtils
         * Compute the buffer size needed to transport a pointer to an array of deep/non-deep variables
         *
         * \param[in] ptr	Pointer to the array to be transported
         * \param[in] len	The number of elements to be transported
         */
        template<typename P>
        inline enable_if_pointer<P, int> BufferSize(P &ptr, const int len) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg.packVar(len);
            msg.packPtr(ptr, len);
            return msg._GetOffset();
        };

        /**
         * \ingroup  DeepP2P
         * Send a deep object reference
         *
         * \param[in] obj	The deep object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_not_pointer<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg & obj;
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to a deep/non-deep object
         *
         * \param[in] ptr	Pointer to the object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Send(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg.packPtr(ptr);
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to an array of deep/non-deep objects
         *
         * \param[in] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to send
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Send(P &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        /**
         * \ingroup  DeepP2P
         * Send a deep object reference using a buffered send. Buffersize must be calculated ahead of time
         *
         * \param[in] obj	The deep object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg & obj;
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Send a deep object reference using a buffered send. Buffersize is calculated before transport
         *
         * \param[in] obj	The deep object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg & obj;
            bufferSize = msg._GetOffset();
            BufferedSend(obj, dst, tag, comm, bufferSize);
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to a deep/non-deep object using a buffered send. Buffersize must be calculated ahead of time
         *
         * \param[in] ptr	Pointer to the object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packPtr(ptr);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to a deep/non-deep object using a buffered send. Buffersize is calculated before transport
         *
         * \param[in] ptr	Pointer to the object to transport
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            bufferSize = msg._GetOffset();
            BufferedSend(ptr, dst, tag, comm, bufferSize);
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to an array of deep/non-deep objects using a buffered send. Buffersize must be calculated ahead of time
         *
         * \param[in] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to send
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Send a pointer to an array of deep/non-deep objects using a buffered send. Buffersize is calculated before transport
         *
         * \param[in] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to send
         * \param[in] dst	The rank of the destination process
         * \param[in] tag	The message tag to send with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedSend(P &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg.packVar(len);
            msg.packPtr(ptr, len);
            bufferSize = msg._GetOffset();
            BufferedSend(ptr, len, dst, tag, comm, bufferSize);
        };

        /**
         * \ingroup  DeepP2P
         * Receive a deep object reference
         *
         * \param[out] obj	The deep object to transport
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_not_pointer<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            msg & obj;
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to a deep/non-deep object
         *
         * \param[out] ptr	Pointer to the object to transport
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to an array of deep/non-deep objects
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[out] len	The number of elements that were received
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to an array of deep/non-deep objects
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to receive
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Recv(P &ptr, const int len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            int _len = len;
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::Recv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };

        /**
         * \ingroup  DeepP2P
         * Receive a deep object reference using a buffered receive. Buffersize determined by probing the incoming message
         *
         * \param[out] obj	The deep object to transport
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedRecv(T &obj, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            msg & obj;
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to a deep/non-deep object using a buffered receive. Buffersize determined by probing the incoming message
         *
         * \param[out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packPtr(ptr);
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to an array of deep/non-deep objects using a buffered receive. Buffersize determined by probing the incoming message
         *
         * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[out] len	The number of elements that were received
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepP2P
         * Receive a pointer to an array of deep/non-deep objects using a buffered receive. Buffersize determined by probing the incoming message
         *
         * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[in] len	The number of elements to receive
         * \param[in] src	The rank of the source process
         * \param[in] tag	The message tag to receive with
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedRecv(P &ptr, const int len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);
            int _len = len;
            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedRecv(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a deep object reference
         *
         * \param[in,out] obj	The deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_not_pointer<T> Bcast(T &obj, const int root, const Comm &comm) {
            Message msg(root, 0, comm, (MEL::CommRank(comm) == root), Message::Mode::Collective, false);
            msg & obj;
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (P) 0x1;
            msg.packPtr(ptr);
        };
        
        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to an array of deep/non-deep objects
         *
         * \param[in,out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[in] len	The number of elements to broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, const int len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (P) 0x1;
            int _len = len;
            msg.packVar(_len);
            if (!source && len != _len) MEL::Exit(-1, "MEL::Deep::Bcast(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
        };
        
        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to an array of deep/non-deep objects
         *
         * \param[in,out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[in,out] len	The number of elements to broadcast / that were broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> Bcast(P &ptr, int &len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a deep object reference using a buffered broadcast. Buffersize must be calculated ahead of time
         *
         * \param[in,out] obj	The deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg & obj;
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                /// Unpack the buffer on the receiver
                msg & obj;
            }

            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a deep object reference using a buffered broadcast. Buffersize is calculated before transport
         *
         * \param[in,out] obj	The deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedBcast(T &obj, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg & obj;
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
            BufferedBcast(obj, root, comm, bufferSize);
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize must be calculated ahead of time
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg.packPtr(ptr);
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                ptr = (P) 0x1;
                /// Unpack the buffer on the receiver
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize is calculated before transport
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg.packPtr(ptr);
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
                BufferedBcast(ptr, root, comm, bufferSize);
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize must be calculated ahead of time
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in,out] len	The number of elements to broadcast / that were broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg.packVar(len);
                msg.packPtr(ptr);
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                ptr = (P) 0x1;
                /// Unpack the buffer on the receiver
                msg.packVar(len);
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize is calculated before transport
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in,out] len	The number of elements to broadcast / that were broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, int &len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg.packVar(len);
                msg.packPtr(ptr);
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
            BufferedBcast(ptr, root, comm, bufferSize);
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize must be calculated ahead of time
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] len	The number of elements to broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int len, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            int _len = len;
            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg.packVar(_len);
                msg.packPtr(ptr);
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                ptr = (P) 0x1;
                /// Unpack the buffer on the receiver
                msg.packVar(_len);
                if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedBcast(ptr, len) const int len provided does not match incomming message size.");
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        /**
         * \ingroup  DeepCOL
         * Broadcast a pointer to a deep/non-deep object using a buffered broadcast. Buffersize is calculated before transport
         *
         * \param[in,out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] len	The number of elements to broadcast
         * \param[in] root	The rank of the source process
         * \param[in] comm	The comm world to transport within
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedBcast(P &ptr, const int len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int _len = len;
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg.packVar(_len);
                msg.packPtr(ptr);
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
            BufferedBcast(ptr, root, comm, bufferSize);
        };

        /**
         * \ingroup  DeepFile
         * Write a deep object reference to file
         *
         * \param[in] obj	The deep object to transport
         * \param[in] file	The file to write to
         */
        template<typename T>
        inline enable_if_not_pointer<T> FileWrite(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Write a pointer to a deep/non-deep object to file
         *
         * \param[in] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] file	The file to write to
         */
        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, false);
            msg._FileAttach(&file);
            msg.packPtr(ptr);
            msg._FileDetach();
        };
        
        /**
         * \ingroup  DeepFile
         * Write a pointer to an array of deep/non-deep objects to file
         *
         * \param[in] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[in] len	The number of elements to write
         * \param[in] file	The file to write to
         */
        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, const int len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, false);
            msg._FileAttach(&file); 
            msg.packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Write a deep object reference to file using a buffered write. Buffersize must be calculated ahead of time
         *
         * \param[in] obj	The deep object to transport
         * \param[in] file	The file to write to
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg & obj;
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Write a deep object reference to file using a buffered write. Buffersize is calculated before transport
         *
         * \param[in] obj	The deep object to transport
         * \param[in] file	The file to write to
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg & obj;
            bufferSize = msg._GetOffset();
            
            msg._FileDetach();
            BufferedFileWrite(obj, file, bufferSize);
        };

        /**
         * \ingroup  DeepFile
         * Write a pointer to a deep/non-deep object to file using a buffered write. Buffersize must be calculated ahead of time
         *
         * \param[in] ptr	Pointer to the object to transport
         * \param[in] file	The file to write to
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packPtr(ptr);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Write a pointer to a deep/non-deep object to file using a buffered write. Buffersize is calculated before transport
         *
         * \param[in] ptr	Pointer to the object to transport
         * \param[in] file	The file to write to
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            bufferSize = msg._GetOffset();

            msg._FileDetach();
            BufferedFileWrite(ptr, file, bufferSize);
        };

        /**
         * \ingroup  DeepFile
         * Write a pointer to an array of deep/non-deep objects to file using a buffered write. Buffersize must be calculated ahead of time
         *
         * \param[in] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to write
         * \param[in] file	The file to write to
         * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Write a pointer to an array of deep/non-deep objects to file using a buffered write. Buffersize is calculated before transport
         *
         * \param[in] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to write
         * \param[in] file	The file to write to
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::MEL_File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg.packVar(len);
            msg.packPtr(ptr, len);
            bufferSize = msg._GetOffset();
            
            msg._FileDetach();
            BufferedFileWrite(ptr, len, file, bufferSize);
        };

        /**
         * \ingroup  DeepFile
         * Read a deep object reference from file
         *
         * \param[out] obj	The deep object to transport
         * \param[in] file	The file to read from
         */
        template<typename T>
        inline enable_if_not_pointer<T> FileRead(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to a deep/non-deep object from file
         *
         * \param[out] ptr	Pointer to the deep/non-deep object to transport
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, false);
            msg._FileAttach(&file); 
            ptr = (P) 0x1;
            msg.packPtr(ptr);
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects from file
         *
         * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[in] len	The number of elements to read
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, const int len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, false);
            int _len = len;
            msg._FileAttach(&file);
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects from file
         *
         * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
         * \param[out] len	The number of elements that were read
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, false);
            msg._FileAttach(&file); 
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a deep object reference to file using a buffered read. Buffersize must be calculated ahead of time
         *
         * \param[out] obj	The deep object to transport
         * \param[in] file	The file to read from
         * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            msg & obj;
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a deep object reference to file using a buffered read. Buffersize is calculated before transport
         *
         * \param[out] obj	The deep object to transport
         * \param[in] file	The file to read from
         */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file) {
            BufferedFileRead(obj, file, MEL::FileGetSize(file));
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to a deep/non-deep object to file using a buffered read. Buffersize must be calculated ahead of time
         *
         * \param[out] ptr	Pointer to the object to transport
         * \param[in] file	The file to read from
         * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packPtr(ptr);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to a deep/non-deep object to file using a buffered read. Buffersize is calculated before transport
         *
         * \param[out] ptr	Pointer to the object to transport
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, MEL::File &file) {
            BufferedFileRead(ptr, file, MEL::FileGetSize(file));
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects to file using a buffered read. Buffersize must be calculated ahead of time
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to read
         * \param[in] file	The file to read from
         * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, true);
            int _len = len;
            
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects to file using a buffered read. Buffersize is calculated before transport
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[in] len	The number of elements to read
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, MEL::File &file) {
            BufferedFileRead(ptr, len, file, MEL::FileGetSize(file));
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects to file using a buffered read. Buffersize must be calculated ahead of time
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[out] len	The number of elements that were read
         * \param[in] file	The file to read from
         * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::MEL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
         * \ingroup  DeepFile
         * Read a pointer to an array of deep/non-deep objects to file using a buffered read. Buffersize is calculated before transport
         *
         * \param[out] ptr	Pointer to the array of objects to transport
         * \param[out] len	The number of elements that were read
         * \param[in] file	The file to read from
         */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, MEL::File &file) {
            BufferedFileRead(ptr, len, file, MEL::FileGetSize(file));
        };

        /**
        * \ingroup  DeepFile
        * Write a deep object reference to an stl file
        *
        * \param[in] obj	The deep object to transport
        * \param[in] file	The file to write to
        */
        template<typename T>
        inline enable_if_not_pointer<T> FileWrite(T &obj, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to a deep/non-deep object to an stl file
        *
        * \param[in] ptr	Pointer to the deep/non-deep object to transport
        * \param[in] file	The file to write to
        */
        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            msg.packPtr(ptr);
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to an array of deep/non-deep objects to an stl file
        *
        * \param[in] ptr	Pointer to the array of deep/non-deep objects to transport
        * \param[in] len	The number of elements to write
        * \param[in] file	The file to write to
        */
        template<typename P>
        inline enable_if_pointer<P> FileWrite(P &ptr, const int len, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            msg.packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a deep object reference to an stl file using a buffered write. Buffersize must be calculated ahead of time
        *
        * \param[in] obj	The deep object to transport
        * \param[in] file	The file to write to
        * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg & obj;
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a deep object reference to an stl file using a buffered write. Buffersize is calculated before transport
        *
        * \param[in] obj	The deep object to transport
        * \param[in] file	The file to write to
        */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            int bufferSize;
            /// Determine the buffersize needed
            msg & obj;
            bufferSize = msg._GetOffset();

            msg._FileDetach();
            BufferedFileWrite(obj, file, bufferSize);
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to a deep/non-deep object to an stl file using a buffered write. Buffersize must be calculated ahead of time
        *
        * \param[in] ptr	Pointer to the object to transport
        * \param[in] file	The file to write to
        * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packPtr(ptr);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to a deep/non-deep object to an stl file using a buffered write. Buffersize is calculated before transport
        *
        * \param[in] ptr	Pointer to the object to transport
        * \param[in] file	The file to write to
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            int bufferSize;
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            bufferSize = msg._GetOffset();

            msg._FileDetach();
            BufferedFileWrite(ptr, file, bufferSize);
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to an array of deep/non-deep objects to an stl file using a buffered write. Buffersize must be calculated ahead of time
        *
        * \param[in] ptr	Pointer to the array of objects to transport
        * \param[in] len	The number of elements to write
        * \param[in] file	The file to write to
        * \param[in] bufferSize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, std::ofstream &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Write a pointer to an array of deep/non-deep objects to an stl file using a buffered write. Buffersize is calculated before transport
        *
        * \param[in] ptr	Pointer to the array of objects to transport
        * \param[in] len	The number of elements to write
        * \param[in] file	The file to write to
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileWrite(P &ptr, const int len, std::ofstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            int bufferSize;
            /// Determine the buffersize needed
            msg.packVar(len);
            msg.packPtr(ptr, len);
            bufferSize = msg._GetOffset();

            msg._FileDetach();
            BufferedFileWrite(ptr, len, file, bufferSize);
        };

        /**
        * \ingroup  DeepFile
        * Read a deep object reference from file
        *
        * \param[out] obj	The deep object to transport
        * \param[in] file	The file to read from
        */
        template<typename T>
        inline enable_if_not_pointer<T> FileRead(T &obj, std::ifstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to a deep/non-deep object from file
        *
        * \param[out] ptr	Pointer to the deep/non-deep object to transport
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, std::ifstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            ptr = (P) 0x1;
            msg.packPtr(ptr);
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects from file
        *
        * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
        * \param[in] len	The number of elements to read
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, const int len, std::ifstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, false);
            int _len = len;
            msg._FileAttach(&file);
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::FileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects from file
        *
        * \param[out] ptr	Pointer to the array of deep/non-deep objects to transport
        * \param[out] len	The number of elements that were read
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> FileRead(P &ptr, int &len, std::ifstream &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, false);
            msg._FileAttach(&file);
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a deep object reference to an stl file using a buffered read. Buffersize must be calculated ahead of time
        *
        * \param[out] obj	The deep object to transport
        * \param[in] file	The file to read from
        * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, std::ifstream &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            msg & obj;
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a deep object reference to an stl file using a buffered read. Buffersize is calculated before transport
        *
        * \param[out] obj	The deep object to transport
        * \param[in] file	The file to read from
        */
        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, std::ifstream &file) {
            std::streampos pos = file.tellg();
            file.seekg(0, std::ios::end);
            int fsize = (int) file.tellg();
            file.seekg(pos);
            
            BufferedFileRead(obj, file, fsize);
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to a deep/non-deep object to an stl file using a buffered read. Buffersize must be calculated ahead of time
        *
        * \param[out] ptr	Pointer to the object to transport
        * \param[in] file	The file to read from
        * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, std::ifstream &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packPtr(ptr);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to a deep/non-deep object to an stl file using a buffered read. Buffersize is calculated before transport
        *
        * \param[out] ptr	Pointer to the object to transport
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, std::ifstream &file) {
            std::streampos pos = file.tellg();
            file.seekg(0, std::ios::end);
            int fsize = (int) file.tellg();
            file.seekg(pos);
            
            BufferedFileRead(ptr, file, fsize);
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects to an stl file using a buffered read. Buffersize must be calculated ahead of time
        *
        * \param[out] ptr	Pointer to the array of objects to transport
        * \param[in] len	The number of elements to read
        * \param[in] file	The file to read from
        * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, std::ifstream &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, true);
            int _len = len;

            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(_len);
            if (len != _len) MEL::Exit(-1, "MEL::Deep::BufferedFileRead(ptr, len) const int len provided does not match incomming message size.");
            msg.packPtr(ptr, _len);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects to an stl file using a buffered read. Buffersize is calculated before transport
        *
        * \param[out] ptr	Pointer to the array of objects to transport
        * \param[in] len	The number of elements to read
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, const int len, std::ifstream &file) {
            std::streampos pos = file.tellg();
            file.seekg(0, std::ios::end);
            int fsize = (int) file.tellg();
            file.seekg(pos);
            
            BufferedFileRead(ptr, len, file, fsize);
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects to an stl file using a buffered read. Buffersize must be calculated ahead of time
        *
        * \param[out] ptr	Pointer to the array of objects to transport
        * \param[out] len	The number of elements that were read
        * \param[in] file	The file to read from
        * \param[in] buffersize	The buffer size needed to pack the entire structure contiguously
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, std::ifstream &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::STL_File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (P) 0x1;
            msg.packVar(len);
            msg.packPtr(ptr, len);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        /**
        * \ingroup  DeepFile
        * Read a pointer to an array of deep/non-deep objects to an stl file using a buffered read. Buffersize is calculated before transport
        *
        * \param[out] ptr	Pointer to the array of objects to transport
        * \param[out] len	The number of elements that were read
        * \param[in] file	The file to read from
        */
        template<typename P>
        inline enable_if_pointer<P> BufferedFileRead(P &ptr, int &len, std::ifstream &file) {
            std::streampos pos = file.tellg();
            file.seekg(0, std::ios::end);
            int fsize = (int) file.tellg();
            file.seekg(pos);
            
            BufferedFileRead(ptr, len, file, fsize);
        };
    };
};