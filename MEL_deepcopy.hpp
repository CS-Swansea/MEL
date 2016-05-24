#pragma once

#include "MEL.hpp"

#include <string>
#include <vector>
#include <list>
#include <iostream>

namespace MEL {
    namespace Deep {
        
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

        class Message {
        public:
            enum class Mode {
                P2P        = 0x1,
                Collective = 0x2,
                File       = 0x4
            };

        private:
            /// Members
            const int pid, tag;
            const Mode mode;
            const Comm comm;
            const bool source, buffered;

            char *buffer;
            int offset, bufferSize;
            MEL::File *filePtr;

            inline bool isBuffered() const {
                return buffered;
            };

            inline bool isCollective() const {
                return mode == Message::Mode::Collective;
            };

            inline bool isP2P() const {
                return mode == Message::Mode::P2P;
            };

            inline bool isFile() const {
                return mode == Message::Mode::File;
            };

            inline bool isSource() const {
                return source;
            };

            inline bool hasBuffer() const {
                return buffer != nullptr;
            };

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

            template<typename T>
            inline void WriteAlloc(T* &src, int len) {
                if (src != nullptr) {
                    if (isBuffered()) {
                        BufferPtr(src, len);
                    }
                    else {
                        MEL::FileWrite<T>(*filePtr, src, len);
                    }
                }
            };

            template<typename T>
            inline void ReadAlloc(T* &dst, int len) {
                if (dst != nullptr) {
                    dst = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;

                    if (isBuffered()) {
                        BufferPtr(dst, len);
                    }
                    else {
                        MEL::FileRead<T>(*filePtr, dst, len);
                    }
                }
            };

            template<typename T>
            inline void BcastAlloc(T* &buf, int len) {
                if (buf != nullptr) {
                    if (!isSource()) buf = (len > 0) ? MEL::MemAlloc<T>(len) : nullptr;

                    if (isBuffered()) {
                        BufferPtr(buf, len);
                    }
                    else {
                        MEL::Bcast<T>(buf, len, pid, comm); 
                    }
                }
            };

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
                else if (isFile()) {
                    if (isSource()) {
                        MEL::FileWrite<T>(*filePtr, &obj, 1);
                    }
                    else {
                        MEL::FileRead<T>(*filePtr, &obj, 1);
                    }
                }
            };
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
                else if (isFile()) {
                    if (isSource()) {
                        MEL::FileWrite<T>(*filePtr, ptr, len);
                    }
                    else {
                        MEL::FileRead<T>(*filePtr, ptr, len);
                    }
                }
            };
            template<typename T>
            inline void TransportAlloc(T *&buf, int len) {
                if (isCollective()) {
                    BcastAlloc(buf, len);
                }
                else if (isP2P()) {
                    if (isSource()) {
                        SendAlloc(buf, len);
                    }
                    else {
                        RecvAlloc(buf, len);
                    }
                }
                else if (isFile()) {
                    if (isSource()) {
                        WriteAlloc(buf, len);
                    }
                    else {
                        ReadAlloc(buf, len);
                    }
                }
            };

        public:
            
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
                else if (isFile()) {
                    if (isSource()) {
                        MEL::FileWrite(*filePtr, buffer, bufferSize);
                    }
                    else {
                        MEL::FileRead(*filePtr, buffer, bufferSize);
                    }
                }
            };

            /// Internal helper - Don't call this yourself!
            inline void _FileAttach(MEL::File *ptr) {
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

            /// Copies the footprint of an object as is
            ///
            template<typename T>
            inline enable_if_not_deep<T> _packVar(T &obj) {
                Transport(obj);
            };

            /// Deep copies the content of an object
            ///
            template<typename T>
            inline enable_if_deep<T> packVar(T &obj) {
                Transport(obj);
                obj.DeepCopy(*this);
            };

            /// Pack a ptr where the sender knows the length
            ///
            template<typename T>
            inline enable_if_not_deep<T> packPtr(T* &ptr, int len = 1) {
                TransportAlloc(ptr, len);
            };

            /// Pack a ptr where the sender knows the length
            /// and deep copy child elements
            ///
            template<typename T>
            inline enable_if_deep<T> packPtr(T* &ptr, int len = 1) {
                TransportAlloc(ptr, len);
                /// Copy elements
                if (ptr != nullptr) {
                    for (int i = 0; i < len; ++i) {
                        ptr[i].DeepCopy(*this);
                    }
                }
            };

            /// Pack an STL String
            ///
            inline void packSTL(std::string &obj) {
                int len;
                if (isSource()) len = obj.size();
                Transport(len);
                if (!isSource()) new (&obj) std::string(len, ' ');

                char *p = &obj[0];
                if (len > 0) Transport(p, len);
            };

            /// Pack an STL Vector
            ///
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

            /// Pack a deep STL Vector
            ///
            template<typename T>
            inline enable_if_deep<T> packSTL(std::vector<T> &obj) {
                //int len;
                //if (isSource()) len = obj.size();
                //Transport(len);

                int len = obj.size();
                if (!isSource()) new (&obj) std::vector<T>(len, T());
                
                T *p = &obj[0];
                if (len > 0) Transport(p, len);
                /// Copy content
                for (int i = 0; i < len; ++i) {
                    obj[i].DeepCopy(*this);
                }
            };

            /// Pack an STL List
            ///
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

            /// Pack a deep STL List
            ///
            template<typename T>
            inline enable_if_deep<T> packSTL(std::list<T> &obj) {
                int len;
                if (isSource()) len = obj.size();
                Transport(len);
                if (!isSource()) new (&obj) std::list<T>(len, T());
                /// Copy content
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    *this & *it;
                    it->DeepCopy(*this);
                }
            };

            /// Operator Overloads
            ///
            inline Message& operator&(std::string &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message& operator&(std::vector<T> &obj) {
                packSTL(obj);
                return *this;
            };

            template<typename T>
            inline Message& operator&(std::list<T> &obj) {
                packSTL(obj);
                return *this;
            };

            /*
            template<typename T>
            inline enable_if_not_deep<T, Message&> operator&(T &obj) {
                _packVar(obj);
                return *this;
            };
            */

            template<typename T>
            inline enable_if_deep<T, Message&> operator&(T &obj) {
                packVar(obj);
                return *this;
            };
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        /// ****************************************** ///
        /// Calculate Buffer Size                      ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T, int> BufferSize(T &obj) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg & obj;
            return msg._GetOffset();
        };

        template<typename T>
        inline enable_if_pointer<T, int> BufferSize(T &ptr) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            return msg._GetOffset();
        };

        template<typename T>
        inline enable_if_pointer<T, int> BufferSize(T &ptr, const int len) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::P2P, true);
            /// Determine the buffersize needed
            msg._packVar(len);
            msg.packPtr(ptr, len);
            return msg._GetOffset();
        };

        /// ****************************************** ///
        /// DEEP SEND                                  ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg & obj;
        };
        template<typename T>
        inline enable_if_not_deep_not_pointer<T> Send(T &obj, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg._packVar(obj);
        };
        template<typename T>
        inline enable_if_pointer<T> Send(T &ptr, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg.packPtr(ptr);
        };
        template<typename T>
        inline enable_if_pointer<T> Send(T &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, false);
            msg._packVar(len);
            msg.packPtr(ptr, len);
        };

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

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedSend(T &obj, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg & obj;
            bufferSize = msg._GetOffset();
            BufferedSend(obj, dst, tag, comm, bufferSize);
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedSend(T &ptr, const int dst, const int tag, const Comm &comm, const int bufferSize) {
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

        template<typename T>
        inline enable_if_pointer<T> BufferedSend(T &ptr, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            bufferSize = msg._GetOffset();
            BufferedSend(ptr, dst, tag, comm, bufferSize);
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedSend(T &ptr, const int len, const int dst, const int tag, const Comm &comm, const int bufferSize) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg._packVar(len);
            msg.packPtr(ptr, len);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedSend(T &ptr, const int len, const int dst, const int tag, const Comm &comm) {
            Message msg(dst, tag, comm, true, Message::Mode::P2P, true);
            int bufferSize;
            /// Determine the buffersize needed
            msg._packVar(len);
            msg.packPtr(ptr, len);
            bufferSize = msg._GetOffset();
            BufferedSend(ptr, len, dst, tag, comm, bufferSize);
        };

        /// ****************************************** ///
        /// DEEP RECV                                  ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            msg & obj;
        };

        template<typename T>
        inline enable_if_not_deep_not_pointer<T> Recv(T &obj, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            msg._packVar(obj);
        };

        template<typename T>
        inline enable_if_pointer<T> Recv(T &ptr, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            ptr = (T) 0x1;
            msg.packPtr(ptr);
        };
        template<typename T>
        inline enable_if_pointer<T> Recv(T &ptr, int &len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, false);
            ptr = (T) 0x1;
            msg._packVar(len);
            msg.packPtr(ptr, len);
        };

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

        template<typename T>
        inline enable_if_pointer<T> BufferedRecv(T &ptr, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (T) 0x1;
            msg.packPtr(ptr);
            /// Clean up
            msg._BufferFree();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedRecv(T &ptr, const int len, const int src, const int tag, const Comm &comm) {
            Message msg(src, tag, comm, false, Message::Mode::P2P, true);

            /// Allocate space for buffer
            msg._BufferProbeAlloc();
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (T) 0x1;
            msg._packVar(len);
            msg.packPtr(ptr, len);
            /// Clean up
            msg._BufferFree();
        };

        /// ****************************************** ///
        /// DEEP BCAST                                 ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T> Bcast(T &obj, const int root, const Comm &comm) {
            Message msg(root, 0, comm, (MEL::CommRank(comm) == root), Message::Mode::Collective, false);
            msg & obj;
        };
        template<typename T>
        inline enable_if_not_deep_not_pointer<T> Bcast(T &obj, const int root, const Comm &comm) {
            Message msg(root, 0, comm, (MEL::CommRank(comm) == root), Message::Mode::Collective, false);
            msg._packVar(obj);
        };

        template<typename T>
        inline enable_if_pointer<T> Bcast(T &ptr, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (T) 0x1;
            msg.packPtr(ptr);
        };
        template<typename T>
        inline enable_if_pointer<T> Bcast(T &ptr, const int _len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (T) 0x1;
            int len = _len;
            msg._packVar(len);
            msg.packPtr(ptr, len);
        };
        template<typename T>
        inline enable_if_pointer<T> Bcast(T &ptr, int &len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, false);
            if (!source) ptr = (T) 0x1;
            msg._packVar(len);
            msg.packPtr(ptr, len);
        };

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

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, const int root, const Comm &comm, const int bufferSize) {
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
                ptr = (T) 0x1;
                /// Unpack the buffer on the receiver
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, const int root, const Comm &comm) {
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

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, int &len, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg._packVar(len);
                msg.packPtr(ptr);
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                ptr = (T) 0x1;
                /// Unpack the buffer on the receiver
                msg._packVar(len);
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, int &len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg._packVar(len);
                msg.packPtr(ptr);
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
            BufferedBcast(ptr, root, comm, bufferSize);
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, const int _len, const int root, const Comm &comm, const int bufferSize) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);

            int len = _len;
            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            if (source) {
                /// Fill the buffer on the sender
                msg._packVar(len);
                msg.packPtr(ptr);
            }

            /// Share the buffer
            msg._BufferTransport();

            if (!source) {
                ptr = (T) 0x1;
                /// Unpack the buffer on the receiver
                msg._packVar(len);
                msg.packPtr(ptr);
            }

            /// Clean up
            msg._BufferFree();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedBcast(T &ptr, const int _len, const int root, const Comm &comm) {
            const bool source = (MEL::CommRank(comm) == root);
            Message msg(root, 0, comm, source, Message::Mode::Collective, true);
            int len = _len;
            int bufferSize;
            if (source) {
                /// Determine the buffersize needed
                msg._packVar(len);
                msg.packPtr(ptr);
                bufferSize = msg._GetOffset();
            }
            MEL::Bcast(&bufferSize, 1, root, comm); 
            BufferedBcast(ptr, root, comm, bufferSize);
        };

        /// ****************************************** ///
        /// DEEP File Write                            ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T> FileWrite(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };
        template<typename T>
        inline enable_if_not_deep_not_pointer<T> FileWrite(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, false);
            msg._FileAttach(&file); 
            msg._packVar(obj);
            msg._FileDetach();
        };
        template<typename T>
        inline enable_if_pointer<T> FileWrite(T &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, false);
            msg._FileAttach(&file);
            msg.packPtr(ptr);
            msg._FileDetach();
        };
        template<typename T>
        inline enable_if_pointer<T> FileWrite(T &ptr, const int len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, false);
            msg._FileAttach(&file); 
            msg._packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
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

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileWrite(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg & obj;
            bufferSize = msg._GetOffset();
            
            msg._FileDetach();
            BufferedFileWrite(obj, file, bufferSize);
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileWrite(T &ptr, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
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

        template<typename T>
        inline enable_if_pointer<T> BufferedFileWrite(T &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg.packPtr(ptr);
            bufferSize = msg._GetOffset();

            msg._FileDetach();
            BufferedFileWrite(ptr, file, bufferSize);
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileWrite(T &ptr, const int len, MEL::File &file, const int bufferSize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(bufferSize);
            /// Fill the buffer on the sender
            msg._packVar(len);
            msg.packPtr(ptr, len);
            /// Share the buffer
            msg._BufferTransport();
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileWrite(T &ptr, const int len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, true, Message::Mode::File, true);
            msg._FileAttach(&file); 
            
            int bufferSize;
            /// Determine the buffersize needed
            msg._packVar(len);
            msg.packPtr(ptr, len);
            bufferSize = msg._GetOffset();
            
            msg._FileDetach();
            BufferedFileWrite(ptr, len, file, bufferSize);
        };

        /// ****************************************** ///
        /// DEEP File Read                             ///
        /// ****************************************** ///

        template<typename T>
        inline enable_if_deep_not_pointer<T> FileRead(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, false);
            msg._FileAttach(&file);
            msg & obj;
            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_not_deep_not_pointer<T> FileRead(T &obj, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, false);
            msg._FileAttach(&file); 
            msg._packVar(obj);
            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_pointer<T> FileRead(T &ptr, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, false);
            msg._FileAttach(&file); 
            ptr = (T) 0x1;
            msg.packPtr(ptr);
            msg._FileDetach();
        };
        template<typename T>
        inline enable_if_pointer<T> FileRead(T &ptr, int &len, MEL::File &file) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, false);
            msg._FileAttach(&file); 
            ptr = (T) 0x1;
            msg._packVar(len);
            msg.packPtr(ptr, len);
            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, true);
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

        template<typename T>
        inline enable_if_deep_not_pointer<T> BufferedFileRead(T &obj, MEL::File &file) {
            BufferedFileRead(obj, file, MEL::FileGetSize(file));
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileRead(T &ptr, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (T) 0x1;
            msg.packPtr(ptr);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileRead(T &ptr, MEL::File &file) {
            BufferedFileRead(ptr, file, MEL::FileGetSize(file));
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileRead(T &ptr, const int len, MEL::File &file, const int buffersize) {
            Message msg(0, 0, MEL::Comm::COMM_NULL, false, Message::Mode::File, true);
            msg._FileAttach(&file);

            /// Allocate space for buffer
            msg._BufferAlloc(buffersize);
            /// Share the buffer
            msg._BufferTransport();
            /// Unpack the buffer 
            ptr = (T) 0x1;
            msg._packVar(len);
            msg.packPtr(ptr, len);
            /// Clean up
            msg._BufferFree();

            msg._FileDetach();
        };

        template<typename T>
        inline enable_if_pointer<T> BufferedFileRead(T &ptr, const int len, MEL::File &file) {
            BufferedFileRead(ptr, len, file, MEL::FileGetSize(file));
        };
    };
};