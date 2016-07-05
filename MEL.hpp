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

#include <mpi.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
#include <thread>

namespace MEL {
    
	/**
	 * \mainpage 
	 * ### Version 0.01 Beta
	 * MEL is a C++11, header-only library, being developed with the goal of creating a light weight and robust framework for building parallel applications on top of MPI. 
	 *  MEL is designed to introduce no (or minimal) overheads while drastically reducing code complexity. It allows for a greater range of common MPI errors to be caught at 
	 * compile-time rather than during program execution where it can be far more difficult to debug what is going wrong. 
	 *
	 * A good example of this is type safety in the MPI standard. The standard does not dictate how many of the object types should be implemented leaving these details to 
	 * the implementation vendor. For instance, in Intel MPI 5.1 `MPI_Comm` objects and many other simple types are implemented as indexes, `typedef int MPI_Comm`
	 * , leaving the implementation to use these indexes to manage the real objects internally. A drawback with this approach is it causes compile time type-checking of 
	 * function parameters to not flag erroneous combinations of variables. The common signature `MPI_Send(void*, int, MPI_Datatype, int, int, MPI_Comm)` is actually seen 
	 * by the compiler as `MPI_Send(void*, int, int, int, int, int)`, allowing any ordering of the last five variables to be compiled as valid MPI code, while causing 
	 * catastrophic failure at run-time. In contrast, OpenMPI 1.10.2 implements these types as structs which are inherently type-safe. 
	 *
	 * With MEL we aim to provide a consistent and unified function syntax that allows all MPI distributions to behave in a common and predictable way; while also providing
	 * some higher-level functionality that is not available from the MPI standard such as deep-copy, mutexes, RMA shared memory synchronization, and more. 
	 *
	 * We plan to keep MEL in active development and hope that the research community will join us as we continue to grow the features and capabilities encompassed within the project. 
	 * MEL is Open-Source and available on Github under the MIT license at: https://github.com/CS-Swansea/MEL .
	 *
	 * ## Todo
	 * 
	 * - Add Distributed Graph Topology functions.
	 * - Add overloads for p2p/collective communications for transmitting std::array/std::vector by start/end iterators.
	 * - Improve error handler implementation. A rough version is currently in place.
	 * - Implement ranged-mutexes. 
	 *
 	 * \defgroup Errors Error Handling
	 * Error Handler Creation / Deletion
	 *
	 * \defgroup Utils Utilities
	 * Utility Functions for Cleaner Coding
	 *
	 * \defgroup Mem Memory Allocation
	 * Dynamic Memory Allocation using the underlying MPI_Alloc allocator
	 *
	 * \defgroup Comm Communicators & Groups
	 * Communicator & Group Creation / Deletion
	 *
 	 * \defgroup Sync Synchronization
	 * Synchronization on Request objects
	 *
	 * \defgroup Datatype Derived Datatypes
	 * Derived Datatype Creation and Deletion
	 *
	 * \defgroup Topo Topology
	 * Cartesian & Distributed Graph Topologies
	 *
	 * \defgroup Ops Operations
	 * Builtin Functors and User Defined Operations
	 *
	 * \defgroup File File-IO
	 * File Creation / Deletion / Read / Write
	 *
	 * \defgroup P2P Point-2-Point Communication
	 * Send / Receive
	 *
	 * \defgroup COL Collective Communication
	 * Broadcast / Scatter / Gather / Alltoall / Reduce
	 *
	 * \defgroup Win RMA One-Sided Communication
	 * RMA Window Creation / Deletion / Get / Put / Accumulate
	 *
	 * \defgroup Mutex Mutex
	 * An implementation of Mutex Semantics between MPI processes. Based loosely off of Andreas Prell's mpi_mutex.c (https://gist.github.com/aprell/1486197) and R. Thakur, R. Ross, and R. Latham, "Implementing Byte-Range Locks Using MPI One-Sided Communication," in Proc. of the 12th European PVM/MPI Users' Group Meeting (Euro PVM/MPI 2005), Recent Advances in Parallel Virtual Machine and Message Passing Interface, Lecture Notes in Computer Science, LNCS 3666, Springer, September 2005, pp. 119-128.
	 *
	 * \defgroup Shared Shared Arrays
	 * A simple shared array implementation using Mutex locks and RMA one-sided communication
	 */

#if (MPI_VERSION == 3)
#define MEL_3
#endif

	typedef MPI_Aint   Aint;
	typedef MPI_Offset Offset;

#ifdef MEL_3
	typedef MPI_Count  Count;
#endif

	/// Macro to help with return error codes
#ifndef MEL_NO_CHECK_ERROR_CODES
#define MEL_THROW(v, message) { int ierr = (v); if ((ierr) != MPI_SUCCESS) MEL::Abort((ierr), std::string(message)); }
#else
#define MEL_THROW(v, message) { (v); }
#endif

    /**
	 * \ingroup  Errors
     * Calls MPI_Abort with the given error code and prints a string message to stderr
     *
     * \param[in] ierr		The error code to throw
     * \param[in] message	The message to print to stderr describing what happened
     */
    inline void Abort(int ierr, const std::string &message) {
        char error_string[BUFSIZ];
        int length_of_error_string, error_class, rank, size;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        fprintf(stderr, "\n\n*** MEL::ABORT ***\nRank %d / %d: %s\n", rank, size, message.c_str());

        MPI_Error_class(ierr, &error_class);
        MPI_Error_string(error_class, error_string, &length_of_error_string);
        fprintf(stderr, "Rank %d / %d: %s\n", rank, size, error_string);

        MPI_Error_string(ierr, error_string, &length_of_error_string);
        fprintf(stderr, "Rank %d / %d: %s\n", rank, size, error_string);
        MPI_Abort(MPI_COMM_WORLD, ierr);
    };

    /// Setup and teardown

    /**
	 * \ingroup Utils 
     * Tests if MPI_Init has been successfully called
     * 
     * \return Returns whether MPI is initialized as a bool
     */
    inline bool IsInitialized() {
        int init;
        MEL_THROW( MPI_Initialized(&init), "Initialized" );
        return init != 0;
    };

    /**
	 * \ingroup Utils 
	 * Tests if MPI_Finalize has been successfully called
     * 
     * \return Returns whether MPI is finalized as a bool
     */
    inline bool IsFinalized() {
        int fin; 
        MEL_THROW( MPI_Finalized(&fin), "Finalized" );
        return fin != 0;
    };

    /**
     * \ingroup Utils 
     * Call MPI_Init and setup default error handling
     *
     * \param[in] argc		Forwarded from program main
     * \param[in] argv		Forwarded from program main
     */
    inline void Init(int &argc, char **&argv) {
        if (!IsInitialized()) {
            MEL_THROW( MPI_Init(&argc, &argv), "Init" );
        }
        /// Allows MEL::Abort to be called properly
        MEL_THROW( MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "Initialize::SetErrorHandler" );
    };

    /**
     * \ingroup Utils 
     * Call MPI_Finalize
     */
    inline void Finalize() {
        if (!IsFinalized()) {
            MEL_THROW( MPI_Finalize(), "Finalize");
        }
    };

    /**
     * \ingroup  Errors
     * MEL alternative to std::exit
     *
     * \param[in] errcode	The error code to exit with
     */
    inline void Exit(const int errcode) {
        MEL::Abort(errcode, "EXIT");
    };

    /**
     * \ingroup  Errors
     * MEL alternative to std::exit
     *
     * \param[in] errcode	The error code to exit with
     * \param[in] msg		A message to print to stderr as the program exits
     */
    inline void Exit(const int errcode, const std::string &msg) {
        std::cerr << msg << std::endl;
        MEL::Abort(errcode, "EXIT");
    };

    /**
     * \ingroup Utils 
     * Gets the current wall time since epoch in seconds
     * 
     * \return Returns the current wall time as a double
     */
    inline double Wtime() {
        return MPI_Wtime();
    };

    /**
     * \ingroup Utils 
     * Gets the current system tick
     * 
     * \return Returns the current system tick as a double
     */
    inline double Wtick() {
        return MPI_Wtick();
    };

    struct ErrorHandler {
        static const ErrorHandler ERRHANDLER_NULL;
        
        MPI_Errhandler errHandl;

        explicit ErrorHandler(const MPI_Errhandler &_e) : errHandl(_e) {};
        inline ErrorHandler& operator=(const MPI_Errhandler &_e) {
            errHandl = _e;
            return *this;
        };

        explicit operator MPI_Errhandler() const {
            return errHandl;
        };
    };

#ifdef MEL_IMPLEMENTATION
    const ErrorHandler ErrorHandler::ERRHANDLER_NULL = ErrorHandler(MPI_ERRHANDLER_NULL);
#endif

    typedef void(*ErrorHandlerFunc)(MPI_Comm*, int*, ...);
    
    /**
     * \ingroup  Errors
     * A default error handler that can be attached to MPI objects to give basic error catching
     * 
     * \param[in] comm		Comm world in which the error occured
     * \param[in] ierr		The error code that was thrown
     */
    inline void DefaultErrorHandler(MPI_Comm *comm, int *ierr, ...) {
        char error_string[BUFSIZ];
        int length_of_error_string, error_class, rank, size;

        MPI_Comm_rank(*comm, &rank);
        MPI_Comm_size(*comm, &size);

        fprintf(stderr, "\n\n*** MEL::DefaultErrorHandler ***\nRank %d / %d\n", rank, size);

        MPI_Error_class(*ierr, &error_class);
        MPI_Error_string(error_class, error_string, &length_of_error_string);
        fprintf(stderr, "Rank %d / %d: %s\n", rank, size, error_string);

        MPI_Error_string(*ierr, error_string, &length_of_error_string);
        fprintf(stderr, "Rank %d / %d: %s\n", rank, size, error_string);
        MPI_Abort(*comm, *ierr);
    };

    /**
     * \ingroup  Errors
     * Add an error class for MPI to reference
     * 
     * \return Returns the new error class code that was added
     */
    inline int AddErrorClass() {
        int err;
        MEL_THROW( MPI_Add_error_class(&err), "ErrorHandler::AddErrorClass" );
        return err;
    };
    
    /**
     * \ingroup  Errors
     * Add an error code to an exisiting error class for MPI to reference
     * 
     * \param[in] errClass	The error class to add the error code to
     * \return			Returns the new error code that was added
     */
    inline int AddErrorCode(const int errClass) {
        int err;
        MEL_THROW( MPI_Add_error_code(errClass, &err), "ErrorHandler::AddErrorCode" );
        return err;
    };

    /**
     * \ingroup  Errors
     * Add an error code to a new error class for MPI to reference
     * 
     * \return			Returns the new error code that was added
     */
    inline int AddErrorCode() {
        return AddErrorCode(AddErrorClass());
    };

    /**
     * \ingroup  Errors
     * Add an error string to an existing error code for MPI to reference
     * 
     * \param[in] err		The error code to bind the string to
     * \param[in] str		The error string
     */
    inline void AddErrorString(const int err, const std::string &str) {
        MEL_THROW( MPI_Add_error_string(err, str.c_str()), "ErrorHandler::AddErrorString" );
    };
    
    /**
     * \ingroup  Errors
     * Add an error string to a new existing error code for MPI to reference
     * 
     * \param[in] str		The error string
     * \return			Returns the new error code added
     */
    inline int AddErrorString(const std::string &str) {
        const int err = AddErrorCode();
        AddErrorString(err, str);
        return err;
    };

    /**
     * \ingroup  Errors
     * Get the error class code of the given error code
     * 
     * \param[in] errCode	The error code
     * \return			Returns the error class
     */
    inline int GetErrorClass(const int errCode) {
        int err;
        MEL_THROW( MPI_Error_class(errCode, &err), "ErrorHandler::GetErrorClass" );
        return err;
    };
    
    /**
     * \ingroup  Errors
     * Get the error class code of the given error code
     * 
     * \param[in] errCode	The error code
     * \return			Returns the error class
     */
    inline std::string GetErrorString(const int errCode) {
        std::string str; str.resize(BUFSIZ); int len;
        MEL_THROW( MPI_Error_string(errCode, &str[0], &len), "ErrorHandler::GetErrorString" );
        str.resize(len);
        return str;
    };

    /**
     * \ingroup  Errors
     * Free an error handler that was previously added
     * 
     * \param[in] errHndl	The error handler object that references the bound function
     */
    inline void ErrorHandlerFree(ErrorHandler &errHndl) {
        MEL_THROW( MPI_Errhandler_free((MPI_Errhandler*) &errHndl), "ErrorHandler::Free" );
        //errHndl = MEL::ErrorHandler::ERRHANDLER_NULL;
    };

    /**
     * \ingroup  Errors
     * Free a vector Error Handlers
     *
     * \param[in] errHndls	A std::vector of Error Handlers
     */
    inline void ErrorHandlerFree(std::vector<ErrorHandler> &errHndls) {
        for (auto &d : errHndls) ErrorHandlerFree(d);
    };

    /**
     * \ingroup  Errors
     * Free the varadic set of error handlers provided
     * 
     * \param[in] d0		The first error handler to free
     * \param[in] d1		The second error handler to free
     * \param[in] args		The varadic set of remaining error handlers to free
     */
    template<typename T0, typename T1, typename ...Args>
    inline void ErrorHandlerFree(T0 &d0, T1 &d1, Args &&...args) {
        ErrorHandlerFree(d0);
        ErrorHandlerFree(d1, args...);
    };

    /**
     * \ingroup  Mem
	 * Allocate a block of memory for 'size' number of type T
     * 
     * \param[in] size		The number of elements of type T to allocate
     * \return			Returns the pointer to the allocated memory
     */
    template<typename T>
    inline T* MemAlloc(const Aint size) {
        T *ptr;
        MEL_THROW( MPI_Alloc_mem(size * sizeof(T), MPI_INFO_NULL, &ptr), "Mem::Alloc" );
        return ptr;
    };

    /**
     * \ingroup  Mem
	 * Allocate a block of memory for 'size' number of type T and assign a default value
     * 
     * \param[in] size		The number of elements of type T to allocate
     * \param[in] val		The value to set each element equal to
     * \return			Returns the pointer to the allocated memory
     */
    template<typename T>
    inline T* MemAlloc(const Aint size, const T &val) {
        T *ptr = MemAlloc<T>(size);
        for (Aint i = 0; i < size; ++i) ptr[i] = val;
        return ptr;
    };

    /**
     * \ingroup  Mem
	 * Allocate a single object of type T and construct it with the set of varadic arguments
     * 
     * \param[in] args		The set of varadic arguments to construct the object with
     * \return			Returns the pointer to the allocated memory
     */
    template<typename T, typename ...Args>
    inline T* MemConstruct(Args &&...args) {
        T *ptr = MemAlloc<T>(1);
        new (ptr) T(args...);
        return ptr;
    };

    /**
     * \ingroup  Mem
	 * Free a pointer allocated with MPI_Alloc or the MEL equivilant functions
     * 
     * \param[in] ptr		The pointer to free
     */
    template<typename T>
    inline void MemFree(T *&ptr) {
        if (ptr != nullptr) {
            MEL_THROW( MPI_Free_mem(ptr), "Mem::Free" );
            ptr = nullptr;
        }
    };

    /**
     * \ingroup  Mem
	 * Free the varadic set of pointers provided
     * 
     * \param[in] d0		The first pointer to free
     * \param[in] d1		The second pointer to free
     * \param[in] args		The varadic set of remaining pointers to free
     */
    template<typename T0, typename T1, typename ...Args>
    inline void MemFree(T0 &d0, T1 &d1, Args &&...args) {
        MemFree(d0);
        MemFree(d1, args...);
    };

    /**
     * \ingroup  Mem
	 * Call the destructor for each element of the given array and then free the memory
     * 
     * \param[in] ptr		The pointer to the memory to be destructed
     * \param[in] len		The length of the array
     */
    template<typename T>
    inline void MemDestruct(T *&ptr, const Aint len = 1) {
        if (ptr == nullptr) return;
        for (Aint i = 0; i < len; ++i) {
            (&ptr[i])->~T();
        }
        MemFree(ptr);
    };

    enum {
        PROC_NULL  = MPI_PROC_NULL,
        ANY_SOURCE = MPI_ANY_SOURCE,
        ANY_TAG    = MPI_ANY_TAG
    };

    struct Comm {
        static const Comm WORLD, SELF, COMM_NULL;

        MPI_Comm comm;

        Comm() : comm(MPI_COMM_NULL) {};
        explicit Comm(const MPI_Comm &_e) : comm(_e) {};
        inline Comm& operator=(const MPI_Comm &_e) {
            comm = _e;
            return *this;
        };
        explicit operator MPI_Comm() const {
            return comm;
        };
    };

    struct Group {
        static const Group GROUP_NULL;

        MPI_Group group;

        Group() : group(MPI_GROUP_NULL) {};
        explicit Group(const MPI_Group &_e) : group(_e) {};
        inline Group& operator=(const MPI_Group &_e) {
            group = _e;
            return *this;
        };
        explicit operator MPI_Group() const {
            return group;
        };
    };

    struct Request {
        static const Request REQUEST_NULL;

        MPI_Request request;

        Request() : request(MPI_REQUEST_NULL) {};
        explicit Request(const MPI_Request &_e) : request(_e) {};
        inline Request& operator=(const MPI_Request &_e) {
            request = _e;
            return *this;
        };
        explicit operator MPI_Request() const {
            return request;
        };
    };

#ifdef MEL_IMPLEMENTATION
    const Comm Comm::WORLD              = Comm(MPI_COMM_WORLD);
    const Comm Comm::SELF               = Comm(MPI_COMM_SELF);
    const Comm Comm::COMM_NULL          = Comm(MPI_COMM_NULL);

    const Group Group::GROUP_NULL       = Group(MPI_GROUP_NULL);

    const Request Request::REQUEST_NULL = Request(MPI_REQUEST_NULL);
#endif


    typedef MPI_Status  Status;
    typedef MPI_Info    Info;

    /**
     * \ingroup  Comm
     * Create a Comm error handler by directly passing the function to use
     * 
     * \param[in] func		The function to use as an error handler
     * \return			Returns an object that MPI can use to reference the error handler
     */
    inline ErrorHandler CommCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Comm_create_errhandler((MPI_Comm_errhandler_function*) func, &errHndl), "Comm::CreateErrorHandler" );
        return ErrorHandler(errHndl);
    };
    
    /**
     * \ingroup  Comm
     * Set a Comm error handler by passing the a error handler reference
     *
     * \param[in] comm		The comm world to attach the error handler to
     * \param[in] errHndl	The reference to a bound error handler
     */
    inline void CommSetErrorHandler(const Comm &comm, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_Comm_set_errhandler((MPI_Comm) comm, (MPI_Errhandler) errHndl), "Comm::SetErrorHandler" );
    };
    
    /**
     * \ingroup  Comm
     * Set a Comm error handler by directly passing the function to use
     *
     * \param[in] comm		The comm world to attach the error handler to
     * \param[in] func		The function to use as an error handler
     */
    inline void CommSetErrorHandler(const Comm &comm, ErrorHandlerFunc func) {
        CommSetErrorHandler(comm, CommCreateErrorHandler(func));
    };
    
    /**
     * \ingroup  Comm
     * Get the Comm error handler attached to a comm world
     *
     * \param[in] comm		The comm world to get the error handler of
     * \return			Returns a reference to a bound error handler
     */
    inline ErrorHandler CommGetErrorHandler(const Comm &comm) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Comm_get_errhandler((MPI_Comm) comm, &errHndl), "Comm::GetErrorHandler");
        return ErrorHandler(errHndl);
    };

    /**
     * \ingroup Comm 
     * Get the Comm rank of the process
     *
     * \param[in] comm		The comm world to get the rank in
     * \return			Returns the rank or the process within comm
     */
    inline int CommRank(const Comm &comm) {
        int r; 
        MEL_THROW( MPI_Comm_rank((MPI_Comm) comm, &r), "Comm::Rank" );
        return r;
    };
    
    /**
     * \ingroup Comm 
     * Get the Comm world size
     *
     * \param[in] comm		The comm world to get the size of
     * \return			Returns the size of the comm world
     */
    inline int CommSize(const Comm &comm) {
        int s; 
        MEL_THROW( MPI_Comm_size((MPI_Comm) comm, &s), "Comm::Size" );
        return s;
    };
    
    /**
     * \ingroup Comm 
     * Get the Comm world remote size
     *
     * \param[in] comm		The comm world to get the remote size of
     * \return			Returns the remote size of the comm world
     */
    inline int CommRemoteSize(const Comm &comm) {
        int s; 
        MEL_THROW( MPI_Comm_remote_size((MPI_Comm) comm, &s), "Comm::RemoteSize" );
        return s;
    };

    /**
     * \ingroup Comm 
     * Split a comm world into seperate comms. Processes with the same colour will end up in the same comm world
     *
     * \param[in] comm		The comm world to split
     * \param[in] colour	The group that this process will end up in in the new comm world
     * \return			Returns a new comm world
     */
    inline Comm CommSplit(const Comm &comm, int colour) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_split((MPI_Comm) comm, colour, CommRank(comm), &out_comm), "Comm::Split" );
        return Comm(out_comm);
    };
    
    /**
     * \ingroup Comm 
     * Duplicate a comm world so that it can be handled independently.
     *
     * \param[in] comm		The comm world to duplicate
     * \return			Returns a new comm world
     */
    inline Comm CommDuplicate(const Comm &comm) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_dup((MPI_Comm) comm, &out_comm), "Comm::Duplicate" );
        return Comm(out_comm);
    };
    
#ifdef MEL_3

    /**
     * \ingroup Comm 
     * Non-Blocking. Duplicate a comm world so that it can be handled independently.
     *
     * \param[in] comm		The comm world to duplicate
     * \param[out] rq		A request object that will signify when the comm world has been fully duplicated
     * \return			Returns a new comm world
     */
    inline Comm CommIduplicate(const Comm &comm, Request &rq) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_idup((MPI_Comm) comm, &out_comm, (MPI_Request*) &rq), "Comm::Iduplicate" );
        return Comm(out_comm);
    };

    /**
     * \ingroup Comm 
     * Non-Blocking. Duplicate a comm world so that it can be handled independently.
     *
     * \param[in] comm		The comm world to duplicate
     * \return			Returns a std::pair of the new comm world and a request object
     */
    inline std::pair<Comm, Request> CommIduplicate(const Comm &comm) {
        Request rq;
        Comm out_comm = CommIduplicate(comm, rq);
        return std::make_pair(out_comm, rq);
    };
#endif
    
    /**
     * \ingroup Comm 
     * Get the group of a comm world
     *
     * \param[in] comm		The comm world to get the group of
     * \return			Returns a Group object representing the processes in comm
     */
    inline Group CommGetGroup(const Comm &comm) {
        MPI_Group group;
        MEL_THROW( MPI_Comm_group((MPI_Comm) comm, &group), "Comm::GetGroup" );
        return Group(group);
    };
    
    /**
     * \ingroup Comm 
     * Create a comm object from an existing comm object and a group object
     *
     * \param[in] comm		The comm world to build off of
     * \param[in] group		The group to use to build the new comm object
     * \return			Returns a new comm object
     */
    inline Comm CommCreateFromGroup(const Comm &comm, const Group &group) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_create((MPI_Comm) comm, (MPI_Group) group, &out_comm), "Comm::CreateFromGroup" );
        return Comm(out_comm);
    };

#ifdef MEL_3
    
    /**
     * \ingroup Comm 
     * Create a comm object from an existing comm object and a group object. This is a non-collective version
     *
     * \param[in] comm		The comm world to build off of
     * \param[in] group		The group to use to build the new comm object
     * \param[in] tag		The tag to use
     * \return			Returns a new comm object
     */
    inline Comm CommCreateFromGroup(const Comm &comm, const Group &group, const int tag) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_create_group((MPI_Comm) comm, (MPI_Group) group, tag, &out_comm), "Comm::CreateFromGroup" );
        return Comm(out_comm);
    };
#endif

    /**
     * \ingroup Comm 
     * Free a comm world
     *
     * \param[in] comm		The comm world to free
     */
    inline void CommFree(Comm &comm) {
        MEL_THROW( MPI_Comm_disconnect((MPI_Comm*) &comm), "Comm::Free" );
        comm = Comm::COMM_NULL;
    };

    /**
     * \ingroup Comm 
     * Free a vector comm world
     *
     * \param[in] comms	A std::vector of comm world
     */
    inline void CommFree(std::vector<Comm> &comms) {
        for (auto &d : comms) CommFree(d);
    };

    /**
     * \ingroup Comm 
     * Free the varadic set of comm worlds provided
     * 
     * \param[in] d0		The first comm world to free
     * \param[in] d1		The second comm world to free
     * \param[in] args		The varadic set of remaining comm worlds to free
     */
    template<typename T0, typename T1, typename ...Args>
    inline void CommFree(T0 &d0, T1 &d1, Args &&...args) {
        CommFree(d0);
        CommFree(d1, args...);
    };

    /**
     * \ingroup Comm 
     * Test if a comm world is the null comm world
     *
     * \param[in] comm		The comm world to test
     * \return			Returns true if comm is the null comm world
     */
    inline bool CommIsNULL(const Comm &comm) {
        return (MPI_Comm) comm == MPI_COMM_NULL;
    };

    /**
     * \ingroup Sync 
     * Collective operation that forces all processes to wait until they are all at the barrier
     *
     * \param[in] comm		The comm world to synchronize
     */
    inline void Barrier(const Comm &comm) {
        MEL_THROW( MPI_Barrier((MPI_Comm) comm), "Comm::Barrier" );
    };

#ifdef MEL_3
    
    /**
     * \ingroup Sync 
     * Collective operation that forces all processes to wait until they are all at the barrier
     *
     * \param[in] comm		The comm world to synchronize
     * \param[out] rq		A reference to a request object used to determine when the barrier has been reached by all processes in comm
     */
    inline void Ibarrier(const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Ibarrier((MPI_Comm) comm, (MPI_Request*) &rq), "Comm::IBarrier" );
    };
    
    /**
     * \ingroup Sync 
     * Collective operation that forces all processes to wait until they are all at the barrier
     *
     * \param[in] comm		The comm world to synchronize
     * \return				Returns a request object used to determine when the barrier has been reached by all processes in comm
     */
    inline Request Ibarrier(const Comm &comm) {
        Request rq{};
        Ibarrier(comm, rq);
        return rq;
    };
#endif
    
    /**
     * \ingroup Sync 
     * Blocking operation to wait until a request object has completed
     *
     * \param[in] rq		The request object to wait for
     */
    inline void Wait(Request &rq) {
        MEL_THROW( MPI_Wait((MPI_Request*) &rq, MPI_STATUS_IGNORE), "Comm::Wait" );
    };
    
    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if a request object has completed
     *
     * \param[in] rq		The request object to test
     */
    inline bool Test(Request &rq) {
        int f;
        MEL_THROW( MPI_Test((MPI_Request*) &rq, &f, MPI_STATUS_IGNORE), "Comm::Test" );
        return f != 0;
    };

    /**
     * \ingroup Sync 
     * Blocking operation to wait until all request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     */
    inline void Waitall(Request *ptr, int num) {
        MEL_THROW( MPI_Waitall(num, (MPI_Request*) ptr, MPI_STATUS_IGNORE), "Comm::Waitall" );
    };
    
    /**
     * \ingroup Sync 
     * Blocking operation to wait until all request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     */
    inline void Waitall(std::vector<Request> &rqs) {
        Waitall(&rqs[0], rqs.size());
    };

    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if all request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     */
    inline bool Testall(Request *ptr, int num) {
        int f;
        MEL_THROW( MPI_Testall(num, (MPI_Request*) ptr, &f, MPI_STATUS_IGNORE), "Comm::Testall" );
        return f != 0;
    };

    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if all request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     */
    inline bool Testall(std::vector<Request> &rqs) {
        return Testall(&rqs[0], rqs.size());
    };

    /**
     * \ingroup Sync 
     * Blocking operation to wait until any of the request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     * \return			Returns the index of the completed request
     */
    inline int Waitany(Request *ptr, int num) {
        int idx;
        MEL_THROW( MPI_Waitany(num, (MPI_Request*) ptr, &idx, MPI_STATUS_IGNORE), "Comm::Waitany" );
        return idx;
    };
    
    /**
     * \ingroup Sync 
     * Blocking operation to wait until any of the request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     * \return			Returns the index of the completed request
     */
    inline int Waitany(std::vector<Request> &rqs) {
        return Waitany(&rqs[0], rqs.size());
    };

    /// Any test

    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if any of the request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     * \return			Returns a std::pair of a bool and int representing if any had completed, and if so what index
     */
    inline std::pair<bool, int> Testany(Request *ptr, int num) {
        int idx, f;
        MEL_THROW( MPI_Testany(num, (MPI_Request*) ptr, &idx, &f, MPI_STATUS_IGNORE), "Comm::Testany" );
        return std::make_pair(f != 0, idx);
    };
    
    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if any of the request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     * \return			Returns a std::pair of a bool and int representing if any had completed, and if so what index
     */
    inline std::pair<bool, int> Testany(std::vector<Request> &rqs) {
        return Testany(&rqs[0], rqs.size());
    };

    /**
     * \ingroup Sync 
     * Blocking operation to wait until some of the request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     * \return			Returns a std::vector of indices of the completed requests
     */
    inline std::vector<int> Waitsome(Request *ptr, int num) {
        std::vector<int> idx(num); int onum;
        MEL_THROW( MPI_Waitsome(num, (MPI_Request*) ptr, &onum, &idx[0], MPI_STATUS_IGNORE), "Comm::Waitsome" );
        idx.resize(onum);
        return idx;
    };
    
    /**
     * \ingroup Sync 
     * Blocking operation to wait until some of the request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     * \return			Returns a std::vector of indices of the completed requests
     */
    inline std::vector<int> Waitsome(std::vector<Request> &rqs) {
        return Waitsome(&rqs[0], rqs.size());
    };

    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if some of the request objects in an array have completed
     *
     * \param[in] ptr		Pointer to the array of request objects
     * \param[in] num		The length of the array
     * \return			Returns a std::vector of indices of the completed requests
     */
    inline std::vector<int> Testsome(Request *ptr, int num) {
        std::vector<int> idx(num); int onum;
        MEL_THROW( MPI_Testsome(num, (MPI_Request*) ptr, &onum, &idx[0], MPI_STATUS_IGNORE), "Comm::Testsome" );
        idx.resize(onum);
        return idx;
    };

    /**
     * \ingroup Sync 
     * Non-Blocking operation to test if some of the request objects in an array have completed
     *
     * \param[in] rqs		A std::vector of request objects to wait for
     * \return			Returns a std::vector of indices of the completed requests
     */
    inline std::vector<int> Testsome(std::vector<Request> &rqs) {
        return Testsome(&rqs[0], rqs.size());
    };

    /**
     * \ingroup Comm 
     * Perform a set union of two comm groups
     *
     * \param[in] lhs		The first operand of the union
     * \param[in] rhs		The second operand of the union
     * \return			Returns the union of the two groups
     */
    inline Group GroupUnion(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_union((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Union" );
        return Group(out_group);
    };
    
    /**
     * \ingroup Comm 
     * Perform a set difference of two comm groups
     *
     * \param[in] lhs		The first operand of the difference
     * \param[in] rhs		The second operand of the difference
     * \return			Returns the difference of the two groups
     */
    inline Group GroupDifference(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_difference((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Difference" );
        return Group(out_group);
    };

    /**
     * \ingroup Comm 
     * Perform a set intersection of two comm groups
     *
     * \param[in] lhs		The first operand of the intersection
     * \param[in] rhs		The second operand of the intersection
     * \return			Returns the intersection of the two groups
     */
    inline Group GroupIntersection(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_intersection((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Intersection" );
        return Group(out_group);
    };

    /**
     * \ingroup Comm 
     * Create a comm group including just the ranks from an exisitng group given in an array
     *
     * \param[in] group		The original group to build off of
     * \param[in] ranks		Pointer to the array of ranks
     * \param[in] num		The length of the array
     * \return			Returns the new groups
     */
    inline Group GroupInclude(const Group& group, const int *ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_incl((MPI_Group) group, num, ranks, &out_group), "Group::Include" );
        return Group(out_group);
    };
    
    /**
     * \ingroup Comm 
     * Create a comm group including just the ranks from an exisitng group given in an array
     *
     * \param[in] group		The original group to build off of
     * \param[in] ranks		A std::vector of ranks
     * \return			Returns the new groups
     */
    inline Group GroupInclude(const Group& group, const std::vector<int> &ranks) {
        return GroupInclude(group, &ranks[0], ranks.size());
    };

    /**
     * \ingroup Comm 
     * Create a comm group excluding just the ranks from an exisitng group given in an array
     *
     * \param[in] group		The original group to build off of
     * \param[in] ranks		Pointer to the array of ranks
     * \param[in] num		The length of the array
     * \return			Returns the new groups
     */
    inline Group GroupExclude(const Group& group, const int *ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_excl((MPI_Group) group, num, ranks, &out_group), "Group::Exclude" );
        return Group(out_group);
    };

    /**
     * \ingroup Comm 
     * Create a comm group excluding just the ranks from an exisitng group given in an array
     *
     * \param[in] group		The original group to build off of
     * \param[in] ranks		A std::vector of ranks
     * \return			Returns the new groups
     */
    inline Group GroupExclude(const Group& group, const std::vector<int> &ranks) {
        return GroupExclude(group, &ranks[0], ranks.size());
    };
        
    /**
     * \ingroup Comm 
     * Compare two comm groups
     *
     * \param[in] lhs		The first operand of the compare
     * \param[in] rhs		The second operand of the compare
     * \return			Returns the comparison of two groups
     */
    inline int GroupCompare(const Group& lhs, const Group& rhs) {
        int r; 
        MEL_THROW( MPI_Group_compare((MPI_Group) lhs, (MPI_Group) rhs, &r), "Group::Compare" );
        return r;
    };
    
    /**
     * \ingroup Comm 
     * Compare if two comm groups are similar
     *
     * \param[in] lhs		The first operand of the compare
     * \param[in] rhs		The second operand of the compare
     * \return			Returns true if the groups are similar
     */
    inline bool GroupIsSimilar(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_SIMILAR;
    };
    
    /**
     * \ingroup Comm 
     * Compare if two comm groups are identical
     *
     * \param[in] lhs		The first operand of the compare
     * \param[in] rhs		The second operand of the compare
     * \return			Returns true if the groups are identical
     */
    inline bool GroupIsIdentical(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_IDENT;
    };

    /**
     * \ingroup Comm 
     * Compare if two comm groups are congruent
     *
     * \param[in] lhs		The first operand of the compare
     * \param[in] rhs		The second operand of the compare
     * \return			Returns true if the groups are congruent
     */
    inline bool GroupIsCongruent(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_CONGRUENT;
    };

    /**
     * \ingroup Comm 
     * Compare if two comm groups are unequal
     *
     * \param[in] lhs		The first operand of the compare
     * \param[in] rhs		The second operand of the compare
     * \return			Returns true if the groups are unequal
     */
    inline bool GroupIsUnequal(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_UNEQUAL;
    };

    /**
     * \ingroup Comm 
     * Compare if a comm group is the null comm group
     *
     * \param[in] group		The group to test
     * \return			Returns true if the group is the null group
     */
    inline bool GroupIsNULL(const Group &group) {
        return (MPI_Group) group == MPI_GROUP_NULL;
    };

    /**
     * \ingroup Comm 
     * Gets the rank of the process within the given comm group
     *
     * \param[in] group		The group to use
     * \return			Returns the rank within the given group
     */
    inline int GroupRank(const Group &group) {
        int r; 
        MEL_THROW( MPI_Group_rank((MPI_Group) group, &r), "Group::Rank" );
        return r;
    };
    
    /**
     * \ingroup Comm 
     * Gets the size of the given comm group
     *
     * \param[in] group		The group to use
     * \return			Returns the size of the given group
     */
    inline int GroupSize(const Group &group) {
        int s; 
        MEL_THROW( MPI_Group_size((MPI_Group) group, &s), "Group::Size" );
        return s;
    };

    /**
     * \ingroup Comm 
     * Frees a comm group
     *
     * \param[in] group		The group to free
     */
    inline void GroupFree(Group &group) {
        MEL_THROW( MPI_Group_free((MPI_Group*) &group), "Group::Free" );
        //group = MPI_GROUP_NULL; // Done automatically
    };

    /**
     * \ingroup Comm 
     * Free a vector comm group
     *
     * \param[in] groups	A std::vector of comm group
     */
    inline void GroupFree(std::vector<Group> &groups) {
        for (auto &d : groups) GroupFree(d);
    };
 
    /**
     * \ingroup Comm 
     * Free the varadic set of comm groups provided
     * 
     * \param[in] d0		The first comm groups to free
     * \param[in] d1		The second comm groups to free
     * \param[in] args		The varadic set of remaining comm groups to free
     */
    template<typename T0, typename T1, typename ...Args>
    inline void GroupFree(T0 &d0, T1 &d1, Args &&...args) {
        GroupFree(d0);
        GroupFree(d1, args...);
    };


    struct Datatype {
        static const Datatype    DATATYPE_NULL,
                                
                                CHAR,
                                SIGNED_CHAR,
                                WCHAR,
                                UNSIGNED_CHAR,

                                INT,
                                SHORT,
                                LONG,
                                LONG_LONG,

                                UNSIGNED,
                                UNSIGNED_SHORT,
                                UNSIGNED_LONG,
                                UNSIGNED_LONG_LONG,

                                FLOAT,
                                DOUBLE,
                                LONG_DOUBLE,

                                INT8_T,
                                INT16_T,
                                INT32_T,
                                INT64_T,

                                UINT8_T,
                                UINT16_T,
                                UINT32_T,
                                UINT64_T,

#ifdef MEL_3
                                FLOAT_COMPLEX,
                                DOUBLE_COMPLEX,
                                LONG_DOUBLE_COMPLEX,
                                BOOL,
                                COUNT,
#endif

                                AINT,
                                OFFSET;

        MPI_Datatype datatype;

        Datatype() : datatype(MPI_DATATYPE_NULL) {};
        explicit Datatype(const MPI_Datatype &_e) : datatype(_e) {};
        inline Datatype& operator=(const MPI_Datatype &_e) {
            datatype = _e;
            return *this;
        };
        explicit operator MPI_Datatype() const {
            return datatype;
        };

        inline bool operator==(const Datatype &rhs) const {
            return datatype == rhs.datatype;
        };
        inline bool operator!=(const Datatype &rhs) const {
            return datatype != rhs.datatype;
        };
    };

#ifdef MEL_IMPLEMENTATION
    const Datatype Datatype::DATATYPE_NULL       = Datatype(MPI_DATATYPE_NULL);

    const Datatype Datatype::CHAR                = Datatype(MPI_CHAR);
    const Datatype Datatype::SIGNED_CHAR         = Datatype(MPI_SIGNED_CHAR);
    const Datatype Datatype::WCHAR               = Datatype(MPI_WCHAR);
    const Datatype Datatype::UNSIGNED_CHAR       = Datatype(MPI_UNSIGNED_CHAR);

    const Datatype Datatype::INT                 = Datatype(MPI_INT);
    const Datatype Datatype::SHORT               = Datatype(MPI_SHORT);
    const Datatype Datatype::LONG                = Datatype(MPI_LONG);
    const Datatype Datatype::LONG_LONG           = Datatype(MPI_LONG_LONG);

    const Datatype Datatype::UNSIGNED            = Datatype(MPI_UNSIGNED);
    const Datatype Datatype::UNSIGNED_SHORT      = Datatype(MPI_UNSIGNED_SHORT);
    const Datatype Datatype::UNSIGNED_LONG       = Datatype(MPI_UNSIGNED_LONG);
    const Datatype Datatype::UNSIGNED_LONG_LONG  = Datatype(MPI_UNSIGNED_LONG_LONG);


    const Datatype Datatype::FLOAT               = Datatype(MPI_FLOAT);
    const Datatype Datatype::DOUBLE              = Datatype(MPI_DOUBLE);
    const Datatype Datatype::LONG_DOUBLE         = Datatype(MPI_LONG_DOUBLE);

    const Datatype Datatype::INT8_T              = Datatype(MPI_INT8_T);
    const Datatype Datatype::INT16_T             = Datatype(MPI_INT16_T);
    const Datatype Datatype::INT32_T             = Datatype(MPI_INT32_T);
    const Datatype Datatype::INT64_T             = Datatype(MPI_INT64_T);


    const Datatype Datatype::UINT8_T             = Datatype(MPI_UINT8_T);
    const Datatype Datatype::UINT16_T            = Datatype(MPI_UINT16_T);
    const Datatype Datatype::UINT32_T            = Datatype(MPI_UINT32_T);
    const Datatype Datatype::UINT64_T            = Datatype(MPI_UINT64_T);

#ifdef MEL_3
    const Datatype Datatype::FLOAT_COMPLEX       = Datatype(MPI_CXX_FLOAT_COMPLEX);
    const Datatype Datatype::DOUBLE_COMPLEX      = Datatype(MPI_CXX_DOUBLE_COMPLEX);
    const Datatype Datatype::LONG_DOUBLE_COMPLEX = Datatype(MPI_CXX_LONG_DOUBLE_COMPLEX);
    const Datatype Datatype::BOOL                = Datatype(MPI_CXX_BOOL);
    const Datatype Datatype::COUNT               = Datatype(MPI_COUNT);
#endif

    const Datatype Datatype::AINT                = Datatype(MPI_AINT);
    const Datatype Datatype::OFFSET              = Datatype(MPI_OFFSET);
#endif

    /**
     * \ingroup Datatype 
     * Create a derived type representing a contiguous block of an elementary type
     *
     * \param[in] datatype	The base type to use
     * \param[in] length	The number of elements in the new type
     * \return			Returns a new type
     */
    inline Datatype TypeCreateContiguous(const Datatype &datatype, const int length) {
        Datatype dt;
        MEL_THROW( MPI_Type_contiguous(length, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeContiguous" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeContiguous)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a struct
     *
     * \param[in] num			The number of members within the struct
     * \param[in] datatypes		Pointer to an array of datatypes
     * \param[in] blockLengths	Pointer to an array of block lengths
     * \param[in] offsets		Pointer to an array of offsets
     * \return				Returns a new type
     */
    inline Datatype TypeCreateStruct(const int num, const Datatype *datatypes, const int *blockLengths, const Aint *offsets) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_struct(num, blockLengths, (MPI_Aint*) offsets, (MPI_Datatype*) datatypes, (MPI_Datatype*) &dt), "Datatype::TypeStruct" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeStruct)" );
        return dt;
    };

    struct TypeStruct_Block {
        Datatype datatype;
        int length;
        Aint offset;

        TypeStruct_Block() : datatype(), length(), offset() {};
        TypeStruct_Block(const Datatype &_dt, int _len, Aint _off) : datatype(_dt), length(_len), offset(_off) {};
        TypeStruct_Block(const Datatype &_dt, Aint _off) : datatype(_dt), length(1), offset(_off) {};
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a struct
     *
     * \param[in] blocks		A std::vector of triples representing the size of the current member block
     * \return				Returns a new type
     */
    inline Datatype TypeCreateStruct(const std::vector<TypeStruct_Block> &blocks) {
        const int num = blocks.size();
        std::vector<Datatype>    datatypes(num);
        std::vector<int>        blockLengths(num);
        std::vector<Aint>        offsets(num);

        for (int i = 0; i < num; ++i) {
            datatypes[i]        = blocks[i].datatype;
            blockLengths[i]     = blocks[i].length;
            offsets[i]          = blocks[i].offset;
        }
        return TypeCreateStruct(num, &datatypes[0], &blockLengths[0], &offsets[0]);
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] num			The number of dimensions of the data
     * \param[in] starts		Pointer to an array of start indices
     * \param[in] subSizes		Pointer to an array of sub sizes
     * \param[in] sizes			Pointer to an array of sizes of the parent array
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray(const Datatype &datatype, const int num, const int *starts, const int *subSizes, const int *sizes) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(num, sizes, subSizes, starts, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray)" );
        return dt;
    };

    struct TypeSubArray_Dim {
        int start, size, extent;

        TypeSubArray_Dim() :start(0), size(0), extent(0) {};
        TypeSubArray_Dim(const int _start, const int _size, const int _extent) :start(_start), size(_size), extent(_extent) {};
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] dims			A std::vector of triples representing the start, sub size, and parent size of each dimension of the data
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray(const Datatype &datatype, const std::vector<TypeSubArray_Dim> &dims) {
        const int numdims = dims.size();
        std::vector<int>    starts(numdims);
        std::vector<int>    subSizes(numdims);
        std::vector<int>    sizes(numdims);

        for (int i = 0; i < numdims; ++i) {
            starts[i]   = dims[i].start;
            subSizes[i] = dims[i].size;
            sizes[i]    = dims[i].extent;
        }
        return TypeCreateSubArray(datatype, numdims, &sizes[0], &subSizes[0], &starts[0]);
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a 1D sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] x				The start index in the x dimension
     * \param[in] sx			The sub size in the x dimension	
     * \param[in] dx			The parent size in the x dimension
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray1D(const Datatype &datatype, const int x, const int sx, const int dx) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(1, &dx, &sx, &x, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray1D" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray1D)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a 2D sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] x				The start index in the x dimension
     * \param[in] y				The start index in the y dimension
     * \param[in] sx			The sub size in the x dimension	
     * \param[in] sy			The sub size in the y dimension
     * \param[in] dx			The parent size in the x dimension
     * \param[in] dy			The parent size in the y dimension
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray2D(const Datatype &datatype,
                                         const int x,    const int y, 
                                         const int sx,   const int sy,
                                         const int dx,   const int dy) {
        int starts[2]   { y,     x  };
        int subSizes[2] { sy,    sx };
        int sizes[2]    { dy,    dx };
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(2, (int*) sizes, (int*) subSizes, (int*) starts, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray2D" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray2D)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a 3D sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] x				The start index in the x dimension
     * \param[in] y				The start index in the y dimension
     * \param[in] z				The start index in the z dimension
     * \param[in] sx			The sub size in the x dimension	
     * \param[in] sy			The sub size in the y dimension
     * \param[in] sz			The sub size in the z dimension
     * \param[in] dx			The parent size in the x dimension
     * \param[in] dy			The parent size in the y dimension
     * \param[in] dz			The parent size in the z dimension
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray3D(const Datatype &datatype,
                                         const int x,    const int y,    const int z, 
                                         const int sx, const int sy, const int sz,
                                         const int dx, const int dy, const int dz) {
        int starts[3]   {  z,     y,         x };
        int subSizes[3] { sz,    sy,        sx };
        int sizes[3]    { dz,    dy,        dx };
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(3, (int*) sizes, (int*) subSizes, (int*) starts, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray3D" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray3D)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a 4D sub array
     *
     * \param[in] datatype		The datatype of the parent array
     * \param[in] x				The start index in the x dimension
     * \param[in] y				The start index in the y dimension
     * \param[in] z				The start index in the z dimension
     * \param[in] w				The start index in the w dimension
     * \param[in] sx			The sub size in the x dimension	
     * \param[in] sy			The sub size in the y dimension
     * \param[in] sz			The sub size in the z dimension
     * \param[in] sw			The sub size in the w dimension
     * \param[in] dx			The parent size in the x dimension
     * \param[in] dy			The parent size in the y dimension
     * \param[in] dz			The parent size in the z dimension
     * \param[in] dw			The parent size in the w dimension
     * \return				Returns a new type
     */
    inline Datatype TypeCreateSubArray4D(const Datatype &datatype,
                                         const int x,  const int y,  const int z,  const int w,
                                         const int sx, const int sy, const int sz, const int sw,
                                         const int dx, const int dy, const int dz, const int dw) {

        int starts[4]   {  w,     z,         y,         x };
        int subSizes[4] { sw,    sz,        sy,        sx };
        int sizes[4]    { dw,    dz,        dy,        dx };
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(4, (int*) sizes, (int*) subSizes, (int*) starts, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray4D" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray4D)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks at different offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of blocks
     * \param[in] lengths		Pointer to an array of block lengths
     * \param[in] displs		Pointer to an array of block displacements
     * \return				Returns a new type
     */
    inline Datatype TypeCreateIndexed(const Datatype &datatype, const int num, const int *lengths, const int *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_indexed(num, lengths, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeIndexed" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeIndexed)" );
        return dt;
    };

    struct TypeIndexed_Block {
        int length, displ;

        TypeIndexed_Block() : length(), displ() {};
        TypeIndexed_Block(int _len, int _displ) : length(_len), displ(_displ) {};
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks at different offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] blocks		A std::vector of pairs representing the length and displacement of the blocks
     * \return				Returns a new type
     */
    inline Datatype TypeCreateIndexed(const Datatype &datatype, const std::vector<TypeIndexed_Block> &blocks) {
        const int num = blocks.size();
        std::vector<int>    lengths(num);
        std::vector<int>    displs(num);

        for (int i = 0; i < num; ++i) {
            lengths[i] = blocks[i].length;
            displs[i]  = blocks[i].displ;
        }
        return TypeCreateIndexed(datatype, num, &lengths[0], &displs[0]);
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks at different offsets, using byte offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of blocks
     * \param[in] lengths		Pointer to an array of block lengths
     * \param[in] displs		Pointer to an array of block displacements
     * \return				Returns a new type
     */
    inline Datatype TypeCreateHIndexed(const Datatype &datatype, const int num, const int *lengths, const Aint *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_hindexed(num, lengths, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeHIndexed" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeHIndexed)" );
        return dt;
    };

    struct TypeHIndexed_Block {
        int length;
        Aint displ;

        TypeHIndexed_Block() : length(), displ() {};
        TypeHIndexed_Block(int _len, Aint _displ) : length(_len), displ(_displ) {};
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks at different offsets, using byte offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] blocks		A std::vector of pairs representing the length and displacement of the blocks
     * \return				Returns a new type
     */
    inline Datatype TypeCreateHIndexed(const Datatype &datatype, const std::vector<TypeHIndexed_Block> &blocks) {
        const int num = blocks.size();
        std::vector<int>    lengths(num);
        std::vector<Aint>    displs(num);

        for (int i = 0; i < num; ++i) {
            lengths[i] = blocks[i].length;
            displs[i]  = blocks[i].displ;
        }
        return TypeCreateHIndexed(datatype, num, &lengths[0], &displs[0]);
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks of the same length at different offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of blocks
     * \param[in] length		The common block length
     * \param[in] displs		Pointer to an array of block displacements
     * \return				Returns a new type
     */
    inline Datatype TypeCreateIndexedBlock(const Datatype &datatype, const int num, const int length, const int *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_indexed_block(num, length, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeIndexedBlock" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeIndexedBlock)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks of the same length at different offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] length		The common block length
     * \param[in] displs		A std::vector representing the displacement of the blocks
     * \return				Returns a new type
     */
    inline Datatype TypeCreateIndexedBlock(const Datatype &datatype, const int length, const std::vector<int> &displs) {
        return TypeCreateIndexedBlock(datatype, displs.size(), length, &displs[0]);
    };

#ifdef MEL_3

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks of the same length at different offsets, using byte offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of blocks
     * \param[in] length		The common block length
     * \param[in] displs		Pointer to an array of block displacements
     * \return				Returns a new type
     */
    inline Datatype TypeCreateHIndexedBlock(const Datatype &datatype, const int num, const int length, const Aint *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_hindexed_block(num, length, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeHIndexedBlock" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeHIndexedBlock)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a set of contiguous blocks of the same length at different offsets, using byte offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] length		The common block length
     * \param[in] displs		A std::vector representing the displacement of the blocks
     * \return				Returns a new type
     */
    inline Datatype TypeCreateHIndexedBlock(const Datatype &datatype, const int length, const std::vector<Aint> &displs) {
        return TypeCreateHIndexedBlock(datatype, displs.size(), length, &displs[0]);
    };
#endif
    
    /**
     * \ingroup Datatype 
     * Create a derived type representing a strided sub array of a parent array
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of strided regions
     * \param[in] length		The common block length of the strided regions
     * \param[in] stride		The number of elements between each region
     * \return				Returns a new type
     */
    inline Datatype TypeCreateVector(const Datatype &datatype, const int num, const int length, const int stride) {
        Datatype dt;
        MEL_THROW( MPI_Type_vector(num, length, stride, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeVector" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeVector)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Create a derived type representing a strided sub array of a parent array, using byte offsets
     *
     * \param[in] datatype		The datatype of the elements
     * \param[in] num			The number of strided regions
     * \param[in] length		The common block length of the strided regions
     * \param[in] stride		The number of bytes between each region
     * \return				Returns a new type
     */
    inline Datatype TypeCreateHVector(const Datatype &datatype, const int num, const int length, const Aint stride) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_hvector(num, length, stride, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeHVector" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeHVector)" );
        return dt;
    };

    /**
     * \ingroup Datatype 
     * Duplicate a derived type so it can be managed independently
     *
     * \param[in] datatype		The datatype to duplicate
     * \return				Returns a new type
     */
    inline Datatype TypeDuplicate(const Datatype &datatype) {
        MPI_Datatype out_datatype;
        MEL_THROW( MPI_Type_dup((MPI_Datatype) datatype, &out_datatype), "Datatype::Duplicate" );
        return Datatype(out_datatype);
    };

    /**
     * \ingroup Datatype 
     * Compute the contiguous packed size of a datatype
     *
     * \param[in] datatype		The datatype to size
     * \return				Returns the contiguous size of the datatype in bytes
     */
    inline int TypeSize(const Datatype &datatype) {
        int out_size;
        MEL_THROW( MPI_Type_size((MPI_Datatype) datatype, &out_size), "Datatype::Size" );
        return out_size;
    };

    /**
     * \ingroup Datatype 
     * Compute the extent of a datatype
     *
     * \param[in] datatype		The datatype to get the extent of 
     * \return				Returns a std::pair of the datatype extent and lower bound
     */
    inline std::pair<Aint, Aint> TypeExtent(const Datatype &datatype) {
        Aint out_lb, out_ext;
        MEL_THROW( MPI_Type_get_extent((MPI_Datatype) datatype, &out_lb, &out_ext), "Datatype::Extent" );
        return std::make_pair(out_lb, out_ext);
    };

    /**
     * \ingroup Datatype 
     * Compute the extent of a datatype and discard the lower bound
     *
     * \param[in] datatype		The datatype to get the extent of 
     * \return				Returns the datatype extent
     */
    inline Aint TypeGetExtent(const Datatype &datatype) {
        Aint out_lb, out_ext;
        MEL_THROW( MPI_Type_get_extent((MPI_Datatype) datatype, &out_lb, &out_ext), "Datatype::GetExtent" );
        return out_ext;
    };

    /**
     * \ingroup Datatype 
     * Free a derived datatype
     *
     * \param[in] datatype		The datatype to free
     */
    inline void TypeFree(Datatype &datatype) {
        if (datatype != MEL::Datatype::DATATYPE_NULL) {
            MEL_THROW( MPI_Type_free((MPI_Datatype*) &datatype), "Datatype::Free" );
            datatype = Datatype::DATATYPE_NULL;
        }
    };

    /**
     * \ingroup Datatype 
     * Free a vector datatypes
     * 
     * \param[in] datatypes	A std::vector of derived datatypes
     */
    inline void TypeFree(std::vector<Datatype> &datatypes) {
        for (auto &d : datatypes) TypeFree(d);
    };
    
    /**
     * \ingroup Datatype 
     * Free the varadic set of datatypes provided
     * 
     * \param[in] d0		The first datatype to free
     * \param[in] d1		The second datatype to free
     * \param[in] args		The varadic set of remaining datatypes to free
     */
    template<typename T0, typename T1, typename ...Args>
    inline void TypeFree(T0 &d0, T1 &d1, Args &&...args) {
        TypeFree(d0);
        TypeFree(d1, args...);
    };

    /**
     * \ingroup Topo 
     * Compute the 'ideal' dimensions for a topolgy over n-processes
     * 
     * \param[in] numProcs	The number of processes in the topology
     * \param[in] numdims	The number of dimensions in the topology
     * \param[out] dims		Pointer to an (already allocated) array of length numdims
     */
    inline void TopoCartesianMakeDims(const int numProcs, const int numdims, int *dims) {
        MEL_THROW( MPI_Dims_create(numProcs, numdims, dims), "Topo::Cartesian::MakeDims" );
    };

    /**
     * \ingroup Topo 
     * Compute the 'ideal' dimensions for a topolgy over n-processes
     * 
     * \param[in] comm		The comm object the topology should represent
     * \param[in] numdims	The number of dimensions in the topology
     * \param[out] dims		Pointer to an (already allocated) array of length numdims
     */
    inline void TopoCartesianMakeDims(const Comm &comm, const int numdims, int *dims) {
        TopoCartesianMakeDims(CommSize(comm), numdims, dims);
    };

    /**
     * \ingroup Topo 
     * Compute the 'ideal' dimensions for a topolgy over n-processes
     * 
     * \param[in] numProcs	The number of processes in the topology
     * \param[in] numdims	The number of dimensions in the topology
     * \return			Returns a std::vector of dimension sizes
     */
    inline std::vector<int> TopoCartesianMakeDims(const int numProcs, const int numdims) {
        std::vector<int> dims(numdims);
        TopoCartesianMakeDims(numProcs, numdims, &dims[0]);
        return dims;
    };

    /**
     * \ingroup Topo 
     * Compute the 'ideal' dimensions for a topolgy over n-processes
     * 
     * \param[in] comm		The comm object the topology should represent
     * \param[in] numdims	The number of dimensions in the topology
     * \return			Returns a std::vector of dimension sizes
     */
    inline std::vector<int> TopoCartesianMakeDims(const Comm &comm, const int numdims) {
        return TopoCartesianMakeDims(CommSize(comm), numdims);
    };

    /**
     * \ingroup Topo 
     * Create a cartesian topology over a comm world
     * 
     * \param[in] comm		The comm object the topology should represent
     * \param[in] numdims	The number of dimensions in the topology
     * \param[in] dims		Pointer to an array of sizes of each dimension
     * \param[in] periods	Pointer to an array of logicals representing if each dimension is periodic or not
     * \return			Returns a Comm world with an attached topology
     */
    inline Comm TopoCartesianCreate(const Comm &comm, int numdims, const int *dims, const int *periods) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Cart_create((MPI_Comm) comm, numdims, dims, periods, 0, &out_comm), "Topo::Cartesian::Create");
        return Comm(out_comm);
    };

    struct TopoCartesian_Dim {
        int size, periodic;

        TopoCartesian_Dim() {};
        TopoCartesian_Dim(const int _size, const bool _p) : size(_size), periodic(_p ? 1 : 0) {};
        TopoCartesian_Dim(const int _size, const int _p) : size(_size), periodic(_p) {};
    };

	/**
	 * \ingroup Topo 
     * Create a cartesian topology over a comm world
	 *
	 * \param[in] comm		The comm object the topology should represent
	 * \param[in] dims		A std::vector of pairs representing dimension sizes and whether dimensions are periodic
	 * \return			Returns a Comm world with an attached topology
	 */
    inline Comm TopoCartesianCreate(const Comm &comm, const std::vector<TopoCartesian_Dim> &dims) {
        const int numdims = dims.size();
        std::vector<int>    sizes(numdims);
        std::vector<int>    periods(numdims);

        for (int i = 0; i < numdims; ++i) {
            sizes[i]    = dims[i].size;
            periods[i]  = dims[i].periodic;
        }
        return TopoCartesianCreate(comm, numdims, &sizes[0], &periods[0]);
    };

	/**
	 * \ingroup Topo 
     * Get the number of dimensions in an attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \return			Returns the number of dimensions in the attached cartesian topology
	 */
    inline int TopoCartesianNumDims(const Comm &comm) {
        int dim;
        MEL_THROW( MPI_Cartdim_get((MPI_Comm) comm, &dim), "Topo::Cartesian::NumDims");
        return dim;
    };

	/**
	 * \ingroup Topo 
     * Get the rank within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] coords	Pointer to an array representing the n-dim coordinates in the topology
	 * \return			Returns the rank in the attached cartesian topology
	 */
    inline int TopoCartesianRank(const Comm &comm, const int *coords) {
        int rank;
        MEL_THROW( MPI_Cart_rank((MPI_Comm) comm, coords, &rank), "Topo::Cartesian::Rank");
        return rank;
    };

	/**
	 * \ingroup Topo 
     * Get the rank within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] coords	A std::vector representing the n-dim coordinates in the topology
	 * \return			Returns the rank in the attached cartesian topology
	 */
    inline int TopoCartesianRank(const Comm &comm, const std::vector<int> coords) {
        return TopoCartesianRank(comm, &coords[0]);
    };

	/**
	 * \ingroup Topo 
     * Get the n-dim coordinates within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] rank		The rank within comm
	 * \param[in] numdims	The number of dimensions in the topology
	 * \param[out] coords	Pointer to an array of numdims ints representing the coordinates
	 */
    inline void TopoCartesianCoords(const Comm &comm, const int rank, int numdims, int *coords) {
        MEL_THROW( MPI_Cart_coords((MPI_Comm) comm, rank, numdims, coords), "Topo::Cartesian::Coords");
    };

	/**
	 * \ingroup Topo 
     * Get the n-dim coordinates within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] rank		The rank within comm
	 * \param[in] numdims	The number of dimensions in the topology
	 * \return			Returns a std::vector representing the n-dim coordinates
	 */
	inline std::vector<int> TopoCartesianCoords(const Comm &comm, const int rank, int numdims) {
		std::vector<int> coords(numdims);
		TopoCartesianCoords(comm, rank, numdims, &coords[0]);
        return coords;
    };

	/**
	 * \ingroup Topo 
     * Get the n-dim coordinates within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] rank		The rank within comm
	 * \return			Returns a std::vector representing the n-dim coordinates
	 */
    inline std::vector<int> TopoCartesianCoords(const Comm &comm, const int rank) {
        return TopoCartesianCoords(comm, rank, TopoCartesianNumDims(comm));
    };

	/**
	 * \ingroup Topo 
     * Get the n-dim coordinates within the attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \return			Returns a std::vector representing the n-dim coordinates
	 */
    inline std::vector<int> TopoCartesianCoords(const Comm &comm) {
        return TopoCartesianCoords(comm, CommRank(comm), TopoCartesianNumDims(comm));
    };

	/**
	 * \ingroup Topo 
     * Get the properties of an attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] numdims	The number of dimensions in the topology
	 * \param[out] dims		Pointer to an array of n-dims size representing the sizes of each dimension
	 * \param[out] periods	Pointer to an array of n-dims size representing whether each dimension is periodic
	 * \param[out] coords	Pointer to an array of n-dims size representing the coordinate in each dimension
	 */
	inline void TopoCartesianGet(const Comm &comm, int numdims, int *dims, int *periods, int *coords) {
		MEL_THROW(MPI_Cart_get((MPI_Comm) comm, numdims, dims, periods, coords), "Topo::Cartesian::Get");
    };

	/**
	 * \ingroup Topo 
     * Get the properties of an attached cartesian topology of a comm world
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \return			Returns a std::pair of a std::vector representing the process coordinates, and a std::vector of pairs representing the dimension sizes and periods
	 */
	inline std::pair<std::vector<int>, std::vector<TopoCartesian_Dim>> TopoCartesianGet(const Comm &comm) {
        const int numdims = TopoCartesianNumDims(comm);
        std::vector<int> coords(numdims), dims(numdims), periods(numdims);
        TopoCartesianGet(comm, numdims, &dims[0], &periods[0], &coords[0]);

        std::vector<TopoCartesian_Dim> r;
        for (int i = 0; i < numdims; ++i) {
            r[i].size        = dims[i];
            r[i].periodic    = periods[i];
        }
        return std::make_pair(coords, r);
    };

	/**
	 * \ingroup Topo 
     * Compute the ranks of a left and right shifted neighbor for a given dimension within a topology
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] direction	The dimension to shift in
	 * \param[in] disp		How much to shift by
	 * \param[out] rank_prev	The left neighbour
	 * \param[out] rank_next	The right neighbour
	 */
    inline void TopoCartesianShift(const Comm &comm, int direction, int disp, int &rank_prev, int &rank_next) {
        MEL_THROW( MPI_Cart_shift((MPI_Comm) comm, direction, disp, &rank_prev, &rank_next), "Topo::Cartesian::Shift");
    };

	/**
	 * \ingroup Topo 
     * Compute the ranks of a left and right shifted neighbor for a given dimension within a topology
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \param[in] direction	The dimension to shift in
	 * \param[in] disp		How much to shift by
	 * \return			Returns a std::pair representing the ranks of the left and right neighbour
	 */
    inline std::pair<int, int> TopoCartesianShift(const Comm &comm, int direction, int disp) {
        int rank_prev, rank_next;
        TopoCartesianShift(comm, direction, disp, rank_prev, rank_next);
        return std::make_pair(rank_prev, rank_next);
    };

    struct CartesianStencil2D5P {
        int y0, y1, x0, x1;

        CartesianStencil2D5P() : y0(MEL::PROC_NULL), y1(MEL::PROC_NULL), x0(MEL::PROC_NULL), x1(MEL::PROC_NULL) {};

        inline int operator[](const int i) {
            switch (i) {
            case 0: return y0;
            case 1: return x1;
            case 2: return y1;
            case 3: return x0;
            };
            return MEL::PROC_NULL;
        };
    };

	/**
	 * \ingroup Topo 
     * Create a 2D 5-point stencil of ranks representing the neighbouring processes
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \return			Returns a 2D 5-point stencil of comm ranks
	 */
    inline CartesianStencil2D5P TopoCartesianStencil2D5P(const Comm &comm) {
        CartesianStencil2D5P stencil;
        TopoCartesianShift(comm, 0, 1, stencil.x0, stencil.x1);
        TopoCartesianShift(comm, 1, 1, stencil.y0, stencil.y1);
        return stencil;
    };

    struct CartesianStencil2D9P {
        int y0, y1, x0, x1, x0y0, x0y1, x1y0, x1y1;

        CartesianStencil2D9P() : y0(MEL::PROC_NULL), y1(MEL::PROC_NULL), x0(MEL::PROC_NULL), x1(MEL::PROC_NULL),
                        x0y0(MEL::PROC_NULL), x0y1(MEL::PROC_NULL), x1y0(MEL::PROC_NULL), x1y1(MEL::PROC_NULL) {};

        inline int operator[](const int i) {
            switch (i) {
            case 0: return y0;
            case 1: return x1y0;
            case 2: return x1;
            case 3: return x1y1;
            case 4: return y1;
            case 5: return x0y1;
            case 6: return x0;
            case 7: return x0y0;
            };
            return MEL::PROC_NULL;
        };
    };

	/**
	 * \ingroup Topo 
     * Create a 2D 9-point stencil of ranks representing the neighbouring processes
	 *
	 * \param[in] comm		The comm object the topology is attached to
	 * \return			Returns a 2D 9-point stencil of comm ranks
	 */
    inline CartesianStencil2D9P TopoCartesianStencil2D9P(const Comm &comm) {
        CartesianStencil2D9P stencil;
        TopoCartesianShift(comm, 0, 1, stencil.x0, stencil.x1);
        TopoCartesianShift(comm, 1, 1, stencil.y0, stencil.y1);
            
        int dims[2], periods[2], coords[2];
        TopoCartesianGet(comm, 2, dims, periods, coords);

        if (!(stencil.x0 == MEL::PROC_NULL || stencil.y0 == MEL::PROC_NULL)) {
            int ncoords[2]{ coords[0] - 1, coords[1] - 1 };
            if (ncoords[0] < 0) ncoords[0] += dims[0];
            if (ncoords[1] < 0) ncoords[1] += dims[1];
            stencil.x0y0 = TopoCartesianRank(comm, ncoords);
        }

        if (!(stencil.x1 == MEL::PROC_NULL || stencil.y0 == MEL::PROC_NULL)) {
            int ncoords[2]{ coords[0] + 1, coords[1] - 1 };
            if (ncoords[0] >= dims[0]) ncoords[0] -= dims[0];
            if (ncoords[1] < 0) ncoords[1] += dims[1];
            stencil.x1y0 = TopoCartesianRank(comm, ncoords);
        }

        if (!(stencil.x0 == MEL::PROC_NULL || stencil.y1 == MEL::PROC_NULL)) {
            int ncoords[2]{ coords[0] - 1, coords[1] + 1 };
            if (ncoords[0] < 0) ncoords[0] += dims[0];
            if (ncoords[1] >= dims[1]) ncoords[1] -= dims[1];
            stencil.x0y1 = TopoCartesianRank(comm, ncoords);
        }

        if (!(stencil.x1 == MEL::PROC_NULL || stencil.y1 == MEL::PROC_NULL)) {
            int ncoords[2]{ coords[0] + 1, coords[1] + 1 };
            if (ncoords[0] >= dims[0]) ncoords[0] -= dims[0];
            if (ncoords[1] >= dims[1]) ncoords[1] -= dims[1];
            stencil.x1y1 = TopoCartesianRank(comm, ncoords);
        }

        return stencil;
    };

    struct Op {
        static const Op MAX,
                        MIN,
                        SUM,
                        PROD,
                        LAND,
                        BAND,
                        LOR,
                        BOR,
                        LXOR,
                        BXOR,
                        MINLOC,
                        MAXLOC,
                        REPLACE,
#ifdef MEL_3
                        NO_OP,
#endif                        
                        OP_NULL;

        MPI_Op op;

        Op() : op(MPI_OP_NULL) {};
        explicit Op(const MPI_Op &_e) : op(_e) {};
        inline Op& operator=(const MPI_Op &_e) {
            op = _e;
            return *this;
        };
        explicit operator MPI_Op() const {
            return op;
        };
    };

#ifdef MEL_IMPLEMENTATION
    const Op Op::MAX      = Op(MPI_MAX);
    const Op Op::MIN      = Op(MPI_MIN);
    const Op Op::SUM      = Op(MPI_SUM);
    const Op Op::PROD     = Op(MPI_PROD);
    const Op Op::LAND     = Op(MPI_LAND);
    const Op Op::BAND     = Op(MPI_BAND);
    const Op Op::LOR      = Op(MPI_LOR);
    const Op Op::BOR      = Op(MPI_BOR);
    const Op Op::LXOR     = Op(MPI_LXOR);
    const Op Op::BXOR     = Op(MPI_BXOR);
    const Op Op::MINLOC   = Op(MPI_MINLOC);
    const Op Op::MAXLOC   = Op(MPI_MAXLOC);
    const Op Op::REPLACE  = Op(MPI_REPLACE);
#ifdef MEL_3
    const Op Op::NO_OP    = Op(MPI_NO_OP);
#endif    
    const Op Op::OP_NULL  = Op(MPI_OP_NULL);
#endif

    namespace Functor {
		/**
		 * \ingroup  Ops
		 * Binary Max Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the max of the two inputs
		 */
        template<typename T>
        T MAX(T &a, T &b) {
            return (a > b) ? a : b;
        };
        
		/**
		 * \ingroup  Ops
		 * Binary Min Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the min of the two inputs
		 */
		template<typename T>
        T MIN(T &a, T &b) {
            return (a < b) ? a : b;
        };

		/**
		 * \ingroup  Ops
		 * Binary Sum Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the sum of the two inputs
		 */
        template<typename T>
        T SUM(T &a, T &b) {
            return (a + b);
        };

		/**
		 * \ingroup  Ops
		 * Binary Product Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the product of the two inputs
		 */
        template<typename T>
        T PROD(T &a, T &b) {
            return (a * b);
        };

		/**
		 * \ingroup  Ops
		 * Binary Logical And Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the logical and of the two inputs
		 */
        template<typename T>
        T LAND(T &a, T &b) {
            return (a && b);
        };

		/**
		 * \ingroup  Ops
		 * Binary bitwise and Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the bitwise and of the two inputs
		 */
        template<typename T>
        T BAND(T &a, T &b) {
            return (a & b);
        };

		/**
		 * \ingroup  Ops
		 * Binary logical or Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the logical or of the two inputs
		 */
        template<typename T>
        T LOR(T &a, T &b) {
            return (a || b);
        };

		/**
		 * \ingroup  Ops
		 * Binary bitwise or Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the bitwise or of the two inputs
		 */
        template<typename T>
        T BOR(T &a, T &b) {
            return (a | b);
        };

		/**
		 * \ingroup  Ops
		 * Binary logical exclusive or Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the logical exclusive or of the two inputs
		 */
        template<typename T>
        T LXOR(T &a, T &b) {
            return (!a != !b);
        };

		/**
		 * \ingroup  Ops
		 * Binary bitwise exclusive or Functor
		 *
		 * \param[in] a			The left argument
		 * \param[in] b			The right argument
		 * \return				Returns the bitwise exclusive or of the two inputs
		 */
        template<typename T>
        T BXOR(T &a, T &b) {
            return (a ^ b);
        };

		/**
		 * \ingroup  Ops
		 * Maps the given binary functor to the local array of a reduction / accumulate operation
		 *
		 * \param[in] in		The left hand array for the reduction
		 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
		 * \param[in] len		Pointer to a single int representing the number of elements to be processed
		 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
		 */
        template<typename T, T(*F)(T&, T&)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            for (int i = 0; i < *len; ++i) inout[i] = F(in[i], inout[i]);
        };

		/**
		 * \ingroup  Ops
		 * Maps the given binary functor to the local array of a reduction / accumulate operation
		 *
		 * \param[in] in		The left hand array for the reduction
		 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
		 * \param[in] len		Pointer to a single int representing the number of elements to be processed
		 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
		 */
        template<typename T, T(*F)(T&, T&, Datatype)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            Datatype dt((Datatype)*dptr);
            for (int i = 0; i < *len; ++i) inout[i] = F(in[i], inout[i], dt);
        };

		/**
		 * \ingroup  Ops
		 * Maps the given binary functor to the local array of a reduction / accumulate operation
		 *
		 * \param[in] in		The left hand array for the reduction
		 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
		 * \param[in] len		Pointer to a single int representing the number of elements to be processed
		 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
		 */
        template<typename T, void(*F)(T*, T*, int)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            F(in, inout, *len);
        };

		/**
		 * \ingroup  Ops
		 * Maps the given binary functor to the local array of a reduction / accumulate operation
		 *
		 * \param[in] in		The left hand array for the reduction
		 * \param[in] inout		The right hand array for the reduction. This array is modified to reflect the result of the functor on each element
		 * \param[in] len		Pointer to a single int representing the number of elements to be processed
		 * \param[in] dptr		Pointer to a single derived datatype representing the data to be processed
		 */
        template<typename T, void(*F)(T*, T*, int, Datatype)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            F(in, inout, *len, (Datatype) *dptr);
        };
    };

	/**
	 * \ingroup Ops 
     * Create a derived operation for reduction type functions
	 *
	 * \param[in] commute	Is the operation commutative?
	 * \return			Returns a handle to a new Op
	 */
    template<typename T, T(*F)(T&, T&)>
	inline Op OpCreate(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

	/**
	 * \ingroup Ops 
     * Create a derived operation for reduction type functions
	 *
	 * \param[in] commute	Is the operation commutative?
	 * \return			Returns a handle to a new Op
	 */
    template<typename T, T(*F)(T&, T&, Datatype)>
	inline Op OpCreate(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

    /**
	 * \ingroup Ops 
     * Create a derived operation for reduction type functions
	 *
	 * \param[in] commute	Is the operation commutative?
	 * \return			Returns a handle to a new Op
	 */
    template<typename T, void(*F)(T*, T*, int)>
	inline Op OpCreate(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

	/**
	 * \ingroup Ops 
     * Create a derived operation for reduction type functions
	 *
	 * \param[in] commute	Is the operation commutative?
	 * \return			Returns a handle to a new Op
	 */
    template<typename T, void(*F)(T*, T*, int, Datatype)>
	inline Op OpCreate(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

    /**
	 * \ingroup Ops 
     * Create a derived operation for reduction type functions
	 *
	 * \param[in] commute	Is the operation commutative?
	 * \return			Returns a handle to a new Op
	 */
    template<typename T, void(*F)(T*, T*, int*, MPI_Datatype*)>
	inline Op OpCreate(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) F, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

	/**
	 * \ingroup Ops 
     * Free a derived operation
	 *
	 * \param[in] op		The op to free
	 */
    inline void OpFree(Op &op) {
        MEL_THROW( MPI_Op_free((MPI_Op*) &op ), "Op::Free" );
    };

	/**
	 * \ingroup Ops 
     * Free a std::vector of derived operations
	 *
	 * \param[in] ops		A std::vector of ops to be freed
	 */
    inline void OpFree(std::vector<Op> &ops) {
        for (auto &e : ops) OpFree(e);
    };

	/**
	 * \ingroup Ops 
     * Free a varadic set of derived operations
	 *
	 * \param[in] d0		The first op to free
	 * \param[in] d1		The second op to free
	 * \param[in] args		The remaining ops to free
	 */
    template<typename T0, typename T1, typename ...Args>
    inline void OpFree(T0 &d0, T1 &d1, Args &&...args) {
        OpFree(d0);
        OpFree(d1, args...);
    };

#ifdef MEL_IMPLEMENTATION
	const void* IN_PLACE = MPI_IN_PLACE;
#endif

    typedef MPI_File File;

    enum class FileMode : int {
        CREATE          = MPI_MODE_CREATE,
        APPEND          = MPI_MODE_APPEND,
        DELETE_ON_CLOSE = MPI_MODE_DELETE_ON_CLOSE,
        EXCL            = MPI_MODE_EXCL,
        
        RDONLY          = MPI_MODE_RDONLY,
        RDWR            = MPI_MODE_RDWR,    
        WRONLY          = MPI_MODE_WRONLY,
        
        SEQUENTIAL      = MPI_MODE_SEQUENTIAL,
        UNIQUE_OPEN     = MPI_MODE_UNIQUE_OPEN
    };

	/**
	 * Logical OR of two file access modes
	 *
	 * \param[in] a		The first file mode
	 * \param[in] b		The second file mode
	 * \return		The combined file mode
	 */
    inline FileMode operator|(const FileMode &a, const FileMode &b) {
        return static_cast<FileMode>(static_cast<int>(a) | static_cast<int>(b));
    };

    enum class SeekMode : int {
        SET                = MPI_SEEK_SET,
        CUR                = MPI_SEEK_CUR,
        END                = MPI_SEEK_END
    };

	/**
	 * \ingroup  File
     * Create a file error handler
	 *
	 * \param[in] func	The function to use
	 * \return		Returns a handle to an error handler
	 */
    inline ErrorHandler FileCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_File_create_errhandler((MPI_File_errhandler_function*) func, &errHndl), "File::CreateErrorHandler" );
        return ErrorHandler(errHndl);
    };

	/**
	 * \ingroup  File
     * Set the error handler for a file
	 *
	 * \param[in] file		The file to attach to
	 * \param[in] errHndl	The handler to use
	 */
    inline void FileSetErrorHandler(const File &file, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_File_set_errhandler(file, (MPI_Errhandler) errHndl), "File::SetErrorHandler" );
    };
    
	/**
	 * \ingroup  File
     * Set the error handler for a file
	 *
	 * \param[in] file	The file to attach to
	 * \param[in] func	The function to use
	 */
	inline void FileSetErrorHandler(const File &file, ErrorHandlerFunc func) {
        FileSetErrorHandler(file, FileCreateErrorHandler(func));
    };
    
	/**
	 * \ingroup  File
     * Get the error handler for a file
	 *
	 * \param[in] file	The file to attach to
	 * \return		Returns a handle to the error handler
	 */
	inline ErrorHandler FileGetErrorHandler(const File &file) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_File_get_errhandler(file, &errHndl), "File::GetErrorHandler");
        return ErrorHandler(errHndl);
    };

	/**
	 * \ingroup File
	 * Get the mode a file was opened with
	 *
	 * \param[in] file	The file to attach to
	 * \return		Returns the file mode
	 */
    inline FileMode FileGetMode(const File &file) {
        int amode;
        MEL_THROW( MPI_File_get_amode(file, &amode), "File::GetMode");
        return FileMode(amode);
    };

	/**
	 * \ingroup File
	 * Is the file opened in an atomic mode?
	 *
	 * \param[in] file	The file to attach to
	 * \return		Returns true is the file is opened atomically
	 */
    inline bool FileIsAtomic(const File &file) {
        int flag;
        MEL_THROW( MPI_File_get_atomicity(file, &flag), "File::GetAtomicity");
        return flag != 0;
    };
    
	/**
	 * \ingroup File
	 * Set the atomicity of the file handle
	 *
	 * \param[in] file	The file to attach to
	 * \param[in] atom	Boolean value representing atomicity
	 */
	inline void FileSetAtomicity(const File &file, const bool atom) {
        MEL_THROW( MPI_File_set_atomicity(file, atom ? 1 : 0), "File::SetAtomicity");
    };

	/**
	 * \ingroup File
	 * Get the byte position of the file cursor relative to a given location
	 *
	 * \param[in] file		The file to attach to
	 * \param[in] offset	The relative offset to measure byte distance against
	 * \return			Returns the number of bytes from the given offset
	 */
    inline Offset FileGetByteOffset(const File &file, const Offset offset) {
        Offset byteOffset;
        MEL_THROW( MPI_File_get_byte_offset(file, offset, &byteOffset), "File::GetByteOffset" );
        return byteOffset;
    };

	/**
	 * \ingroup File
	 * Get the comm group a file was opened as a part of
	 *
	 * \param[in] file		The file to attach to
	 * \return			Returns comm group the file handle belongs to
	 */
    inline Group FileGetGroup(const File &file) {
        MPI_Group group;
        MEL_THROW( MPI_File_get_group(file, &group), "File::GetGroup");
        return Group(group);
    };

	/**
	 * \ingroup File
	 * Get the MPI_Info object attached to a file handle
	 *
	 * \param[in] file		The file to attach to
	 * \return			Returns the info object
	 */
    inline Info FileGetInfo(const File &file) {
        MPI_Info info;
        MEL_THROW( MPI_File_get_info(file, &info), "File::GetInfo");
        return info;
    };

	/**
	 * \ingroup File
	 * Set the MPI_Info object attached to a file handle
	 *
	 * \param[in] file		The file to attach to
	 * \param[in] info		The info object to attach
	 */
    inline void FileSetInfo(const File &file, const Info &info) {
        MEL_THROW( MPI_File_set_info(file, info), "File::SetInfo");
    };

	/**
	 * \ingroup File
	 * Get the position of the file cursor
	 *
	 * \param[in] file		The file to attach to
	 * \return			Returns the location of the file cursor in bytes
	 */
    inline Offset FileGetPosition(const File &file) {
        Offset offset;
        MEL_THROW( MPI_File_get_position(file, &offset), "File::GetPosition" );
        return offset;
    };
    
	/**
	 * \ingroup File
	 * Get the position of the shared file cursor
	 *
	 * \param[in] file		The file to attach to
	 * \return			Returns the location of the shared file cursor in bytes
	 */
	inline Offset FileGetPositionShared(const File &file) {
        Offset offset;
        MEL_THROW( MPI_File_get_position_shared(file, &offset), "File::GetPositionShared" );
        return offset;
    };

	/**
	 * \ingroup File
	 * Get the size of the file in bytes
	 *
	 * \param[in] file		The file to attach to
	 * \return			Returns the size of the file in bytes
	 */
    inline Offset FileGetSize(const File &file) {
        Offset size;
        MEL_THROW( MPI_File_get_size(file, &size), "File::GetSize" );
        return size;
    };
    
	/**
	 * \ingroup File
	 * Set the size of the file in bytes
	 *
	 * \param[in] file	The file to attach to
	 * \param[in] size	The size in bytes to set the file size to
	 */
	inline void FileSetSize(const File &file, const Offset size) {
        MEL_THROW( MPI_File_set_size(file, size), "File::SetSize" );
    };
    
	/**
	 * \ingroup File
	 * Get the extent of the derived type set to the file handle
	 *
	 * \param[in] file			The file to attach to
	 * \param[in] datatype		The derived datatype to measure the extent of
	 * \return					Returns the extent of the type
	 */
	inline Aint FileGetTypeExtent(const File &file, const Datatype &datatype) {
        Aint size;
        MEL_THROW( MPI_File_get_type_extent(file, (MPI_Datatype) datatype, &size), "File::GetTypeExtent" );
        return size;
    };

	/**
	 * \ingroup File
	 * Open a file and return a handle to it
	 *
	 * \param[in] comm			The comm world to open the file with
	 * \param[in] path			The path to the desired file
	 * \param[in] amode			The file mode to open the file with
	 * \return					Returns a handle to the file pointer
	 */
    inline File FileOpen(const Comm &comm, const std::string &path, const FileMode amode) {
        MPI_File file;
        MEL_THROW( MPI_File_open((MPI_Comm) comm, path.c_str(), (int) amode, MPI_INFO_NULL, &file), "File::Open");
        MEL_THROW( MPI_File_set_errhandler(file, MPI_ERRORS_RETURN), "File::Open(SetErrorHandler)" );
        return file;
    };

	/**
	 * \ingroup File
	 * Open a file on an individual process and return a handle to it
	 *
	 * \param[in] path			The path to the desired file
	 * \param[in] amode			The file mode to open the file with
	 * \return					Returns a handle to the file pointer
	 */
    inline File FileOpenIndividual(const std::string &path, const FileMode amode) {
        return FileOpen(MEL::Comm::SELF, path, amode);
    };

	/**
	 * \ingroup File
	 * Delete a file by its path
	 *
	 * \param[in] path			The path to the file to be deleted
	 */
    inline void FileDelete(const std::string &path) {
        MEL_THROW( MPI_File_delete(path.c_str(), MPI_INFO_NULL), "File::Delete");
    };

	/**
	 * \ingroup File
	 * Close the file attached to the given file handle
	 *
	 * \param[in] file			The file handle to be closed
	 */
    inline void FileClose(File &file) {
        MEL_THROW( MPI_File_close(&file), "File::Close");
    };

	/**
	 * \ingroup File
	 * Preallocate the opened file to the given size on the file system
	 *
	 * \param[in] file			The file to be preallocated
	 * \param[in] fileSize		The size of the file in bytes
	 */
    inline void FilePreallocate(const File &file, const Offset fileSize) {
        MEL_THROW( MPI_File_preallocate(file, fileSize), "File::Preallocate" );
    };

	/**
	 * \ingroup File
	 * Move the file cursor to a specific position
	 *
	 * \param[in] file			The file
	 * \param[in] offset		The position to move the file cursor to
	 * \param[in] seekMode		The mode to move the cursor by
	 */
    inline void FileSeek(const File &file, const Offset offset, const SeekMode seekMode = MEL::SeekMode::SET) {
        MEL_THROW( MPI_File_seek(file, offset, (int) seekMode), "File::Seek" );
    };

	/**
	 * \ingroup File
	 * Move the shared file cursor to a specific position. The same values must be provided by all processes
	 *
	 * \param[in] file			The shared file
	 * \param[in] offset		The position to move the file cursor to
	 * \param[in] seekMode		The mode to move the cursor by
	 */
    inline void FileSeekShared(const File &file, const Offset offset, const SeekMode seekMode = MEL::SeekMode::SET) {
        MEL_THROW( MPI_File_seek_shared(file, offset, (int) seekMode), "File::SeekShared" );
    };

	/**
	 * \ingroup File
	 * Force all queued and pending disk operations on a file to be completed
	 *
	 * \param[in] file			The file to be synchronized
	 */
    inline void FileSync(const File &file) {
        MEL_THROW( MPI_File_sync(file), "File::Sync");
    };

    struct FileView {
        Offset offset;
        Datatype elementaryType, fileType;
        std::string datarep;
        FileView() {};
        FileView(const Offset _offset, const Datatype _elementaryType, const Datatype _fileType, const std::string &_datarep = "native") 
                : offset(_offset), elementaryType(_elementaryType), fileType(_fileType), datarep(_datarep) {};
    };

	/**
	 * \ingroup File
	 * Set the view of a file handle for subsequent read/writes
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset from start of the file
	 * \param[in] elementaryType	The derived type representing the type of each element to be written
	 * \param[in] fileType			The derived type representing the structure of data to be written
	 * \param[in] datarep			String argument telling the MPI implementation how data is represented. Default is "native"
	 */
    inline void FileSetView(const File &file, const Offset offset, const Datatype elementaryType, const Datatype fileType, const std::string &datarep = "native") {
        MEL_THROW( MPI_File_set_view(file, offset, (MPI_Datatype) elementaryType, (MPI_Datatype) fileType, datarep.c_str(), MPI_INFO_NULL), "File::SetView" );    
    };

	/**
	 * \ingroup File
	 * Set the view of a file handle for subsequent read/writes
	 *
	 * \param[in] file				The file handle
	 * \param[in] view				A utility structure that stores the values of a file view
	 */
    inline void FileSetView(const File &file, const FileView &view) {
        FileSetView(file, view.offset, view.elementaryType, view.fileType, view.datarep);
    };

	/**
	 * \ingroup File
	 * Get the view attached to a file handle
	 *
	 * \param[in] file				The file handle
	 * \param[out] offset			Byte offset from start of the file
	 * \param[out] elementaryType	The derived type representing the type of each element to be written
	 * \param[out] fileType			The derived type representing the structure of data to be written
	 * \param[out] datarep			String argument telling the MPI implementation how data is represented. Default is "native"
	 */
    inline void FileGetView(const File &file, Offset &offset, Datatype &elementaryType, Datatype &fileType, std::string &datarep) {
        datarep.resize(BUFSIZ);
        MEL_THROW( MPI_File_get_view(file, &offset, (MPI_Datatype*) &elementaryType, (MPI_Datatype*) &fileType, (char*) &datarep[0]), "File::GetView" ); 
    };

	/**
	 * \ingroup File
	 * Get the view attached to a file handle
	 *
	 * \param[in] file				The file handle
	 * \return						Returns a utility structure that stores the values of a file view
	 */
    inline FileView FileGetView(const File &file) {
        FileView view;
        FileGetView(file, view.offset, view.elementaryType, view.fileType, view.datarep);
        return view;
    };

	/**
	 * \ingroup File
	 * Write to file from a single process
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
    inline Status FileWrite(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::Write" );
        return status;
    };
    
	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
	inline Status FileWriteAll(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_all(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAll" );
        return status;
    };

	/**
	 * \ingroup File
	 * Write to file from a single process at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write at
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
    inline Status FileWriteAt(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_at(file, offset, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAt" );
        return status;
    };
    
	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write at
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
	inline Status FileWriteAtAll(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_at_all(file, offset, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAtAll" );
        return status;
    };

	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file in sequence
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
    inline Status FileWriteOrdered(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_ordered(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteOrdered" );
        return status;
    };

	/**
	 * \ingroup File
	 * Write to file from any processes that opened the file in parallel
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A status object
	 */
    inline Status FileWriteShared(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_shared(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteShared" );
        return status;
    };

	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from a single processes 
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A request object
	 */
    inline Request FileIwrite(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iwrite(file, sptr, snum, (MPI_Datatype) datatype, &request), "File::Iwrite" );
        return Request(request);
    };
    
	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from a single process at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write to
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A request object
	 */
	inline Request FileIwriteAt(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW(MPI_File_iwrite_at(file, offset, sptr, snum, (MPI_Datatype) datatype, &request), "File::IwriteAt");
        return Request(request);
    };

	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from any processes that opened the file in parallel
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \param[in] datatype			The derived type representing the elements to be written
	 * \return						A request object
	 */
    inline Request FileIwriteShared(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iwrite_shared(file, sptr, snum, (MPI_Datatype) datatype, &request), "File::IwriteShared" );
        return Request(request);
    };

	/**
	 * \ingroup File
	 * Read from file from a single process
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
    inline Status FileRead(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::Read" );
        return status;
    };

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
    inline Status FileReadAll(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_all(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAll" );
        return status;
    };

	/**
	 * \ingroup File
	 * Read from file from a single process at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
    inline Status FileReadAt(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_at(file, offset, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAt" );
        return status;
    };

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
    inline Status FileReadAtAll(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_at_all(file, offset, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAtAll" );
        return status;
    };

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file in sequence
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
    inline Status FileReadOrdered(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_ordered(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadOrdered" );
        return status;
    };
    
	/**
	 * \ingroup File
	 * Read from file from any processes that opened the file in parallel
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A status object
	 */
	inline Status FileReadShared(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_shared(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadShared" );
        return status;
    };

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from a single process
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A request object
	 */
    inline Request FileIread(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iread(file, rptr, rnum, (MPI_Datatype) datatype, &request), "File::Iread" );
        return Request(request);
    };

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from a single process at the desired offset
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A request object
	 */
    inline Request FileIreadAt(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW(MPI_File_iread_at(file, offset, rptr, rnum, (MPI_Datatype) datatype, &request), "File::IreadAt");
        return Request(request);
    };

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from any process that opened the file in parallel
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \param[in] datatype			The derived type representing the elements to be read
	 * \return						A request object
	 */
    inline Request FileIreadShared(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iread_shared(file, rptr, rnum, (MPI_Datatype) datatype, &request), "File::IreadShared" );
        return Request(request);
    };    

	/// \cond HIDE
#define MEL_FILE(T, D) inline Status FileWrite(const File &file, const T *sptr, const int snum) {                                    \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write(file, sptr, snum,  D, &status), "File::Write(#T, #D)" );                                            \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileWriteAll(const File &file, const T *sptr, const int snum) {                                                    \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write_all(file, sptr, snum,  D, &status), "File::WriteAll(#T, #D)" );                                    \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileWriteAt(const File &file, const Offset offset, const T *sptr, const int snum) {                                \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write_at(file, offset, sptr, snum,  D, &status), "File::WriteAt(#T, #D)" );                                \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileWriteAtAll(const File &file, const Offset offset, const T *sptr, const int snum) {                            \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write_at_all(file, offset, sptr, snum,  D, &status), "File::WriteAtAll(#T, #D)" );                        \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileWriteOrdered(const File &file, const T *sptr, const int snum) {                                                \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write_ordered(file, sptr, snum,  D, &status), "File::WriteOrdered(#T, #D)" );                            \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileWriteShared(const File &file, const T *sptr, const int snum) {                                                \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_write_shared(file, sptr, snum,  D, &status), "File::WriteShared(#T, #D)" );                                \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Request FileIwrite(const File &file, const T *sptr, const int snum) {                                                    \
        MPI_Request request;                                                                                                        \
        MEL_THROW( MPI_File_iwrite(file, sptr, snum,  D, &request), "File::Iwrite(#T, #D)" );                                        \
        return Request(request);                                                                                                    \
    };                                                                                                                                \
    inline Request FileIwriteAt(const File &file, const Offset offset, const T *sptr, const int snum) {                                \
        MPI_Request request;                                                                                                        \
        MEL_THROW(MPI_File_iwrite_at(file, offset, sptr, snum,  D, &request), "File::IwriteAt");                                    \
        return Request(request);                                                                                                    \
    };                                                                                                                                \
    inline Request FileIwriteShared(const File &file, const T *sptr, const int snum) {                                                \
        MPI_Request request;                                                                                                        \
        MEL_THROW( MPI_File_iwrite_shared(file, sptr, snum,  D, &request), "File::IwriteShared(#T, #D)" );                            \
        return Request(request);                                                                                                    \
    };                                                                                                                                \
    inline Status FileRead(const File &file, T *rptr, const int rnum) {                                                                \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read(file, rptr, rnum,  D, &status), "File::Read(#T, #D)" );                                            \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileReadAll(const File &file, T *rptr, const int rnum) {                                                            \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read_all(file, rptr, rnum,  D, &status), "File::ReadAll(#T, #D)" );                                        \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileReadAt(const File &file, const Offset offset, T *rptr, const int rnum) {                                        \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read_at(file, offset, rptr, rnum,  D, &status), "File::ReadAt(#T, #D)" );                                \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileReadAtAll(const File &file, const Offset offset, T *rptr, const int rnum) {                                    \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read_at_all(file, offset, rptr, rnum,  D, &status), "File::ReadAtAll(#T, #D)" );                        \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileReadOrdered(const File &file, T *rptr, const int rnum) {                                                        \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read_ordered(file, rptr, rnum,  D, &status), "File::ReadOrdered(#T, #D)" );                                \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Status FileReadShared(const File &file, T *rptr, const int rnum) {                                                        \
        MPI_Status status;                                                                                                            \
        MEL_THROW( MPI_File_read_shared(file, rptr, rnum,  D, &status), "File::ReadShared(#T, #D)" );                                \
        return status;                                                                                                                \
    };                                                                                                                                \
    inline Request FileIread(const File &file, T *rptr, const int rnum) {                                                            \
        MPI_Request request;                                                                                                        \
        MEL_THROW( MPI_File_iread(file, rptr, rnum,  D, &request), "File::Iread(#T, #D)" );                                            \
        return Request(request);                                                                                                    \
    };                                                                                                                                \
    inline Request FileIreadAt(const File &file, const Offset offset, T *rptr, const int rnum) {                                    \
        MPI_Request request;                                                                                                        \
        MEL_THROW(MPI_File_iread_at(file, offset, rptr, rnum,  D, &request), "File::IreadAt");                                        \
        return Request(request);                                                                                                    \
    };                                                                                                                                \
    inline Request FileIreadShared(const File &file, T *rptr, const int rnum) {                                                        \
        MPI_Request request;                                                                                                        \
        MEL_THROW( MPI_File_iread_shared(file, rptr, rnum,  D, &request), "File::IreadShared(#T, #D)" );                            \
        return Request(request);                                                                                                    \
    };                                                                                                                                
    
	MEL_FILE(wchar_t, MPI_WCHAR);

	MEL_FILE(float, MPI_FLOAT);
	MEL_FILE(double, MPI_DOUBLE);
	MEL_FILE(long double, MPI_LONG_DOUBLE);

	MEL_FILE(int8_t, MPI_INT8_T);
	MEL_FILE(int16_t, MPI_INT16_T);
	MEL_FILE(int32_t, MPI_INT32_T);
	MEL_FILE(int64_t, MPI_INT64_T);

	MEL_FILE(uint8_t, MPI_UINT8_T);
	MEL_FILE(uint16_t, MPI_UINT16_T);
	MEL_FILE(uint32_t, MPI_UINT32_T);
	MEL_FILE(uint64_t, MPI_UINT64_T);

#ifdef MEL_3
	MEL_FILE(std::complex<float>, MPI_CXX_FLOAT_COMPLEX);
	MEL_FILE(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);
	MEL_FILE(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);
	MEL_FILE(bool, MPI_CXX_BOOL);
#endif

#undef MEL_FILE
	/// \endcond

	/**
	 * \ingroup File
	 * Write to file from a single process. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
    template<typename T>
    inline Status FileWrite(const File &file, const T *sptr, const int snum) {
        return FileWrite(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
    };

	/**
	 * \ingroup File
	 * Write to file from a single process at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write to
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileWriteAt(const File &file, const Offset offset, const T *sptr, const int snum) {
		return FileWriteAt(file, offset, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileWriteAll(const File &file, const T *sptr, const int snum) {
		return FileWriteAll(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write to
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileWriteAtAll(const File &file, const Offset offset, const T *sptr, const int snum) {
		return FileWriteAtAll(file, offset, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Write to file from all processes that opened the file in sequence. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileWriteOrdered(const File &file, const T *sptr, const int snum) {
		return FileWriteOrdered(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Write to file from any processes that opened the file in parallel. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileWriteShared(const File &file, const T *sptr, const int snum) {
		return FileWriteShared(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Read from file from a single process. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
    template<typename T>
    inline Status FileRead(const File &file, T *rptr, const int rnum) {
        return FileRead(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
    };

	/**
	 * \ingroup File
	 * Read from file from a single process at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileReadAt(const File &file, const Offset offset, T *rptr, const int rnum) {
		return FileReadAt(file, offset, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileReadAll(const File &file, T *rptr, const int rnum) {
		return FileReadAll(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileReadAtAll(const File &file, const Offset offset, T *rptr, const int rnum) {
		return FileReadAtAll(file, offset, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Read from file from all processes that opened the file in sequence. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileReadOrdered(const File &file, T *rptr, const int rnum) {
		return FileReadOrdered(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Read from file from any processes that opened the file in parallel. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A status object
	 */
	template<typename T>
	inline Status FileReadShared(const File &file, T *rptr, const int rnum) {
		return FileReadShared(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from a single process. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIwrite(const File &file, const T *sptr, const int snum) {
		return FileIwrite(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from all processes that opened the file at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to write to
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIwriteAt(const File &file, const Offset offset, const T *sptr, const int snum) {
		return FileIwriteAt(file, offset, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Write to file from any processes that opened the file in parallel. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] sptr				Pointer to the memory to be written
	 * \param[in] snum				The number of elements to write
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIwriteShared(const File &file, const T *sptr, const int snum) {
		return FileIwriteShared(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from a single process. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIread(const File &file, T *rptr, const int rnum) {
		return FileIread(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from a single process at the desired offset. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[in] offset			Byte offset into the file to read from
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIreadAt(const File &file, const Offset offset, T *rptr, const int rnum) {
		return FileIreadAt(file, offset, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup File
	 * Non-Blocking. Read from file from any processes that opened the file in parallel. Element size determined by template type
	 *
	 * \param[in] file				The file handle
	 * \param[out] rptr				Pointer to the memory to be read into
	 * \param[in] rnum				The number of elements to read
	 * \return						A request object
	 */
	template<typename T>
	inline Request FileIreadShared(const File &file, T *rptr, const int rnum) {
		return FileIreadShared(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
	};

	/**
	 * \ingroup P2P
	 * Send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
    inline void Send(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                
        MEL_THROW( MPI_Send(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Send" );                                            
    };                                                                                                                                
    
	/**
	 * \ingroup P2P
	 * Buffered send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
	inline void Bsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Bsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Bsend" );                                        
    };
	
	/**
	 * \ingroup P2P
	 * Synchronous send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */                                                                                                                                
    inline void Ssend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Ssend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Ssend" );                                        
    };
	
	/**
	 * \ingroup P2P
	 * Ready send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */                                                                                                                                  
    inline void Rsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Rsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Rsend" );                                        
    };  
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */                                                                                                                               
    inline void Isend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Isend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Isend" );                                    
    };                                                                                                                               
    
	/**
	 * \ingroup P2P
	 * Non-Blocking. Send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */
	inline Request Isend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Isend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    };
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Buffered send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */                                                                                                                                  
    inline void Ibsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Ibsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibsend" );                                
    };  
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Buffered send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */                                                                                                                              
    inline Request Ibsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Ibsend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }; 
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Synchronous send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */                                                                                                                                  
    inline void Issend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Issend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Issend" );                                
    }; 
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Synchronous send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */                                                                                                                                 
    inline Request Issend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Issend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }; 
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Ready send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */                                                                                                                               
    inline void Irsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Irsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irsend" );                                
    };
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Ready send num elements of a derived type from the given address
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */                                                                                                                                
    inline Request Irsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Irsend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    };

	/// \cond HIDE
#define MEL_SEND(T, D)    inline void Send(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                \
        MEL_THROW( MPI_Send(ptr, num, D, dst, tag, (MPI_Comm) comm), "Comm::Send( " #T ", " #D " )" );                                \
    }                                                                                                                                \
    inline void Bsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                                \
        MEL_THROW( MPI_Bsend(ptr, num, D, dst, tag, (MPI_Comm) comm), "Comm::Bsend( " #T ", " #D " )" );                            \
    }                                                                                                                                \
    inline void Ssend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                                \
        MEL_THROW( MPI_Ssend(ptr, num, D, dst, tag, (MPI_Comm) comm), "Comm::Ssend( " #T ", " #D " )" );                            \
    }                                                                                                                                \
    inline void Rsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                                \
        MEL_THROW( MPI_Rsend(ptr, num, D, dst, tag, (MPI_Comm) comm), "Comm::Rsend( " #T ", " #D " )" );                            \
    }                                                                                                                                \
    inline void Isend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {                    \
        MEL_THROW( MPI_Isend(ptr, num, D, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Isend( " #T ", " #D " )" );        \
    }                                                                                                                                \
    inline Request Isend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                                \
        Request rq{};                                                                                                                \
        Isend(ptr, num, dst, tag, comm, rq);                                                                                        \
        return rq;                                                                                                                    \
    }                                                                                                                                \
    inline void Ibsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {                    \
        MEL_THROW( MPI_Ibsend(ptr, num, D, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibsend( " #T ", " #D " )" );        \
    }                                                                                                                                \
    inline Request Ibsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                            \
        Request rq{};                                                                                                                \
        Ibsend(ptr, num, dst, tag, comm, rq);                                                                                        \
        return rq;                                                                                                                    \
    }                                                                                                                                \
    inline void Issend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {                    \
        MEL_THROW( MPI_Issend(ptr, num, D, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Issend( " #T ", " #D " )" );        \
    }                                                                                                                                \
    inline Request Issend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                            \
        Request rq{};                                                                                                                \
        Issend(ptr, num, dst, tag, comm, rq);                                                                                        \
        return rq;                                                                                                                    \
    }                                                                                                                                \
    inline void Irsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {                    \
        MEL_THROW( MPI_Irsend(ptr, num, D, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irsend( " #T ", " #D " )" );        \
    }                                                                                                                                \
    inline Request Irsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {                            \
        Request rq{};                                                                                                                \
        Irsend(ptr, num, dst, tag, comm, rq);                                                                                        \
        return rq;                                                                                                                    \
    }                                                                                                                                

    MEL_SEND(char,                            MPI_CHAR);
    MEL_SEND(wchar_t,                        MPI_WCHAR);

    MEL_SEND(float,                            MPI_FLOAT);
    MEL_SEND(double,                        MPI_DOUBLE);
    MEL_SEND(long double,                    MPI_LONG_DOUBLE);

    MEL_SEND(int8_t,                        MPI_INT8_T);
    MEL_SEND(int16_t,                        MPI_INT16_T);
    MEL_SEND(int32_t,                        MPI_INT32_T);
    MEL_SEND(int64_t,                        MPI_INT64_T);

    MEL_SEND(uint8_t,                        MPI_UINT8_T);
    MEL_SEND(uint16_t,                        MPI_UINT16_T);
    MEL_SEND(uint32_t,                        MPI_UINT32_T);
    MEL_SEND(uint64_t,                        MPI_UINT64_T);

#ifdef MEL_3
    MEL_SEND(std::complex<float>,            MPI_CXX_FLOAT_COMPLEX);
    MEL_SEND(std::complex<double>,            MPI_CXX_DOUBLE_COMPLEX);
    MEL_SEND(std::complex<long double>,        MPI_CXX_LONG_DOUBLE_COMPLEX);
    MEL_SEND(bool,                            MPI_CXX_BOOL);
#endif

#undef MEL_SEND
	/// \endcond

	/**
	 * \ingroup P2P
	 * Send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
    template<typename T>
    inline void Send(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        Send(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Buffered send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
    template<typename T>
    inline void Bsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        Bsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Synchronous send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
    template<typename T>
    inline void Ssend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        Ssend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Ready send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 */
    template<typename T>
    inline void Rsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        Rsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */
    template<typename T>
    inline void Isend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {
        Isend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm, rq);
    };
    
	/**
	 * \ingroup P2P
	 * Non-Blocking. Send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */
	template<typename T>
    inline Request Isend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        return Isend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Buffered send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */
    template<typename T>
    inline void Ibsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {
        Ibsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm, rq);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Buffered end num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */
    template<typename T>
    inline Request Ibsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        return Ibsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Synchronous send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */
    template<typename T>
    inline void Issend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {
        Issend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm, rq);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Synchronous end num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */
    template<typename T>
    inline Request Issend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        return Issend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Ready send num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \param[out] rq				A request object
	 */
    template<typename T>
    inline void Irsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm, Request &rq) {
        Irsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm, rq);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Ready end num elements of a derived type from the given address. Element size determined by template parameter
	 *
	 * \param[in] ptr				Pointer to the memory to be sent
	 * \param[in] num				The number of elements to send
	 * \param[in] dst				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a request object
	 */
    template<typename T>
    inline Request Irsend(const T *ptr, const int num, const int dst, const int tag, const Comm &comm) {
        return Irsend(ptr, num * sizeof(T), MEL::Datatype::CHAR, dst, tag, comm);
    };

    /**
	 * \ingroup P2P
	 * Probe an incoming message to predetermine its contents
	 *
	 * \param[in] source			The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a status object
	 */
	 inline Status Probe(const int source, const int tag, const Comm &comm) {
        MPI_Status status{};
        MEL_THROW( MPI_Probe(source, tag, (MPI_Comm) comm, &status), "Comm::Probe" );
        return status;
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Probe an incoming message to predetermine its contents
	 *
	 * \param[in] source			The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns a std::pair of a bool representing if a message was available and status object for that message
	 */
    inline std::pair<bool, Status> Iprobe(const int source, const int tag, const Comm &comm) {
        MPI_Status status{}; int f;
        MEL_THROW( MPI_Iprobe(source, tag, (MPI_Comm) comm, &f, &status), "Comm::Iprobe" );
        return std::make_pair(f != 0, status);
    };

	/**
	 * \ingroup P2P
	 * Probe the length of an incoming message. Element type is determined from the template parameter
	 *
	 * \param[in] status			A status object containing the rank and tag for the message
	 * \return						Returns the number of elements in the message
	 */
    template<typename T>
    inline int ProbeGetCount(const MPI_Status &status) {
        int c;
        MEL_THROW(MPI_Get_count(&status, MPI_CHAR, &c), "Comm::ProbeGetCount");
        return c / sizeof(T);
    };
    
	/**
	 * \ingroup P2P
	 * Probe the length of an incoming message 
	 *
	 * \param[in] datatype			The derived datatype of the elements 
	 * \param[in] status			A status object containing the rank and tag for the message
	 * \return						Returns the number of elements in the message
	 */
	inline int ProbeGetCount(const Datatype &datatype, const Status &status) {
        int c;
        MEL_THROW(MPI_Get_count(&status, (MPI_Datatype) datatype, &c), "Comm::ProbeGetCount");
        return c;
    };

	/**
	 * \ingroup P2P
	 * Probe the length of an incoming message. Element type is determined from the template parameter
	 *
	 * \param[in] src				The rank of the process to send to
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to send within
	 * \return						Returns the number of elements in the message
	 */
    template<typename T>
    inline int ProbeGetCount(const int src, const int tag, const Comm &comm) {
        Status status = Probe(src, tag, comm);
        return ProbeGetCount<T>(status);
    };

	/**
	 * \ingroup P2P
	 * Probe the length of an incoming message 
	 *
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] src				The rank of the process to probe from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \return						Returns the number of elements in the message
	 */
    inline int ProbeGetCount(const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Status status = Probe(src, tag, comm);
        return ProbeGetCount(datatype, status);
    };

	/**
	 * \ingroup P2P
	 * Recieve a message of known length into the given pointer 
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \return						Returns a status object
	 */
    inline Status Recv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Status status{};                                                                                                        
        MEL_THROW( MPI_Recv(ptr, num, (MPI_Datatype) datatype, src, tag, (MPI_Comm) comm, &status), "Comm::Recv" );                                
        return status;                                                                                                                
    };
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Recieve a message of known length into the given pointer 
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \param[out] rq				A request object
	 */                                                                                                                                
    inline void Irecv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Irecv(ptr, num, (MPI_Datatype) datatype, src, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irecv" );                                    
    };
	
	/**
	 * \ingroup P2P
	 * Non-Blocking. Recieve a message of known length into the given pointer 
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] datatype			The derived datatype of the elements
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \return						Returns a request object
	 */                                                                                                                               
    inline Request Irecv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Request rq{};                                                                                                            
        Irecv(ptr, num, datatype, src, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    };

	/// \cond HIDE
#define MEL_RECV(T, D) inline Status Recv(T *ptr, const int num, const int src, const int tag, const Comm &comm) {                    \
        Status status{};                                                                                                            \
        MEL_THROW( MPI_Recv(ptr, num, D, src, tag, (MPI_Comm) comm, &status), "Comm::Recv( " #T ", " #D " )" );                        \
        return status;                                                                                                                \
    }                                                                                                                                \
    inline void Irecv(T *ptr, const int num, const int src, const int tag, const Comm &comm, Request &rq) {                            \
        MEL_THROW( MPI_Irecv(ptr, num, D, src, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irecv( " #T ", " #D " )" );        \
    }                                                                                                                                \
    inline Request Irecv(T *ptr, const int num, const int src, const int tag, const Comm &comm) {                                    \
        Request rq{};                                                                                                                \
        Irecv(ptr, num, src, tag, comm, rq);                                                                                        \
        return rq;                                                                                                                    \
    }

    MEL_RECV(char,                            MPI_CHAR);
    MEL_RECV(wchar_t,                        MPI_WCHAR);

    MEL_RECV(float,                            MPI_FLOAT);
    MEL_RECV(double,                        MPI_DOUBLE);
    MEL_RECV(long double,                    MPI_LONG_DOUBLE);

    MEL_RECV(int8_t,                        MPI_INT8_T);
    MEL_RECV(int16_t,                        MPI_INT16_T);
    MEL_RECV(int32_t,                        MPI_INT32_T);
    MEL_RECV(int64_t,                        MPI_INT64_T);

    MEL_RECV(uint8_t,                        MPI_UINT8_T);
    MEL_RECV(uint16_t,                        MPI_UINT16_T);
    MEL_RECV(uint32_t,                        MPI_UINT32_T);
    MEL_RECV(uint64_t,                        MPI_UINT64_T);

#ifdef MEL_3
    MEL_RECV(std::complex<float>,            MPI_CXX_FLOAT_COMPLEX);
    MEL_RECV(std::complex<double>,            MPI_CXX_DOUBLE_COMPLEX);
    MEL_RECV(std::complex<long double>,        MPI_CXX_LONG_DOUBLE_COMPLEX);
    MEL_RECV(bool,                            MPI_CXX_BOOL);
#endif
#undef MEL_RECV
	/// \endcond

	/**
	 * \ingroup P2P
	 * Recieve a message of known length into the given pointer. Element size is determined from the template parameter
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \return						Returns a status object
	 */
    template<typename T>
    inline Status Recv(T *ptr, const int num, const int src, const int tag, const Comm &comm) {
        return Recv(ptr, num * sizeof(T), MEL::Datatype::CHAR, src, tag, comm);
    };

	/**
	 * \ingroup P2P
	 * Non-Blocking. Recieve a message of known length into the given pointer. Element size is determined from the template parameter 
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \param[out] rq				A request object
	 */   
    template<typename T>
    inline void Irecv(T *ptr, const int num, const int src, const int tag, const Comm &comm, Request &rq) {
        Irecv(ptr, num * sizeof(T), MEL::Datatype::CHAR, src, tag, comm, rq);
    };
    
	/**
	 * \ingroup P2P
	 * Non-Blocking. Recieve a message of known length into the given pointer. Element size is determined from the template parameter 
	 *
	 * \param[out] ptr				Pointer to the memory receive into
	 * \param[in] num				The number of elements to receive
	 * \param[in] src				The rank of the process to receive from
	 * \param[in] tag				A tag for the message
	 * \param[in] comm				The comm world to receive within
	 * \return						Returns a request object
	 */
	template<typename T>
    inline Request Irecv(T *ptr, const int num, const int src, const int tag, const Comm &comm) {
        return Irecv(ptr, num * sizeof(T), MEL::Datatype::CHAR, src, tag, comm);
    };

    /**
	 * \ingroup COL
	 * Broadcast an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] datatype			The derived datatype of the elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 */                                                                                                                      
    inline void Bcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm) {                                                                            
        MEL_THROW( MPI_Bcast(ptr, num, (MPI_Datatype) datatype, root, (MPI_Comm) comm), "Comm::Bcast" );                                                                
    };

    /**
	 * \ingroup COL
	 * Scatter an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				The number of elements to scatter, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 */
	inline void Scatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Scatter(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Scatter" );                                            
    };        
    
	/**
	 * \ingroup COL
	 * Scatter an array to all processes in comm, where all processes have an independent number of elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				Pointer to an array of number of elements to send to each process, significant only on root
	 * \param[in] displs			Pointer to an array of the element displacements of where each processes data is to be sent, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 */                                                                                                                                            
    inline void Scatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Scatterv(sptr, snum, displs, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Scatterv" );                                
    };    
                                                                                                                                                    
	/**
	 * \ingroup COL
	 * Gather an array from all processes in comm 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				The number of elements to receive, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 */                                                                                                                                
    inline void Gather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Gather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Gather" );                                            
    };    
      
	/**
	 * \ingroup COL
	 * Gather an array from all processes in comm, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process, significant only on root
	 * \param[in] displs			Pointer to an array of the displacements of elements to be received from each process, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 */                                                                                                                                                      
    inline void Gatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Gatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displs, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Gatherv" );                                    
    };
                                                                                                                                                            
    /**
	 * \ingroup COL
	 * Gather an array from all processes in comm and distribute it to all processes 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 */                                                                                                                            
    inline void Allgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Allgather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Allgather" );                                            
    };    
    
	/**
	 * \ingroup COL
	 * Gather an array from all processes in comm and distribute it to all processes, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] displ				Pointer to an array of the element displacements of where each processes data is to be received
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 */                                                                                                                                                          
    inline void Allgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Allgatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displ, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Allgather" );  
    };   
                                                                                                                                                
    /**
	 * \ingroup COL
	 * Broadcast from all processes to all processes
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				The number of elements to send
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to rnum * commsize elements to receive
	 * \param[in] rnum				The number of elements to receive from each process in comm
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 */                                                                                                                                
    inline void Alltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoall(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Alltoall" );                                                
    };    
    
	/**
	 * \ingroup COL
	 * Broadcast from all processes to all processes with independent number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 */                                                                                                                                         
    inline void Alltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoallv(sptr, snum, sdispl, (MPI_Datatype) sdatatype, rptr, rnum, rdispl, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Alltoallv" );                            
    };

	/**
	 * \ingroup COL
	 * Broadcast from all processes to all processes with independent derived datatypes and number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			Pointer to an array of the derived datatypes of the elements to send for each process
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			Pointer to an array of the derived datatypes of the elements to receive for each process
	 * \param[in] comm				The comm world to broadcast within
	 */    
    inline void Alltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoallw(sptr, snum, sdispl, (MPI_Datatype*) sdatatype, rptr, rnum, rdispl, (MPI_Datatype*) rdatatype, (MPI_Comm) comm), "Comm::Alltoallw" );
    };

    /**
	 * \ingroup COL
	 * Reduce an array of known length across all processes in comm using the given operation
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer, significant only on root
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] root				The process to receive the result
	 * \param[in] comm				The comm world to reduce within
	 */                                                                                                                             
    inline void Reduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        MEL_THROW( MPI_Reduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, root, (MPI_Comm) comm), "Comm::Reduce" );                                                
    };                                                                                                                                                
    
	/**
	 * \ingroup COL
	 * Reduce an array of known length across all processes in comm using the given operation, and distribute the result to all processes
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] comm				The comm world to reduce within
	 */
	inline void Allreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm) {
        MEL_THROW( MPI_Allreduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, (MPI_Comm) comm), "Comm::Allreduce" );                                            
    };
    
#ifdef MEL_3
	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] datatype			The derived datatype of the elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 * \param[out] rq				A request object
	 */
    inline void Ibcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ibcast(ptr, num, (MPI_Datatype) datatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibcast");
    };
    
	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] datatype			The derived datatype of the elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 * \return						Returns a request object
	 */
	inline Request Ibcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm) {
        Request rq{};
        Ibcast(ptr, num, datatype, root, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Scatter an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				The number of elements to scatter, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 * \param[out] rq				A request object
	 */
    inline void Iscatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iscatter(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatter");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Scatter an array to all processes in comm, where all processes know how many elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				The number of elements to scatter, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 * \return						Returns a request object
	 */
    inline Request Iscatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Iscatter(sptr, snum, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Scatter an array to all processes in comm, where all processes have an independent number of elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				Pointer to an array of number of elements to send to each process, significant only on root
	 * \param[in] displs			Pointer to an array of the element displacements of where each processes data is to be sent, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 * \param[out] rq				A request object
	 */
    inline void Iscatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iscatterv(sptr, snum, displs, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatterv");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Scatter an array to all processes in comm, where all processes have an independent number of elements to expect 
	 *
	 * \param[in] sptr				Pointer to the memory to scatter, significant only on root
	 * \param[in] snum				Pointer to an array of number of elements to send to each process, significant only on root
	 * \param[in] displs			Pointer to an array of the element displacements of where each processes data is to be sent, significant only on root
	 * \param[in] sdatatype			The derived datatype of the elements to send, significant only on root
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to scatter within
	 * \return						Returns a request object
	 */
    inline Request Iscatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Iscatterv(sptr, snum, displs, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				The number of elements to receive, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 * \param[out] rq				A request object
	 */ 
    inline void Igather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Igather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igather");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				The number of elements to receive, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 * \return						Returns a request object
	 */ 
    inline Request Igather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Igather(sptr, snum, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process, significant only on root
	 * \param[in] displs			Pointer to an array of the displacements of elements to be received from each process, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 * \param[out] rq				A request object
	 */ 
    inline void Igatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Igatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displs, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igatherv");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into, significant only on root
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process, significant only on root
	 * \param[in] displs			Pointer to an array of the displacements of elements to be received from each process, significant only on root
	 * \param[in] rdatatype			The derived datatype of the elements to receive, significant only on root
	 * \param[in] root				The rank of the process to send to
	 * \param[in] comm				The comm world to gather within
	 * \return						Returns a request object
	 */ 
    inline Request Igatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Igatherv(sptr, snum, sdatatype, rptr, rnum, displs, rdatatype, root, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm and distribute it to all processes 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 * \param[out] rq				A request object
	 */  
    inline void Iallgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iallgather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgather");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm and distribute it to all processes 
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				The number of elements to receive
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 * \return						Returns a request object
	 */  
    inline Request Iallgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Iallgather(sptr, snum, sdatatype, rptr, rnum, rdatatype, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm and distribute it to all processes, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] displ				Pointer to an array of the element displacements of where each processes data is to be received
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 * \param[out] rq				A request object
	 */  
    inline void Iallgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iallgatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displ, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgather");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Gather an array from all processes in comm and distribute it to all processes, where all processes have an independent number of elements to send
	 *
	 * \param[in] sptr				Pointer to the memory to gather
	 * \param[in] snum				The number of elements to gather
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the memory to receive into
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] displ				Pointer to an array of the element displacements of where each processes data is to be received
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to gather within
	 * \return						Returns a request object
	 */  
    inline Request Iallgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Iallgatherv(sptr, snum, sdatatype, rptr, rnum, displ, rdatatype, comm, rq);
        return rq;
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				The number of elements to send
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to rnum * commsize elements to receive
	 * \param[in] rnum				The number of elements to receive from each process in comm
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 * \param[out] rq				A request object
	 */
    inline void Ialltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoall(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoall");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				The number of elements to send
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to rnum * commsize elements to receive
	 * \param[in] rnum				The number of elements to receive from each process in comm
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 * \return						Returns a request object
	 */
    inline Request Ialltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoall(sptr, snum, sdatatype, rptr, rnum, rdatatype, comm, rq);
        return rq;
    };
    
	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes with independent number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 * \param[out] rq				A request object
	 */
    inline void Ialltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoallv(sptr, snum, sdispl, (MPI_Datatype) sdatatype, rptr, rnum, rdispl, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoallv");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes with independent number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			The derived datatype of the elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			The derived datatype of the elements to receive
	 * \param[in] comm				The comm world to broadcast within
	 * \return						Returns a request object
	 */
    inline Request Ialltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoallv(sptr, snum, sdispl, sdatatype, rptr, rnum, rdispl, rdatatype, comm, rq);
        return rq;
    };
    
	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes with independent derived datatypes and number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			Pointer to an array of the derived datatypes of the elements to send for each process
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			Pointer to an array of the derived datatypes of the elements to receive for each process
	 * \param[in] comm				The comm world to broadcast within
	 * \param[out] rq				A request object
	 */   
    inline void Ialltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoallw(sptr, snum, sdispl, (MPI_Datatype*) sdatatype, rptr, rnum, rdispl, (MPI_Datatype*) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoallw");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast from all processes to all processes with independent derived datatypes and number of elements for each process
	 *
	 * \param[in] sptr				Pointer to snum elements to send
	 * \param[in] snum				Pointer to an array of the number of elements to send from each process
	 * \param[in] sdispl			Pointer to an array of element displacements of where each processes data is to be sent from
	 * \param[in] sdatatype			Pointer to an array of the derived datatypes of the elements to send for each process
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] rnum				Pointer to an array of the number of elements to receive from each process
	 * \param[in] rdispl			Pointer to an array of element displacements of where each processes data is to be recieved to
	 * \param[in] rdatatype			Pointer to an array of the derived datatypes of the elements to receive for each process
	 * \param[in] comm				The comm world to broadcast within
	 * \return						Returns a request object
	 */   
    inline Request Ialltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoallw(sptr, snum, sdispl, sdatatype, rptr, rnum, rdispl, rdatatype, comm, rq);
        return rq;
    };
    
	/**
	 * \ingroup COL
	 * Non-Blocking. Reduce an array of known length across all processes in comm using the given operation
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer, significant only on root
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] root				The process to receive the result
	 * \param[in] comm				The comm world to reduce within
	 * \param[out] rq				A request object
	 */     
    inline void Ireduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ireduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ireduce");
    };

	/**
	 * \ingroup COL
	 * Non-Blocking. Reduce an array of known length across all processes in comm using the given operation
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer, significant only on root
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] root				The process to receive the result
	 * \param[in] comm				The comm world to reduce within
	 * \return						Returns a request object
	 */     
    inline Request Ireduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        Request rq{};
        Ireduce(sptr, rptr, num, datatype, op, root, comm, rq);
        return rq;
    };    
    
	/**
	 * \ingroup COL
	 * Non-Blocking. Reduce an array of known length across all processes in comm using the given operation, and distribute the result to all processes
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] comm				The comm world to reduce within
	 * \param[out] rq				A request object
	 */
    inline void Iallreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Iallreduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallreduce" );                            
    }; 
	
	/**
	 * \ingroup COL
	 * Non-Blocking. Reduce an array of known length across all processes in comm using the given operation, and distribute the result to all processes
	 *
	 * \param[in] sptr				Pointer to num elements to send
	 * \param[out] rptr				Pointer to the receive buffer
	 * \param[in] num				The number of elements in the array
	 * \param[in] datatype			The derived datatype of the elements to reduce
	 * \param[in] op				The operation to perform for the reduction
	 * \param[in] comm				The comm world to reduce within
	 * \return						Returns a request object
	 */                                                                                                                                                       
    inline Request Iallreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm) {
        Request rq{};                                                                                                                                        
        Iallreduce(sptr, rptr, num, datatype, op, comm, rq);                                                                                                    
        return rq;                                                                                                                                            
    };
#endif

	/// \cond HIDE
#define MEL_COLLECTIVE(T, D) inline void Bcast(T *ptr, const int num, const int root, const Comm &comm) {                                                    \
        MEL_THROW( MPI_Bcast(ptr, num, D, root, (MPI_Comm) comm), "Comm::Bcast( " #T ", " #D " )" );                                                        \
    }                                                                                                                                                        \
    /* Scatter / Scatterv */                                                                                                                                \
    inline void Scatter(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm) {                                                \
        MEL_THROW( MPI_Scatter(sptr, snum, D, rptr, rnum, D, root, (MPI_Comm) comm), "Comm::Scatter( " #T ", " #D " )" );                                    \
    }                                                                                                                                                        \
    inline void Scatterv(T *sptr, const int *snum, const int *displs, T *rptr, const int rnum, const int root, const Comm &comm) {                            \
        MEL_THROW( MPI_Scatterv(sptr, snum, displs, D, rptr, rnum, D, root, (MPI_Comm) comm), "Comm::Scatterv( " #T ", " #D " )" );                            \
    }                                                                                                                                                        \
    /* Gather / Gatherv */                                                                                                                                    \
    inline void Gather(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm) {                                                \
        MEL_THROW( MPI_Gather(sptr, snum, D, rptr, rnum, D, root, (MPI_Comm) comm), "Comm::Gather( " #T ", " #D " )" );                                        \
    }                                                                                                                                                        \
    inline void Gatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displs, const int root, const Comm &comm) {                            \
        MEL_THROW( MPI_Gatherv(sptr, snum, D, rptr, rnum, displs, D, root, (MPI_Comm) comm), "Comm::Gatherv( " #T ", " #D " )" );                            \
    }                                                                                                                                                        \
    /* Allgather / Allgatherv */                                                                                                                            \
    inline void Allgather(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm) {                                                                \
        MEL_THROW( MPI_Allgather(sptr, snum, D, rptr, rnum, D, (MPI_Comm) comm), "Comm::Allgather( " #T ", " #D " )" );                                        \
    }                                                                                                                                                        \
    inline void Allgatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displ, const Comm &comm) {                                            \
        MEL_THROW( MPI_Allgatherv(sptr, snum, D, rptr, rnum, displ, D, (MPI_Comm) comm), "Comm::Allgatherv( " #T ", " #D " )" );                            \
    }                                                                                                                                                        \
    /* Alltoall / Alltoallv */                                                                                                                                \
    inline void Alltoall(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm) {                                                                \
        MEL_THROW( MPI_Alltoall(sptr, snum, D, rptr, rnum, D, (MPI_Comm) comm), "Comm::Alltoall( " #T ", " #D " )" );                                        \
    }                                                                                                                                                        \
    inline void Alltoallv(T *sptr, const int *snum, const int *sdispl, T *rptr, const int *rnum, const int *rdispl, const Comm &comm) {                        \
        MEL_THROW( MPI_Alltoallv(sptr, snum, sdispl, D, rptr, rnum, rdispl, D, (MPI_Comm) comm), "Comm::Alltoallv( " #T ", " #D " )" );                        \
    }                                                                                                                                                        \
    /* Reduce / Allreduce */                                                                                                                                \
    inline void Reduce(T *sptr, T *rptr, const int num, const Op &op, const int root, const Comm &comm) {                                                    \
        MEL_THROW( MPI_Reduce(sptr, rptr, num, D, (MPI_Op) op, root, (MPI_Comm) comm), "Comm::Reduce( " #T ", " #D " )" );                                    \
    }                                                                                                                                                        \
    inline void Allreduce(T *sptr, T *rptr, const int num, const Op &op, const Comm &comm) {                                                                \
        MEL_THROW( MPI_Allreduce(sptr, rptr, num, D, (MPI_Op) op, (MPI_Comm) comm), "Comm::Allreduce( " #T ", " #D " )" );                                    \
    }                                                                                                                                                        

#define MEL_3_COLLECTIVE(T, D) inline void Ibcast(T *ptr, const int num, const int root, const Comm &comm, Request &rq) {                                    \
        MEL_THROW( MPI_Ibcast(ptr, num, D, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibcast( " #T ", " #D " )" );                                    \
    }                                                                                                                                                        \
    inline Request Ibcast(T *ptr, const int num, const int root, const Comm &comm) {                                                                        \
        Request rq{};                                                                                                                                        \
        Ibcast(ptr, num, root, comm, rq);                                                                                                                    \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    /* Scatter / Scatterv */                                                                                                                                \
    inline void Iscatter(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm, Request &rq) {                                    \
        MEL_THROW( MPI_Iscatter(sptr, snum, D, rptr, rnum, D, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatter( " #T ", " #D " )" );                \
    }                                                                                                                                                        \
    inline Request Iscatter(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm) {                                            \
        Request rq{};                                                                                                                                        \
        Iscatter(sptr, snum, rptr, rnum, root, comm, rq);                                                                                                    \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    inline void Iscatterv(T *sptr, const int *snum, const int *displs, T *rptr, const int rnum, const int root, const Comm &comm, Request &rq) {            \
        MEL_THROW( MPI_Iscatterv(sptr, snum, displs, D, rptr, rnum, D, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatterv( " #T ", " #D " )" );    \
    }                                                                                                                                                        \
    inline Request Iscatterv(T *sptr, const int *snum, const int *displs, T *rptr, const int rnum, const int root, const Comm &comm) {                        \
        Request rq{};                                                                                                                                        \
        Iscatterv(sptr, snum, displs, rptr, rnum, root, comm, rq);                                                                                            \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    /* Gather / Gatherv */                                                                                                                                    \
    inline void Igather(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm, Request &rq) {                                    \
        MEL_THROW( MPI_Igather(sptr, snum, D, rptr, rnum, D, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igather( " #T ", " #D " )" );                \
    }                                                                                                                                                        \
    inline Request Igather(T *sptr, const int snum, T *rptr, const int rnum, const int root, const Comm &comm) {                                            \
        Request rq{};                                                                                                                                        \
        Igather(sptr, snum, rptr, rnum, root, comm, rq);                                                                                                    \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    inline void Igatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displs, const int root, const Comm &comm, Request &rq) {                \
        MEL_THROW( MPI_Igatherv(sptr, snum, D, rptr, rnum, displs, D, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igatherv( " #T ", " #D " )" );        \
    }                                                                                                                                                        \
    inline Request Igatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displs, const int root, const Comm &comm) {                        \
        Request rq{};                                                                                                                                        \
        Igatherv(sptr, snum, rptr, rnum, displs, root, comm, rq);                                                                                            \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    /* Allgather / Allgatherv */                                                                                                                            \
    inline void Iallgather(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm, Request &rq) {                                                \
        MEL_THROW( MPI_Iallgather(sptr, snum, D, rptr, rnum, D, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgather( " #T ", " #D " )" );                \
    }                                                                                                                                                        \
    inline Request Iallgather(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm) {                                                            \
        Request rq{};                                                                                                                                        \
        Iallgather(sptr, snum, rptr, rnum, comm, rq);                                                                                                        \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    inline void Iallgatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displ, const Comm &comm, Request &rq) {                            \
        MEL_THROW( MPI_Iallgatherv(sptr, snum, D, rptr, rnum, displ, D, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgatherv( " #T ", " #D " )" );        \
    }                                                                                                                                                        \
    inline Request Iallgatherv(T *sptr, const int snum, T *rptr, const int *rnum, const int *displ, const Comm &comm) {                                        \
        Request rq{};                                                                                                                                        \
        Iallgatherv(sptr, snum, rptr, rnum, displ, comm, rq);                                                                                                \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    /* Alltoall / Alltoallv */                                                                                                                                \
    inline void Ialltoall(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm, Request &rq) {                                                \
        MEL_THROW( MPI_Ialltoall(sptr, snum, D, rptr, rnum, D, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoall( " #T ", " #D " )" );                    \
    }                                                                                                                                                        \
    inline Request Ialltoall(T *sptr, const int snum, T *rptr, const int rnum, const Comm &comm) {                                                            \
        Request rq{};                                                                                                                                        \
        Ialltoall(sptr, snum, rptr, rnum, comm, rq);                                                                                                        \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    inline void Ialltoallv(T *sptr, const int *snum, const int *sdispl, T *rptr, const int *rnum, const int *rdispl, const Comm &comm, Request &rq) {        \
        MEL_THROW( MPI_Ialltoallv(sptr, snum, sdispl, D, rptr, rnum, rdispl, D, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoallv( " #T ", " #D " )" );    \
    }                                                                                                                                                        \
    inline Request Ialltoallv(T *sptr, const int *snum, const int *sdispl, T *rptr, const int *rnum, const int *rdispl, const Comm &comm) {                    \
        Request rq{};                                                                                                                                        \
        Ialltoallv(sptr, snum, sdispl, rptr, rnum, rdispl, comm, rq);                                                                                        \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    /* Reduce / Allreduce */                                                                                                                                \
    inline void Ireduce(T *sptr, T *rptr, const int num, const Op &op, const int root, const Comm &comm, Request &rq) {                                        \
        MEL_THROW( MPI_Ireduce(sptr, rptr, num, D, (MPI_Op) op, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ireduce( " #T ", " #D " )" );            \
    }                                                                                                                                                        \
    inline Request Ireduce(T *sptr, T *rptr, const int num, const Op &op, const int root, const Comm &comm) {                                                \
        Request rq{};                                                                                                                                        \
        Ireduce(sptr, rptr, num, op, root, comm, rq);                                                                                                        \
        return rq;                                                                                                                                            \
    }                                                                                                                                                        \
    inline void Iallreduce(T *sptr, T *rptr, const int num, const Op &op, const Comm &comm, Request &rq) {                                                    \
        MEL_THROW( MPI_Iallreduce(sptr, rptr, num, D, (MPI_Op) op, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallreduce( " #T ", " #D " )" );            \
    }                                                                                                                                                        \
    inline Request Iallreduce(T *sptr, T *rptr, const int num, const Op &op, const Comm &comm) {                                                            \
        Request rq{};                                                                                                                                        \
        Iallreduce(sptr, rptr, num, op, comm, rq);                                                                                                            \
        return rq;                                                                                                                                            \
    }

    MEL_COLLECTIVE(char,                        MPI_CHAR);
    MEL_COLLECTIVE(wchar_t,                        MPI_WCHAR);

    MEL_COLLECTIVE(float,                        MPI_FLOAT);
    MEL_COLLECTIVE(double,                        MPI_DOUBLE);
    MEL_COLLECTIVE(long double,                    MPI_LONG_DOUBLE);

    MEL_COLLECTIVE(int8_t,                        MPI_INT8_T);
    MEL_COLLECTIVE(int16_t,                        MPI_INT16_T);
    MEL_COLLECTIVE(int32_t,                        MPI_INT32_T);
    MEL_COLLECTIVE(int64_t,                        MPI_INT64_T);

    MEL_COLLECTIVE(uint8_t,                        MPI_UINT8_T);
    MEL_COLLECTIVE(uint16_t,                    MPI_UINT16_T);
    MEL_COLLECTIVE(uint32_t,                    MPI_UINT32_T);
    MEL_COLLECTIVE(uint64_t,                    MPI_UINT64_T);

#ifdef MEL_3
    MEL_COLLECTIVE(std::complex<float>,            MPI_CXX_FLOAT_COMPLEX);
    MEL_COLLECTIVE(std::complex<double>,        MPI_CXX_DOUBLE_COMPLEX);
    MEL_COLLECTIVE(std::complex<long double>,    MPI_CXX_LONG_DOUBLE_COMPLEX);
    MEL_COLLECTIVE(bool,                        MPI_CXX_BOOL);

    MEL_3_COLLECTIVE(char,                        MPI_CHAR);
    MEL_3_COLLECTIVE(wchar_t,                    MPI_WCHAR);

    MEL_3_COLLECTIVE(float, MPI_FLOAT);
    MEL_3_COLLECTIVE(double,                    MPI_DOUBLE);
    MEL_3_COLLECTIVE(long double,                MPI_LONG_DOUBLE);

    MEL_3_COLLECTIVE(int8_t,                    MPI_INT8_T);
    MEL_3_COLLECTIVE(int16_t,                    MPI_INT16_T);
    MEL_3_COLLECTIVE(int32_t,                    MPI_INT32_T);
    MEL_3_COLLECTIVE(int64_t,                    MPI_INT64_T);

    MEL_3_COLLECTIVE(uint8_t,                    MPI_UINT8_T);
    MEL_3_COLLECTIVE(uint16_t,                    MPI_UINT16_T);
    MEL_3_COLLECTIVE(uint32_t,                    MPI_UINT32_T);
    MEL_3_COLLECTIVE(uint64_t,                    MPI_UINT64_T);

    MEL_3_COLLECTIVE(std::complex<float>,        MPI_CXX_FLOAT_COMPLEX);
    MEL_3_COLLECTIVE(std::complex<double>,        MPI_CXX_DOUBLE_COMPLEX);
    MEL_3_COLLECTIVE(std::complex<long double>,    MPI_CXX_LONG_DOUBLE_COMPLEX);
    MEL_3_COLLECTIVE(bool,                        MPI_CXX_BOOL);
#endif

#undef MEL_COLLECTIVE
#undef MEL_3_COLLECTIVE
	/// \endcond

	/**
	 * \ingroup COL
	 * Broadcast an array to all processes in comm, where all processes know how many elements to expect
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 */
	template<typename T>
	inline void Bcast(T *ptr, const int num, const int root, const Comm &comm) {
		Bcast(ptr, num * sizeof(T), MEL::Datatype::CHAR, root, comm);
	};

#ifdef MEL_3

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast an array to all processes in comm, where all processes know how many elements to expect
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 * \param[out] rq				A request object
	 */
	template<typename T>
	inline void Ibcast(T *ptr, const int num, const int root, const Comm &comm, Request &rq) {
		Ibcast(ptr, num * sizeof(T), MEL::Datatype::CHAR, root, comm, rq);
	}; 

	/**
	 * \ingroup COL
	 * Non-Blocking. Broadcast an array to all processes in comm, where all processes know how many elements to expect
	 *
	 * \param[in,out] ptr			Pointer to the memory receive into
	 * \param[in] num				The number of elements to broadcast
	 * \param[in] root				The rank of the process to send from
	 * \param[in] comm				The comm world to broadcast within
	 * \return						Returns a request object
	 */
	template<typename T>
	inline Request Ibcast(T *ptr, const int num, const int root, const Comm &comm) {
		return Ibcast(ptr, num * sizeof(T), MEL::Datatype::CHAR, root, comm);
	};

#endif

    enum class LockType {
        EXCLUSIVE = MPI_LOCK_EXCLUSIVE,
        SHARED = MPI_LOCK_SHARED
    };

    struct Win {
        static const Win WIN_NULL;

        MPI_Win win;

        Win() : win(MPI_WIN_NULL) {};
        explicit Win(const MPI_Win &_w) : win(_w) {};
        inline Win& operator=(const MPI_Win &_w) {
            win = _w;
            return *this;
        };
        explicit operator MPI_Win() const {
            return win;
        };

        inline bool operator==(const Win &rhs) const {
            return win == rhs.win;
        };
        inline bool operator!=(const Win &rhs) const {
            return win != rhs.win;
        };
    };

#ifdef MEL_IMPLEMENTATION
    const Win Win::WIN_NULL = Win(MPI_WIN_NULL);
#endif

    /**
	 * \ingroup  Win
     * Create a window error handler
	 *
	 * \param[in] func	The function to use
	 * \return			Returns a handle to an error handler
	 */
	inline ErrorHandler WinCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Win_create_errhandler((MPI_Win_errhandler_function*) func, &errHndl), "RMA::WinCreateErrorHandler" );
        return ErrorHandler(errHndl);
    };

	/**
	 * \ingroup  Win
     * Set the error handler for a window
	 *
	 * \param[in] win		The file to attach to
	 * \param[in] errHndl	The handler to use
	 */
    inline void WinSetErrorHandler(const Win &win, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_Win_set_errhandler((MPI_Win) win, (MPI_Errhandler) errHndl), "RMA::WinSetErrorHandler" );
    };

	/**
	 * \ingroup  Win
     * Set the error handler for a window
	 *
	 * \param[in] win	The file to attach to
	 * \param[in] func	The function to use
	 */
    inline void WinSetErrorHandler(const Win &win, ErrorHandlerFunc func) {
        WinSetErrorHandler(win, WinCreateErrorHandler(func));
    };

	/**
	 * \ingroup  Win
     * Get the error handler for a window
	 *
	 * \param[in] win	The file to attach to
	 * \return			Returns a handle to the error handler
	 */
    inline ErrorHandler WinGetErrorHandler(const Win &win) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Win_get_errhandler((MPI_Win) win, &errHndl), "RMA::WinGetErrorHandler");
        return ErrorHandler(errHndl);
    };

    /**
	 * \ingroup  Win
     * Create a window on memory allocated with MPI/MEL alloc functions
	 *
	 * \param[in] ptr			Pointer to the memory to be mapped
	 * \param[in] size			The number of elements to be mapped
	 * \param[in] disp_unit		The size of each element in bytes
	 * \param[in] comm			The comm world to map the window within
	 * \return					Returns a handle to the window
	 */
    inline Win WinCreate(void *ptr, const Aint size, const int disp_unit, const Comm &comm) {
        MPI_Win win;                                                                        
        MEL_THROW( MPI_Win_create(ptr, size * disp_unit, disp_unit, MPI_INFO_NULL, (MPI_Comm) comm, (MPI_Win*) &win), "RMA::WinCreate" );
        MEL_THROW( MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN), "RMA::WinCreate(SetErrorHandler)" );                                                                
        return Win(win);
    };
    
	/**
	 * \ingroup  Win
     * Create a window on memory allocated with MPI/MEL alloc functions. Element size determined from template parameter
	 *
	 * \param[in] ptr			Pointer to the memory to be mapped
	 * \param[in] size			The number of elements to be mapped
	 * \param[in] comm			The comm world to map the window within
	 * \return					Returns a handle to the window
	 */
	template<typename T> 
    inline Win WinCreate(T *ptr, const Aint size, const Comm &comm) {
        return WinCreate(ptr, size, sizeof(T), comm);
    };

    /**
	 * \ingroup  Win
     * Synchronize the RMA access epoch for win across all processes attached to it
	 *
	 * \param[in] win			The window to synchronize
	 * \param[in] assert_tag	Program assertion
	 */
    inline void WinFence(const Win &win, const int assert_tag) {
        MEL_THROW( MPI_Win_fence(assert_tag, (MPI_Win) win), "RMA::WinFence" );
    };
    
	/**
	 * \ingroup  Win
     * Synchronize the RMA access epoch for win across all processes attached to it
	 *
	 * \param[in] win			The window to synchronize
	 */
	inline void WinFence(const Win &win) {
        WinFence(win, 0);
    };

	/**
	 * \ingroup  Win
     * Get the lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 * \param[in] assert_tag	Program assertion
	 * \param[in] lock_type		The mode to get the lock in
	 */
    inline void WinLock(const Win &win, const int rank, const int assert_tag, const LockType lock_type) {
        MEL_THROW( MPI_Win_lock((int) lock_type, rank, assert_tag, (MPI_Win) win), "RMA::WinLock" );
    };
    
	/**
	 * \ingroup  Win
     * Get the lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 * \param[in] lock_type		The mode to get the lock in
	 */
	inline void WinLock(const Win &win, const int rank, const LockType lock_type) {
        WinLock(win, rank, 0, lock_type);
    };

	/**
	 * \ingroup  Win
     * Get an exclusive lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 * \param[in] assert_tag	Program assertion
	 */
    inline void WinLockExclusive(const Win &win, const int rank, const int assert_tag) {
        WinLock(win, rank, assert_tag, LockType::EXCLUSIVE);
    };
    
	/**
	 * \ingroup  Win
     * Get an exclusive lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 */
	inline void WinLockExclusive(const Win &win, const int rank) {
        WinLockExclusive(win, rank, 0);
    };

	/**
	 * \ingroup  Win
     * Get a shared lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 * \param[in] assert_tag	Program assertion
	 */
    inline void WinLockShared(const Win &win, const int rank, const int assert_tag) {
        WinLock(win, rank, assert_tag, LockType::SHARED);
    };

	/**
	 * \ingroup  Win
     * Get a shared lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 */
    inline void WinLockShared(const Win &win, const int rank) {
        WinLockShared(win, rank, 0);
    };

	/**
	 * \ingroup  Win
     * Release the lock on an RMA access epoch for a window
	 *
	 * \param[in] win			The window to lock
	 * \param[in] rank			The process rank to get the lock from
	 */
    inline void WinUnlock(const Win &win, const int rank) {
        MEL_THROW( MPI_Win_unlock(rank, (MPI_Win) win), "RMA::WinUnlock" );
    };

    /**
	 * \ingroup  Win
     * Put data into the mapped window of another process
	 *
	 * \param[in] origin_ptr		Pointer to the array to put
	 * \param[in] origin_num		The number of elements to put from the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be put
	 * \param[in] target_disp		Element displacement into the window to put data into
	 * \param[in] target_num		The number of elements to put into the window
	 * \param[in] target_datatype	The derived datatype of the elements to be put into the window
	 * \param[in] target_rank		Rank of the process to put into
	 * \param[in] win				The window to put into
	 */
    inline void Put(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        MEL_THROW( MPI_Put(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win), "RMA::Put" );
    };

	/**
	 * \ingroup  Win
     * Accumulate data into the mapped window of another process
	 *
	 * \param[in] origin_ptr		Pointer to the array to put
	 * \param[in] origin_num		The number of elements to put from the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be put
	 * \param[in] target_disp		Element displacement into the window to put data into
	 * \param[in] target_num		The number of elements to put into the window
	 * \param[in] target_datatype	The derived datatype of the elements to be put into the window
	 * \param[in] op				The MPI operation to use
	 * \param[in] target_rank		Rank of the process to put into
	 * \param[in] win				The window to put into
	 */
    inline void Accumulate(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const Op &op, const int target_rank, const Win &win) {
        MEL_THROW( MPI_Accumulate(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Op) op, (MPI_Win) win), "RMA::Accumulate" );
    };

    /**
	 * \ingroup  Win
     * Get data from the mapped window of another process
	 *
	 * \param[out] origin_ptr		Pointer to the array to get
	 * \param[in] origin_num		The number of elements to get into the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be gotten
	 * \param[in] target_disp		Element displacement into the window to get data from
	 * \param[in] target_num		The number of elements to get from the window
	 * \param[in] target_datatype	The derived datatype of the elements to be gotten from the window
	 * \param[in] target_rank		Rank of the process to get from
	 * \param[in] win				The window to get from
	 */
    inline void Get(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        MEL_THROW( MPI_Get(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win), "RMA::Get" );
    };

#ifdef MEL_3
	
	/**
	 * \ingroup  Win
     * Get the lock on an RMA access epoch for a window on all attached processes
	 *
	 * \param[in] win			The window to lock
	 * \param[in] assert_tag	Program assertion
	 */
    inline void WinLockAll(const Win &win, const int assert_tag) {
        MEL_THROW(MPI_Win_lock_all(assert_tag, (MPI_Win) win), "RMA::WinLockAll");
    };

	/**
	 * \ingroup  Win
     * Get the lock on an RMA access epoch for a window on all attached processes
	 *
	 * \param[in] win			The window to lock
	 */
    inline void WinLockAll(const Win &win) {
        WinLockAll(win, 0);
    };

	/**
	 * \ingroup  Win
     * Release the lock on an RMA access epoch for a window on all attached processes
	 *
	 * \param[in] win			The window to unlock
	 */
    inline void WinUnlockAll(const Win &win) {
        MEL_THROW(MPI_Win_unlock_all((MPI_Win) win), "RMA::WinUnlockAll");
    };

	/**
	 * \ingroup  Win
     * Force all operations within an RMA access epoch for a window to finish
	 *
	 * \param[in] win			The window to flush
	 * \param[in] rank			Rank to force synchronization with
	 */
    inline void WinFlush(const Win &win, const int rank) {
        MEL_THROW(MPI_Win_flush(rank, (MPI_Win) win), "RMA::WinFlush");
    };
    
	/**
	 * \ingroup  Win
     * Force all operations within an RMA access epoch for a window to finish for all attached processes
	 *
	 * \param[in] win			The window to flush
	 */
	inline void WinFlushAll(const Win &win) {
        MEL_THROW(MPI_Win_flush_all((MPI_Win) win), "RMA::WinFlushAll");
    };

	/**
	 * \ingroup  Win
     * Force all local operations within an RMA access epoch for a window to finish
	 *
	 * \param[in] win			The window to flush
	 * \param[in] rank			Rank to force synchronization with
	 */
    inline void WinFlushLocal(const Win &win, const int rank) {
        MEL_THROW(MPI_Win_flush_local(rank, (MPI_Win) win), "RMA::WinFlushLocal");
    };

	/**
	 * \ingroup  Win
     * Force all local operations within an RMA access epoch for a window to finish for all attached processes
	 *
	 * \param[in] win			The window to flush
	 */
    inline void WinFlushLocalAll(const Win &win) {
        MEL_THROW(MPI_Win_flush_local_all((MPI_Win) win), "RMA::WinFlushLocalAll");
    };

	/**
	 * \ingroup  Win
     * Synchronize the public and private copies of the window
	 *
	 * \param[in] win			The window to synchronize
	 */
    inline void WinSync(const Win &win) {
        MEL_THROW(MPI_Win_sync((MPI_Win) win), "RMA::WinSync");
    };

	/**
	 * \ingroup  Win
     * Put data into the mapped window of another process
	 *
	 * \param[in] origin_ptr		Pointer to the array to put
	 * \param[in] origin_num		The number of elements to put from the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be put
	 * \param[in] target_disp		Element displacement into the window to put data into
	 * \param[in] target_num		The number of elements to put into the window
	 * \param[in] target_datatype	The derived datatype of the elements to be put into the window
	 * \param[in] target_rank		Rank of the process to put into
	 * \param[in] win				The window to put into
	 * \param[out] rq				A request object
	 */
    inline void Rput(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        MEL_THROW(MPI_Rput(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win, (MPI_Request*) &rq), "RMA::Rput");
    };

	/**
	 * \ingroup  Win
     * Put data into the mapped window of another process
	 *
	 * \param[in] origin_ptr		Pointer to the array to put
	 * \param[in] origin_num		The number of elements to put from the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be put
	 * \param[in] target_disp		Element displacement into the window to put data into
	 * \param[in] target_num		The number of elements to put into the window
	 * \param[in] target_datatype	The derived datatype of the elements to be put into the window
	 * \param[in] target_rank		Rank of the process to put into
	 * \param[in] win				The window to put into
	 * \return						Returns a request object
	 */
    inline Request Rput(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Request rq;
        Rput(origin_ptr, origin_num, origin_datatype, target_disp, target_num, target_datatype, target_rank, win, rq);
        return rq;
    };

	/**
	 * \ingroup  Win
     * Get data from the mapped window of another process
	 *
	 * \param[out] origin_ptr		Pointer to the array to get
	 * \param[in] origin_num		The number of elements to get into the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be gotten
	 * \param[in] target_disp		Element displacement into the window to get data from
	 * \param[in] target_num		The number of elements to get from the window
	 * \param[in] target_datatype	The derived datatype of the elements to be gotten from the window
	 * \param[in] target_rank		Rank of the process to get from
	 * \param[in] win				The window to get from
	 * \param[out] rq				A request object
	 */
    inline void Rget(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        MEL_THROW(MPI_Rget(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win, (MPI_Request*) &rq), "RMA::Rget");
    };
    
	/**
	 * \ingroup  Win
     * Get data from the mapped window of another process
	 *
	 * \param[out] origin_ptr		Pointer to the array to get
	 * \param[in] origin_num		The number of elements to get into the local array
	 * \param[in] origin_datatype	The derived datatype of the elements to be gotten
	 * \param[in] target_disp		Element displacement into the window to get data from
	 * \param[in] target_num		The number of elements to get from the window
	 * \param[in] target_datatype	The derived datatype of the elements to be gotten from the window
	 * \param[in] target_rank		Rank of the process to get from
	 * \param[in] win				The window to get from
	 * \return						Returns a request object
	 */
    inline Request Rget(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Request rq;
        Rget(origin_ptr, origin_num, origin_datatype, target_disp, target_num, target_datatype, target_rank, win, rq);
        return rq;
    };
    
#endif
	
	/**
	 * \ingroup  Win
     * Free an RMA window
	 *
	 * \param[in] win			The window to free
	 */
    inline void WinFree(Win &win) {
        if (win != MEL::Win::WIN_NULL)
            MEL_THROW( MPI_Win_free((MPI_Win*) &win), "RMA::FreeWin" );                                                                
    };    

	/**
	 * \ingroup  Win
     * Free a std::vector of RMA window handles
	 *
	 * \param[in] wins			A std::vector of windows to free
	 */
    inline void WinFree(std::vector<Win> &wins) {
        for (auto &w : wins) WinFree(w);
    };

	/**
	 * \ingroup  Win
     * Free a varadic set of RMA windows
	 *
	 * \param[in] d0			The first window to free
	 * \param[in] d1			The second window to free
	 * \param[in] args			The remaining windows to free
	 */
    template<typename T0, typename T1, typename ...Args>
    inline void WinFree(T0 &d0, T1 &d1, Args &&...args) {
        WinFree(d0);
        WinFree(d1, args...);
    };

	/// \cond HIDE
    struct Mutex {
        /// Members
        unsigned char *val; 
        bool locked;
        int rank, size, root;
        Comm        comm;
        Win            win;
    
        Mutex() : val(nullptr), locked(false), rank(0), size(0), root(0),
                      comm(MEL::Comm::COMM_NULL), win(MEL::Win::WIN_NULL) {};
    };
	/// \endcond

	/**
	 * \ingroup Mutex
	 * Create a MEL::Mutex across a comm world
	 *
	 * \param[in] rank		The rank of the calling process
	 * \param[in] size		The size of the comm world
	 * \param[in] root		The rank of the process who will own the mutex
	 * \param[in] comm		The comm world to share the mutex across
	 */
    inline Mutex MutexCreate(const int rank, const int size, const int root, const Comm &comm) {
        MEL::Barrier(comm);
        Mutex mutex;
        mutex.locked = false;
        mutex.rank = rank;
        mutex.size = size;
        mutex.root = root;
        mutex.comm = comm;

        if (rank == root) {
            /// Allocate and expose
            mutex.val = MEL::MemAlloc<unsigned char>(size);
            memset(mutex.val, 0, size);
            mutex.win = MEL::WinCreate(mutex.val, size, comm);
        }
        else {
            /// Indecent exposure
            mutex.val = nullptr;
            mutex.win = MEL::WinCreate(mutex.val, 0, comm);
        }

        return mutex;
    };

	/**
	 * \ingroup Mutex
	 * Create a MEL::Mutex across a comm world
	 *
	 * \param[in] root		The rank of the process who will own the mutex
	 * \param[in] comm		The comm world to share the mutex across
	 */
	inline Mutex MutexCreate(const int root, const Comm &comm) {
		return MutexCreate(CommRank(comm), CommSize(comm), root, comm);
	};

	/**
	 * \ingroup Mutex
	 * Free a MEL::Mutex
	 *
	 * \param[in] mutex		The mutex to free
	 */
    inline void MutexFree(Mutex &mutex) {
        MEL::Barrier(mutex.comm);
        MEL::WinFree(mutex.win);
        MEL::MemFree(mutex.val);
    };

    /**
	 * \ingroup Mutex
	 * Get the exclusive lock on a MEL::Mutex
	 *
	 * \param[in] mutex		The mutex to lock
	 */
    inline void MutexLock(Mutex &mutex) {
        if (mutex.locked) return;

        unsigned char *waitlist = MEL::MemAlloc<unsigned char>(mutex.size);

        /// Declare our intent to lock and sync waitlist
        unsigned char lock = 1;
        MEL::WinLockExclusive(mutex.win, mutex.root);
        MEL::Put(&lock,       1,           MEL::Datatype::UNSIGNED_CHAR, mutex.rank, 1,             MEL::Datatype::UNSIGNED_CHAR, mutex.root, mutex.win);
        MEL::Get(waitlist, mutex.size, MEL::Datatype::UNSIGNED_CHAR, 0,             mutex.size, MEL::Datatype::UNSIGNED_CHAR, mutex.root, mutex.win);
        MEL::WinUnlock(mutex.win, mutex.root);

        /// Is there a conflict?
        for (int i = 0; i < mutex.size; i++) {
            if (i != mutex.rank && waitlist[i] != 0) {
                /// If at least one conflict exists wait for a message to say everything is groovy
                MEL::Recv(&lock, 0, MEL::Datatype::UNSIGNED_CHAR, MEL::ANY_SOURCE, 99, mutex.comm);
                break;
            }
        }

        /// Clean up for if we got the lock without conflict
        MEL::MemFree(waitlist);

        /// We have the lock
        mutex.locked = true;
    };

    /**
	 * \ingroup Mutex
	 * Test if the mutex is currently locked
	 *
	 * \param[in] mutex		The mutex to lock
	 */
	inline bool MutexTest(const Mutex &mutex) {
        return mutex.locked;
    };

	/**
	 * \ingroup Mutex
	 * Release the exclusive lock on a MEL::Mutex
	 *
	 * \param[in] mutex		The mutex to lock
	 */
    inline void MutexUnlock(Mutex &mutex) {
        if (!mutex.locked) return;

        unsigned char *waitlist = MEL::MemAlloc<unsigned char>(mutex.size);
        mutex.locked = 0;

        /// Declare we are done and sync waitlist
        unsigned char lock = 0;
        MEL::WinLockExclusive(mutex.win, mutex.root);
        MEL::Put(&lock,       1,           MEL::Datatype::UNSIGNED_CHAR, mutex.rank, 1,             MEL::Datatype::UNSIGNED_CHAR, mutex.root, mutex.win);
        MEL::Get(waitlist, mutex.size, MEL::Datatype::UNSIGNED_CHAR, 0,             mutex.size, MEL::Datatype::UNSIGNED_CHAR, mutex.root, mutex.win);
        MEL::WinUnlock(mutex.win, mutex.root);

        /// Starting at a random process, scan for a intent to lock 
        const int r = rand() % mutex.size;
        for (int k = 0; k < mutex.size; k++) {
            const int i = (mutex.rank + k + r) % mutex.size;
            if (i != mutex.rank && waitlist[i] == 1) {
                /// If a process is waiting for the lock, send them a message
                MEL::Send(&lock, 0, MEL::Datatype::UNSIGNED_CHAR, i, 99, mutex.comm);
                break;
            }
        }

        /// Clean up for if no one else wanted the lock
        MEL::MemFree(waitlist);
    };

	/// \cond HIDE
    template<typename T>
    struct Shared {
        /// Members
        Mutex mutex;
        Win win;
        Datatype typeData;
        T *ptr;
        int len;

        Shared() : ptr(nullptr), len(0), mutex(), win(MEL::Win::WIN_NULL), typeData(MEL::Datatype::DATATYPE_NULL) {};
        
        inline bool is_locked() const {
            return MEL::MutexTest(mutex);
        };

        inline T& operator[](const int i) {
            if (!is_locked()) MEL::Abort(-1, "RMA::Shared<T>::operator[] We do not have the lock!");
            return ptr[i];
        };

        inline T* operator->() {
            if (!is_locked()) MEL::Abort(-1, "RMA::Shared<T>::operator-> We do not have the lock!");
            return ptr;
        };

        inline operator T*() {
            if (!is_locked()) MEL::Abort(-1, "RMA::Shared<T>::operator (T*) We do not have the lock!");
            return ptr;
        };

        inline T& operator*() {
            if (!is_locked()) MEL::Abort(-1, "RMA::Shared<T>::operator* We do not have the lock!");
            return *ptr;
        };
    };
	/// \endcond

	/**
	 * \ingroup Shared
	 * Create a MEL::Shared array across a comm world
	 *
	 * \param[in] len		The number of elements to allocate
	 * \param[in] rank		The rank of the calling process
	 * \param[in] size		The size of the comm world
	 * \param[in] root		The rank of the process who will own the shared array
	 * \param[in] comm		The comm world to share the array across
	 */
    template<typename T>
    inline Shared<T> SharedCreate(const int len, const int rank, const int size, const int root, const Comm &comm) {
        MEL::Barrier(comm);
        Shared<T> shared;
        shared.len = len;
        shared.mutex = MEL::MutexCreate(rank, size, root, comm);

        shared.ptr = MEL::MemAlloc<T>(len);
        memset(shared.ptr, 0, sizeof(T) * len);

        if (rank == root) {
            shared.win = MEL::WinCreate(shared.ptr, len, comm);
        }
        else {
            shared.win = MEL::WinCreate(shared.ptr, 0, comm);
        }

        shared.typeData = MEL::TypeCreateContiguous(MEL::Datatype::UNSIGNED_CHAR, sizeof(T));
        return shared;
    };

	/**
	 * \ingroup Shared
	 * Create a MEL::Shared array across a comm world
	 *
	 * \param[in] len		The number of elements to allocate
	 * \param[in] root		The rank of the process who will own the shared array
	 * \param[in] comm		The comm world to share the array across
	 */
    template<typename T>
    inline Shared<T> SharedCreate(const int len, const int root, const Comm &comm) {
        return SharedCreate<T>(len, CommRank(comm), CommSize(comm), root, comm);
    };

	/**
	 * \ingroup Shared
	 * Free a MEL::Shared array
	 *
	 * \param[in] shared	The shared array to free
	 */
    template<typename T>
    inline void SharedFree(Shared<T> &shared) {
        MEL::Barrier(shared.mutex.comm);
        MEL::WinFree(shared.win);
        MEL::MemFree(shared.ptr);
        MEL::MutexFree(shared.mutex);
        MEL::TypeFree(shared.typeData);
    };

	/**
	 * \ingroup Shared
	 * Test if the shared array is currently locked
	 *
	 * \param[in] shared	The shared array to test
	 */
    template<typename T>
    inline bool SharedTest(const Shared<T> &shared) {
        return shared.is_locked();
    };

	/**
	 * \ingroup Shared
	 * Get the lock on the shared array without synchronizing the data. Useful for if you only intend to write to the array
	 *
	 * \param[in] shared	The shared array to lock
	 */
    template<typename T>
    inline void SharedLock_noget(Shared<T> &shared) {
        SharedLock_noget(shared, 0, shared.len - 1);
    };
    
	/**
	 * \ingroup Shared
	 * Get the lock on the shared array without synchronizing the data. Useful for if you only intend to write to the array
	 *
	 * \param[in] shared	The shared array to lock
	 * \param[in] start		The start index to lock
	 * \param[in] end		The end index to lock
	 */
	template<typename T>
    inline void SharedLock_noget(Shared<T> &shared, const int start, const int end) {
        MEL::MutexLock(shared.mutex); // , start, end
    };

	/**
	 * \ingroup Shared
	 * Get the lock on the shared array and synchronize the data
	 *
	 * \param[in] shared	The shared array to lock
	 */
    template<typename T>
    inline void SharedLock(Shared<T> &shared) {
        SharedLock(shared, 0, shared.len - 1);
    };

	/**
	 * \ingroup Shared
	 * Get the lock on the shared array and synchronize the data
	 *
	 * \param[in] shared	The shared array to lock
	 * \param[in] start		The start index to lock
	 * \param[in] end		The end index to lock
	 */
    template<typename T>
    inline void SharedLock(Shared<T> &shared, const int start, const int end) {
        SharedLock_noget(shared, start, end);

        if (shared.mutex.rank != shared.mutex.root) {
            const int num = (end - start) + 1;
            MEL::WinLockExclusive(shared.win, shared.mutex.root);
            MEL::Get(shared.ptr + (start), num, shared.typeData, start, num, shared.typeData, shared.mutex.root, shared.win);
            MEL::WinUnlock(shared.win, shared.mutex.root);
        }
    };

	/**
	 * \ingroup Shared
	 * Release the lock on the shared array without synchronizing the data. Useful for if you only read from the array
	 * 
	 * \param[in] shared	The shared array to unlock
	 */
    template<typename T>
    inline void SharedUnlock_noput(Shared<T> &shared) {
        SharedUnlock_noput(shared, 0, shared.len - 1);
    };
    
	/**
	 * \ingroup Shared
	 * Release the lock on the shared array without synchronizing the data. Useful for if you only read from the array
	 * 
	 * \param[in] shared	The shared array to unlock
	 * \param[in] start		The start index to unlock
	 * \param[in] end		The end index to unlock
	 */
	template<typename T>
    inline void SharedUnlock_noput(Shared<T> &shared, const int start, const int end) {
        MEL::MutexUnlock(shared.mutex); // , start, end
    };

	/**
	 * \ingroup Shared
	 * Release the lock on the shared array and synchronize the data
	 * 
	 * \param[in] shared	The shared array to unlock
	 */
    template<typename T>
    inline void SharedUnlock(Shared<T> &shared) {
        SharedUnlock(shared, 0, shared.len - 1);
    };

	/**
	 * \ingroup Shared
	 * Release the lock on the shared array and synchronize the data
	 * 
	 * \param[in] shared	The shared array to unlock
	 * \param[in] start		The start index to unlock
	 * \param[in] end		The end index to unlock
	 */
    template<typename T>
    inline void SharedUnlock(Shared<T> &shared, const int start, const int end) {
        if (shared.mutex.rank != shared.mutex.root) {
            const int num = (end - start) + 1;
            MEL::WinLockExclusive(shared.win, shared.mutex.root);
            MEL::Put(shared.ptr + (start), num, shared.typeData, start, num, shared.typeData, shared.mutex.root, shared.win);
            MEL::WinUnlock(shared.win, shared.mutex.root);
        }

        SharedUnlock_noput(shared, start, end);
    };

};