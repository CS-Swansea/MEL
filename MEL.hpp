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

#if (MPI_VERSION == 3)
#define MEL_3
#endif
    
    /// Helper types to keep things in MEL namespace
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
	 * Calls MPI_Abort with the given error code and prints a string message to stderr
	 *
	 * @param ierr		The error code to throw
	 * @param message	The message to print to stderr describing what happened
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
	 * Tests if MPI_Init has been successfully called
	 * 
	 * @return Returns whether MPI is initialized as a bool
	 */
    inline bool IsInitialized() {
        int init;
        MEL_THROW( MPI_Initialized(&init), "Initialized" );
        return init != 0;
    };

	/**
	 * Tests if MPI_Finalize has been successfully called
	 * 
	 * @return Returns whether MPI is finalized as a bool
	 */
    inline bool IsFinalized() {
        int fin; 
        MEL_THROW( MPI_Finalized(&fin), "Finalized" );
        return fin != 0;
    };

	/**
	 * Call MPI_Init and setup default error handling
	 *
	 * @param argc		Forwarded from program main
	 * @param argv		Forwarded from program main
	 */
    inline void Init(int &argc, char **&argv) {
        if (!IsInitialized()) {
            MEL_THROW( MPI_Init(&argc, &argv), "Init" );
        }
        /// Allows MEL::Abort to be called properly
        MEL_THROW( MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "Initialize::SetErrorHandler" );
    };

	/**
	 * Call MPI_Finalize
	 */
    inline void Finalize() {
        if (!IsFinalized()) {
            MEL_THROW( MPI_Finalize(), "Finalize");
        }
    };

	/**
	 * MEL alternative to std::exit
	 *
	 * @param errcode	The error code to exit with
	 */
    inline void Exit(const int errcode) {
        MEL::Abort(errcode, "EXIT");
    };

	/**
	 * MEL alternative to std::exit
	 *
	 * @param errcode	The error code to exit with
	 * @param msg		A message to print to stderr as the program exits
	 */
    inline void Exit(const int errcode, const std::string &msg) {
        std::cerr << msg << std::endl;
        MEL::Abort(errcode, "EXIT");
    };

	/**
	 * Gets the current wall time since epoch in seconds
	 * 
	 * @return Returns the current wall time as a double
	 */
    inline double Wtime() {
        return MPI_Wtime();
    };

	/**
	 * Gets the current system tick
	 * 
	 * @return Returns the current system tick as a double
	 */
    inline double Wtick() {
        return MPI_Wtick();
    };

    /// ----------------------------------------------------------------------------------------------------------
    ///  Error Handler
    /// ----------------------------------------------------------------------------------------------------------

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

    inline int AddErrorClass() {
        int err;
        MEL_THROW( MPI_Add_error_class(&err), "ErrorHandler::AddErrorClass" );
        return err;
    };
    inline int AddErrorCode(const int errClass) {
        int err;
        MEL_THROW( MPI_Add_error_code(errClass, &err), "ErrorHandler::AddErrorCode" );
        return err;
    };
    inline int AddErrorCode() {
        return AddErrorCode(AddErrorClass());
    };
    inline void AddErrorString(const int err, const std::string &str) {
        MEL_THROW( MPI_Add_error_string(err, str.c_str()), "ErrorHandler::AddErrorString" );
    };
    inline int AddErrorString(const std::string &str) {
        const int err = AddErrorCode();
        AddErrorString(err, str);
        return err;
    };

    inline int GetErrorClass(const int errCode) {
        int err;
        MEL_THROW( MPI_Error_class(errCode, &err), "ErrorHandler::GetErrorClass" );
        return err;
    };
    inline std::string GetErrorString(const int errCode) {
        std::string str; str.resize(BUFSIZ); int len;
        MEL_THROW( MPI_Error_string(errCode, &str[0], &len), "ErrorHandler::GetErrorString" );
        str.resize(len);
        return str;
    };

    inline void ErrorHandlerFree(ErrorHandler &errHndl) {
        MEL_THROW( MPI_Errhandler_free((MPI_Errhandler*) &errHndl), "ErrorHandler::Free" );
        //errHndl = MEL::ErrorHandler::ERRHANDLER_NULL;
    };

    inline void ErrorHandlerFree(std::vector<ErrorHandler> &errHndls) {
        for (auto &e : errHndls) ErrorHandlerFree(e);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void ErrorHandlerFree(T0 &d0, T1 &d1, Args &&...args) {
        ErrorHandlerFree(d0);
        ErrorHandlerFree(d1, args...);
    };

    /// ----------------------------------------------------------------------------------------------------------
    ///  Memory Allocation
    /// ----------------------------------------------------------------------------------------------------------

    template<typename T>
    inline T* MemAlloc(const Aint size) {
        T *ptr;
        MEL_THROW( MPI_Alloc_mem(size * sizeof(T), MPI_INFO_NULL, &ptr), "Mem::Alloc" );
        return ptr;
    };

    template<typename T, typename ...Args>
    inline T* MemConstruct(Args &&...args) {
        T *ptr = MemAlloc<T>(1);
        new (ptr) T(args...);
        return ptr;
    };

    template<typename T>
    inline void MemFree(T *&ptr) {
        if (ptr != nullptr) {
            MEL_THROW( MPI_Free_mem(ptr), "Mem::Free" );
            ptr = nullptr;
        }
    };

    template<typename T0, typename T1, typename ...Args>
    inline void MemFree(T0 &d0, T1 &d1, Args &&...args) {
        MemFree(d0);
        MemFree(d1, args...);
    };

    template<typename T>
    inline void MemDestruct(T *&ptr, const Aint len = 1) {
        if (ptr == nullptr) return;
        for (Aint i = 0; i < len; ++i) {
            (&ptr[i])->~T();
        }
        MemFree(ptr);
    };

    /// ----------------------------------------------------------------------------------------------------------
    ///  Communicators & Requests
    /// ----------------------------------------------------------------------------------------------------------

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


    typedef MPI_Status    Status;
    typedef MPI_Info    Info;

    /// Error handling
    inline ErrorHandler CommCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Comm_create_errhandler((MPI_Comm_errhandler_function*) func, &errHndl), "Comm::CreateErrorHandler" );
        return ErrorHandler(errHndl);
    };
    inline void CommSetErrorHandler(const Comm &comm, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_Comm_set_errhandler((MPI_Comm) comm, (MPI_Errhandler) errHndl), "Comm::SetErrorHandler" );
    };
    inline void CommSetErrorHandler(const Comm &comm, ErrorHandlerFunc func) {
        CommSetErrorHandler(comm, CommCreateErrorHandler(func));
    };
    inline ErrorHandler CommGetErrorHandler(const Comm &comm) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Comm_get_errhandler((MPI_Comm) comm, &errHndl), "Comm::GetErrorHandler");
        return ErrorHandler(errHndl);
    };

    /// Who am i
    inline int CommRank(const Comm &comm) {
        int r; 
        MEL_THROW( MPI_Comm_rank((MPI_Comm) comm, &r), "Comm::Rank" );
        return r;
    };
    inline int CommSize(const Comm &comm) {
        int s; 
        MEL_THROW( MPI_Comm_size((MPI_Comm) comm, &s), "Comm::Size" );
        return s;
    };
    inline int CommRemoteSize(const Comm &comm) {
        int s; 
        MEL_THROW( MPI_Comm_remote_size((MPI_Comm) comm, &s), "Comm::RemoteSize" );
        return s;
    };

    /// Creation
    inline Comm CommSplit(const Comm &comm, int color) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_split((MPI_Comm) comm, color, CommRank(comm), &out_comm), "Comm::Split" );
        return Comm(out_comm);
    };
    inline Comm CommDuplicate(const Comm &comm) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_dup((MPI_Comm) comm, &out_comm), "Comm::Duplicate" );
        return Comm(out_comm);
    };
    
#ifdef MEL_3
    inline Comm CommIduplicate(const Comm &comm, Request &rq) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_idup((MPI_Comm) comm, &out_comm, (MPI_Request*) &rq), "Comm::Iduplicate" );
        return Comm(out_comm);
    };
    inline std::pair<Comm, Request> CommIduplicate(const Comm &comm) {
        Request rq;
        Comm out_comm = CommIduplicate(comm, rq);
        return std::make_pair(out_comm, rq);
    };
#endif
    
    inline Group CommGetGroup(const Comm &comm) {
        MPI_Group group;
        MEL_THROW( MPI_Comm_group((MPI_Comm) comm, &group), "Comm::GetGroup" );
        return Group(group);
    };
    inline Comm CommCreateFromGroup(const Comm &comm, const Group &group) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_create((MPI_Comm) comm, (MPI_Group) group, &out_comm), "Comm::CreateFromGroup" );
        return Comm(out_comm);
    };

#ifdef MEL_3
    inline Comm CommCreateFromGroup(const Comm &comm, const Group &group, const int tag) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Comm_create_group((MPI_Comm) comm, (MPI_Group) group, tag, &out_comm), "Comm::CreateFromGroup" );
        return Comm(out_comm);
    };
#endif

    /// Deletion
    inline void CommFree(Comm &comm) {
        MEL_THROW( MPI_Comm_disconnect((MPI_Comm*) &comm), "Comm::Free" );
        comm = Comm::COMM_NULL;
    };

    inline void CommFree(std::vector<Comm> &comms) {
        for (auto &e : comms) CommFree(e);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void CommFree(T0 &d0, T1 &d1, Args &&...args) {
        CommFree(d0);
        CommFree(d1, args...);
    };

    /// Testing
    inline bool CommIsNULL(const Comm &comm) {
        return (MPI_Comm) comm == MPI_COMM_NULL;
    };

    /// Synchronization
    inline void Barrier(const Comm &comm) {
        MEL_THROW( MPI_Barrier((MPI_Comm) comm), "Comm::Barrier" );
    };

#ifdef MEL_3
    inline void Ibarrier(const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Ibarrier((MPI_Comm) comm, (MPI_Request*) &rq), "Comm::IBarrier" );
    };
    inline Request Ibarrier(const Comm &comm) {
        Request rq{};
        Ibarrier(comm, rq);
        return rq;
    };
#endif
    
    inline void Wait(Request &rq) {
        MEL_THROW( MPI_Wait((MPI_Request*) &rq, MPI_STATUS_IGNORE), "Comm::Wait" );
    };
    inline bool Test(Request &rq) {
        int f;
        MEL_THROW( MPI_Test((MPI_Request*) &rq, &f, MPI_STATUS_IGNORE), "Comm::Test" );
        return f != 0;
    };

    /// All wait
    inline void Waitall(Request *ptr, int num) {
        MEL_THROW( MPI_Waitall(num, (MPI_Request*) ptr, MPI_STATUS_IGNORE), "Comm::Waitall" );
    };
    inline void Waitall(std::vector<Request> &rqs) {
        Waitall(&rqs[0], rqs.size());
    };

    /// All test
    inline bool Testall(Request *ptr, int num) {
        int f;
        MEL_THROW( MPI_Testall(num, (MPI_Request*) ptr, &f, MPI_STATUS_IGNORE), "Comm::Testall" );
        return f != 0;
    };
    inline bool Testall(std::vector<Request> &rqs) {
        return Testall(&rqs[0], rqs.size());
    };

    /// Any wait
    inline int Waitany(Request *ptr, int num) {
        int idx;
        MEL_THROW( MPI_Waitany(num, (MPI_Request*) ptr, &idx, MPI_STATUS_IGNORE), "Comm::Waitany" );
        return idx;
    };
    inline int Waitany(std::vector<Request> &rqs) {
        return Waitany(&rqs[0], rqs.size());
    };

    /// Any test
    inline std::pair<bool, int> Testany(Request *ptr, int num) {
        int idx, f;
        MEL_THROW( MPI_Testany(num, (MPI_Request*) ptr, &idx, &f, MPI_STATUS_IGNORE), "Comm::Testany" );
        return std::make_pair(f != 0, idx);
    };
    inline std::pair<bool, int> Testany(std::vector<Request> &rqs) {
        return Testany(&rqs[0], rqs.size());
    };

    /// Some wait
    inline std::vector<int> Waitsome(Request *ptr, int num) {
        std::vector<int> idx(num); int onum;
        MEL_THROW( MPI_Waitsome(num, (MPI_Request*) ptr, &onum, &idx[0], MPI_STATUS_IGNORE), "Comm::Waitsome" );
        idx.resize(onum);
        return idx;
    };
    inline std::vector<int> Waitsome(std::vector<Request> &rqs) {
        return Waitsome(&rqs[0], rqs.size());
    };

    /// Some test
    inline std::vector<int> Testsome(Request *ptr, int num) {
        std::vector<int> idx(num); int onum;
        MEL_THROW( MPI_Testsome(num, (MPI_Request*) ptr, &onum, &idx[0], MPI_STATUS_IGNORE), "Comm::Testsome" );
        idx.resize(onum);
        return idx;
    };
    inline std::vector<int> Testsome(std::vector<Request> &rqs) {
        return Testsome(&rqs[0], rqs.size());
    };

    /// Set ops
    inline Group GroupUnion(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_union((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Union" );
        return Group(out_group);
    };
    inline Group GroupDifference(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_difference((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Difference" );
        return Group(out_group);
    };
    inline Group GroupIntersection(const Group& lhs, const Group& rhs) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_intersection((MPI_Group) lhs, (MPI_Group) rhs, &out_group), "Group::Intersection" );
        return Group(out_group);
    };

    /// Include Ranks
    inline Group GroupInclude(const Group& group, const int *ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_incl((MPI_Group) group, num, ranks, &out_group), "Group::Include" );
        return Group(out_group);
    };
    inline Group GroupInclude(const Group& group, const std::vector<int> &ranks) {
        return GroupInclude(group, &ranks[0], ranks.size());
    };
    inline Group GroupIncludeRange(const Group& group, const int **ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_range_incl((MPI_Group) group, num, (int(*)[3]) ranks, &out_group), "Group::IncludeRange" );
        return Group(out_group);
    };
    inline Group GroupIncludeRange(const Group& group, const std::vector<int[3]> &ranks) {
        return GroupIncludeRange(group, (const int**) &ranks[0], (int) ranks.size());
    };

    /// Exclude Ranks
    inline Group GroupExclude(const Group& group, const int *ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_excl((MPI_Group) group, num, ranks, &out_group), "Group::Exclude" );
        return Group(out_group);
    };
    inline Group GroupExclude(const Group& group, const std::vector<int> &ranks) {
        return GroupExclude(group, &ranks[0], ranks.size());
    };
    inline Group GroupExcludeRange(const Group& group, const int **ranks, const int num) {
        MPI_Group out_group;
        MEL_THROW( MPI_Group_range_excl((MPI_Group) group, num, (int(*)[3]) ranks, &out_group), "Group::ExcludeRange" );
        return Group(out_group);
    };
    inline Group GroupExcludeRange(const Group& group, const std::vector<int[3]> &ranks) {
        return GroupExcludeRange(group, (const int**) &ranks[0], (int) ranks.size());
    };
        
    /// Comparisons
    inline int GroupCompare(const Group& lhs, const Group& rhs) {
        int r; 
        MEL_THROW( MPI_Group_compare((MPI_Group) lhs, (MPI_Group) rhs, &r), "Group::Compare" );
        return r;
    };
    inline bool GroupIsSimilar(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_SIMILAR;
    };
    inline bool GroupIsIdentical(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_IDENT;
    };
    inline bool GroupIsCongruent(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_CONGRUENT;
    };
    inline bool GroupIsUnequal(const Group& lhs, const Group& rhs) {
        return GroupCompare(lhs, rhs) == MPI_UNEQUAL;
    };
    inline bool GroupIsNULL(const Group &group) {
        return (MPI_Group) group == MPI_GROUP_NULL;
    };

    /// Who am i
    inline int GroupRank(const Group &group) {
        int r; 
        MEL_THROW( MPI_Group_rank((MPI_Group) group, &r), "Group::Rank" );
        return r;
    };
    inline int GroupSize(const Group &group) {
        int s; 
        MEL_THROW( MPI_Group_size((MPI_Group) group, &s), "Group::Size" );
        return s;
    };

    /// Deletion
    inline void GroupFree(Group &group) {
        MEL_THROW( MPI_Group_free((MPI_Group*) &group), "Group::Free" );
        //group = MPI_GROUP_NULL; // Done automatically
    };

    inline void GroupFree(std::vector<Group> &groups) {
        for (auto &e : groups) GroupFree(e);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void GroupFree(T0 &d0, T1 &d1, Args &&...args) {
        GroupFree(d0);
        GroupFree(d1, args...);
    };


    /// ----------------------------------------------------------------------------------------------------------
    ///  Datatypes
    /// ----------------------------------------------------------------------------------------------------------

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
#endif

                                AINT,
                                OFFSET,
                                COUNT;

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
#endif

    const Datatype Datatype::AINT                = Datatype(MPI_AINT);
    const Datatype Datatype::OFFSET              = Datatype(MPI_OFFSET);

#ifdef MEL_3
    const Datatype Datatype::COUNT               = Datatype(MPI_COUNT);
#endif

#endif


    inline Datatype TypeCreateContiguous(const Datatype &datatype, const int length) {
        Datatype dt;
        MEL_THROW( MPI_Type_contiguous(length, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeContiguous" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeContiguous)" );
        return dt;
    };

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

    inline Datatype TypeCreateSubArray(const Datatype &datatype, const std::vector<TypeSubArray_Dim> &dims) {
        const int numDims = dims.size();
        std::vector<int>    starts(numDims);
        std::vector<int>    subSizes(numDims);
        std::vector<int>    sizes(numDims);

        for (int i = 0; i < numDims; ++i) {
            starts[i]   = dims[i].start;
            subSizes[i] = dims[i].size;
            sizes[i]    = dims[i].extent;
        }
        return TypeCreateSubArray(datatype, numDims, &sizes[0], &subSizes[0], &starts[0]);
    };

    inline Datatype TypeCreateSubArray1D(const Datatype &datatype, const int x, const int sx, const int dx) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_subarray(1, &dx, &sx, &x, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeSubArray1D" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeSubArray1D)" );
        return dt;
    };

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

    inline Datatype TypeCreateIndexedBlock(const Datatype &datatype, const int num, const int length, const int *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_indexed_block(num, length, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeIndexedBlock" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeIndexedBlock)" );
        return dt;
    };

    inline Datatype TypeCreateIndexedBlock(const Datatype &datatype, const int length, const std::vector<int> &displs) {
        return TypeCreateIndexedBlock(datatype, displs.size(), length, &displs[0]);
    };

#ifdef MEL_3
    inline Datatype TypeCreateHIndexedBlock(const Datatype &datatype, const int num, const int length, const Aint *displs) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_hindexed_block(num, length, displs, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeHIndexedBlock" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeHIndexedBlock)" );
        return dt;
    };
    inline Datatype TypeCreateHIndexedBlock(const Datatype &datatype, const int length, const std::vector<Aint> &displs) {
        return TypeCreateHIndexedBlock(datatype, displs.size(), length, &displs[0]);
    };
#endif

    inline Datatype TypeCreateVector(const Datatype &datatype, const int num, const int length, const int stride) {
        Datatype dt;
        MEL_THROW( MPI_Type_vector(num, length, stride, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeVector" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeVector)" );
        return dt;
    };

    inline Datatype TypeCreateHVector(const Datatype &datatype, const int num, const int length, const Aint stride) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_hvector(num, length, stride, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeHVector" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeHVector)" );
        return dt;
    };

    enum class Distribute : int {
        NONE      = MPI_DISTRIBUTE_NONE, 
        BLOCK     = MPI_DISTRIBUTE_BLOCK,
        CYCLIC    = MPI_DISTRIBUTE_CYCLIC,
    };

    enum {
        DFLT_DARG = MPI_DISTRIBUTE_DFLT_DARG
    };

    inline Datatype TypeCreateDArray(const Datatype &datatype, const int rank, const int size, const int numDims, const int *gsizes, const Distribute *distribs, const int *dargs, const int *psizes) {
        Datatype dt;
        MEL_THROW( MPI_Type_create_darray(size, rank, numDims, gsizes, (int*) distribs, dargs, psizes, MPI_ORDER_C, (MPI_Datatype) datatype, (MPI_Datatype*) &dt), "Datatype::TypeDArray" );
        MEL_THROW( MPI_Type_commit((MPI_Datatype*) &dt), "Datatype::TypeCommit(TypeDArray)" );
        return dt;
    };

    inline Datatype TypeCreateDArray(const Datatype &datatype, const Comm &comm, const int numDims, const int *gsizes, const Distribute *distribs, const int *dargs, const int *psizes) {
        return TypeCreateDArray(datatype, CommRank(comm), CommSize(comm), numDims, gsizes, distribs, dargs, psizes);
    };

    struct TypeDArray_Dim {
        int gsize, darg, psize;
        Distribute distrib;

        TypeDArray_Dim() : gsize(0), distrib(MEL::Distribute::NONE), darg(0), psize(0) {};
        TypeDArray_Dim(const int _gsize, const Distribute _distrib, const int _darg, const int _psize)
                        : gsize(_gsize), distrib(_distrib), darg(_darg), psize(_psize) {};
    };

    inline Datatype TypeCreateDArray(const Datatype &datatype, const int rank, const int size, const std::vector<TypeDArray_Dim> &dims) {
        const int numDims = dims.size();
        std::vector<int>        gsizes(numDims);
        std::vector<Distribute> distribs(numDims);
        std::vector<int>        dargs(numDims);
        std::vector<int>        psizes(numDims);

        for (int i = 0; i < numDims; ++i) {
            gsizes[i]   = dims[i].gsize;
            distribs[i] = dims[i].distrib;
            dargs[i]    = dims[i].darg;
            psizes[i]   = dims[i].psize;
        }
        return TypeCreateDArray(datatype, rank, size, numDims, &gsizes[0], &distribs[0], &dargs[0], &psizes[0]);
    };

    inline Datatype TypeCreateDArray(const Datatype &datatype, const Comm &comm, const std::vector<TypeDArray_Dim> &dims) {
        return TypeCreateDArray(datatype, CommRank(comm), CommSize(comm), dims);
    };

    inline Datatype TypeDuplicate(const Datatype &datatype) {
        MPI_Datatype out_datatype;
        MEL_THROW( MPI_Type_dup((MPI_Datatype) datatype, &out_datatype), "Datatype::Duplicate" );
        return Datatype(out_datatype);
    };

    inline int TypeSize(const Datatype &datatype) {
        int out_size;
        MEL_THROW( MPI_Type_size((MPI_Datatype) datatype, &out_size), "Datatype::Size" );
        return out_size;
    };

    inline std::pair<Aint, Aint> TypeExtent(const Datatype &datatype) {
        Aint out_lb, out_ext;
        MEL_THROW( MPI_Type_get_extent((MPI_Datatype) datatype, &out_lb, &out_ext), "Datatype::Extent" );
        return std::make_pair(out_lb, out_ext);
    };

    inline Aint TypeGetExtent(const Datatype &datatype) {
        Aint out_lb, out_ext;
        MEL_THROW( MPI_Type_get_extent((MPI_Datatype) datatype, &out_lb, &out_ext), "Datatype::GetExtent" );
        return out_ext;
    };

    inline void TypeFree(Datatype &datatype) {
        if (datatype != MEL::Datatype::DATATYPE_NULL) {
            MEL_THROW( MPI_Type_free((MPI_Datatype*) &datatype), "Datatype::Free" );
            datatype = Datatype::DATATYPE_NULL;
        }
    };

    inline void TypeFree(std::vector<Datatype> &datatypes) {
        for (auto &d : datatypes) TypeFree(d);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void TypeFree(T0 &d0, T1 &d1, Args &&...args) {
        TypeFree(d0);
        TypeFree(d1, args...);
    };

    /// ----------------------------------------------------------------------------------------------------------
    ///  Topology
    /// ----------------------------------------------------------------------------------------------------------

    inline void TopoCartesianMakeDims(const int numProcs, const int numDims, int *dims) {
        MEL_THROW( MPI_Dims_create(numProcs, numDims, dims), "Topo::Cartesian::MakeDims" );
    };

    inline void TopoCartesianMakeDims(const Comm &comm, const int numDims, int *dims) {
        TopoCartesianMakeDims(CommSize(comm), numDims, dims);
    };

    inline std::vector<int> TopoCartesianMakeDims(const int numProcs, const int numDims) {
        std::vector<int> dims(numDims);
        TopoCartesianMakeDims(numProcs, numDims, &dims[0]);
        return dims;
    };

    inline std::vector<int> TopoCartesianMakeDims(const Comm &comm, const int numDims) {
        return TopoCartesianMakeDims(CommSize(comm), numDims);
    };

    inline Comm TopoCartesianCreate(const Comm &comm, int maxdims, const int *dims, const int *periods) {
        MPI_Comm out_comm;
        MEL_THROW( MPI_Cart_create((MPI_Comm) comm, maxdims, dims, periods, 0, &out_comm), "Topo::Cartesian::Create");
        return Comm(out_comm);
    };

    struct TopoCartesian_Dim {
        int size, periodic;

        TopoCartesian_Dim() {};
        TopoCartesian_Dim(const int _size, const bool _p) : size(_size), periodic(_p ? 1 : 0) {};
        TopoCartesian_Dim(const int _size, const int _p) : size(_size), periodic(_p) {};
    };

    inline Comm TopoCartesianCreate(const Comm &comm, const std::vector<TopoCartesian_Dim> &dims) {
        const int numDims = dims.size();
        std::vector<int>    sizes(numDims);
        std::vector<int>    periods(numDims);

        for (int i = 0; i < numDims; ++i) {
            sizes[i]    = dims[i].size;
            periods[i]  = dims[i].periodic;
        }
        return TopoCartesianCreate(comm, numDims, &sizes[0], &periods[0]);
    };

    inline int TopoCartesianNumDims(const Comm &comm) {
        int dim;
        MEL_THROW( MPI_Cartdim_get((MPI_Comm) comm, &dim), "Topo::Cartesian::NumDims");
        return dim;
    };

    inline int TopoCartesianRank(const Comm &comm, const int *coords) {
        int rank;
        MEL_THROW( MPI_Cart_rank((MPI_Comm) comm, coords, &rank), "Topo::Cartesian::Rank");
        return rank;
    };

    inline int TopoCartesianRank(const Comm &comm, const std::vector<int> coords) {
        return TopoCartesianRank(comm, &coords[0]);
    };

    inline void TopoCartesianCoords(const Comm &comm, const int rank, int maxdims, int *coords) {
        MEL_THROW( MPI_Cart_coords((MPI_Comm) comm, rank, maxdims, coords), "Topo::Cartesian::Coords");
    };

    inline std::vector<int> TopoCartesianCoords(const Comm &comm, const int rank, int maxdims) {
        std::vector<int> coords(maxdims);
        TopoCartesianCoords(comm, rank, maxdims, &coords[0]);
        return coords;
    };

    inline std::vector<int> TopoCartesianCoords(const Comm &comm, const int rank) {
        return TopoCartesianCoords(comm, rank, TopoCartesianNumDims(comm));
    };

    inline std::vector<int> TopoCartesianCoords(const Comm &comm) {
        return TopoCartesianCoords(comm, CommRank(comm), TopoCartesianNumDims(comm));
    };

    inline void TopoCartesianGet(const Comm &comm, int maxdims, int *dims, int *periods, int *coords) {
        MEL_THROW( MPI_Cart_get((MPI_Comm) comm, maxdims, dims, periods, coords), "Topo::Cartesian::Get");
    };

    inline void TopoCartesianGet(const Comm &comm, int *dims, int *periods, int *coords) {
        TopoCartesianGet(comm, TopoCartesianNumDims(comm), dims, periods, coords);
    };

    inline std::pair<std::vector<int>, std::vector<TopoCartesian_Dim>> TopoCartesianGet(const Comm &comm) {
        const int numDims = TopoCartesianNumDims(comm);
        std::vector<int> coords(numDims), dims(numDims), periods(numDims);
        TopoCartesianGet(comm, numDims, &dims[0], &periods[0], &coords[0]);

        std::vector<TopoCartesian_Dim> r;
        for (int i = 0; i < numDims; ++i) {
            r[i].size        = dims[i];
            r[i].periodic    = periods[i];
        }
        return std::make_pair(coords, r);
    };

    inline int TopoCartesianMap(const Comm &comm, int maxdims, int *dims, int *periods) {
        int rank;
        MEL_THROW( MPI_Cart_map((MPI_Comm) comm, maxdims, dims, periods, &rank), "Topo::Cartesian::Map");
        return rank;
    };

    inline void TopoCartesianShift(const Comm &comm, int direction, int disp, int &rank_prev, int &rank_next) {
        MEL_THROW( MPI_Cart_shift((MPI_Comm) comm, direction, disp, &rank_prev, &rank_next), "Topo::Cartesian::Shift");
    };

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

    /// ----------------------------------------------------------------------------------------------------------
    ///  Ops
    /// ----------------------------------------------------------------------------------------------------------

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
        template<typename T>
        T MAX(T &a, T &b) {
            return (a > b) ? a : b;
        };
        template<typename T>
        T MIN(T &a, T &b) {
            return (a < b) ? a : b;
        };
        template<typename T>
        T SUM(T &a, T &b) {
            return (a + b);
        };
        template<typename T>
        T PROD(T &a, T &b) {
            return (a * b);
        };
        template<typename T>
        T LAND(T &a, T &b) {
            return (a && b);
        };
        template<typename T>
        T BAND(T &a, T &b) {
            return (a & b);
        };
        template<typename T>
        T LOR(T &a, T &b) {
            return (a || b);
        };
        template<typename T>
        T BOR(T &a, T &b) {
            return (a | b);
        };
        template<typename T>
        T LXOR(T &a, T &b) {
            return (!a != !b);
        };
        template<typename T>
        T BXOR(T &a, T &b) {
            return (a ^ b);
        };

        /// MEL-style mapped functor
        template<typename T, T(*F)(T&, T&)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            for (int i = 0; i < *len; ++i) inout[i] = F(in[i], inout[i]);
        };
        template<typename T, T(*F)(T&, T&, Datatype)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            Datatype dt((Datatype)*dptr);
            for (int i = 0; i < *len; ++i) inout[i] = F(in[i], inout[i], dt);
        };

        /// MEL-style buffers & length
        template<typename T, void(*F)(T*, T*, int)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            F(in, inout, *len);
        };

        template<typename T, void(*F)(T*, T*, int, Datatype)>
        void ARRAY_OP_FUNC(T *in, T *inout, int *len, MPI_Datatype *dptr) {
            F(in, inout, *len, (Datatype) *dptr);
        };
    };

    /// MEL-style mapped functor
    template<typename T, T(*F)(T&, T&)>
    inline Op CreateOp(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };
    template<typename T, T(*F)(T&, T&, Datatype)>
    inline Op CreateOp(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

    /// MEL-style buffers & length
    template<typename T, void(*F)(T*, T*, int)>
    inline Op CreateOp(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };
    template<typename T, void(*F)(T*, T*, int, Datatype)>
    inline Op CreateOp(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) Functor::ARRAY_OP_FUNC<T, F>, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

    /// C-style MPI everything is a pointer
    template<typename T, void(*F)(T*, T*, int*, MPI_Datatype*)>
    inline Op CreateOp(bool commute = true) {
        MPI_Op op;
        MEL_THROW( MPI_Op_create((void(*)(void*, void*, int*, MPI_Datatype*)) F, commute, (MPI_Op*) &op), "Op::CreatOp" );
        return Op(op);
    };

    inline void OpFree(Op &op) {
        MEL_THROW( MPI_Op_free((MPI_Op*) &op ), "Op::Free" );
    };

    inline void OpFree(std::vector<Op> &ops) {
        for (auto &e : ops) OpFree(e);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void OpFree(T0 &d0, T1 &d1, Args &&...args) {
        OpFree(d0);
        OpFree(d1, args...);
    };


    /// ----------------------------------------------------------------------------------------------------------
    ///  File IO
    /// ----------------------------------------------------------------------------------------------------------

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
    inline FileMode operator|(const FileMode &a, const FileMode &b) {
        return static_cast<FileMode>(static_cast<int>(a) | static_cast<int>(b));
    };

    enum class SeekMode : int {
        SET                = MPI_SEEK_SET,
        CUR                = MPI_SEEK_CUR,
        END                = MPI_SEEK_END
    };

    inline ErrorHandler FileCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_File_create_errhandler((MPI_File_errhandler_function*) func, &errHndl), "File::CreateErrorHandler" );
        return ErrorHandler(errHndl);
    };
    inline void FileSetErrorHandler(const File &file, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_File_set_errhandler(file, (MPI_Errhandler) errHndl), "File::SetErrorHandler" );
    };
    inline void FileSetErrorHandler(const File &file, ErrorHandlerFunc func) {
        FileSetErrorHandler(file, FileCreateErrorHandler(func));
    };
    inline ErrorHandler FileGetErrorHandler(const File &file) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_File_get_errhandler(file, &errHndl), "File::GetErrorHandler");
        return ErrorHandler(errHndl);
    };

    inline FileMode FileGetMode(const File &file) {
        int amode;
        MEL_THROW( MPI_File_get_amode(file, &amode), "File::GetMode");
        return FileMode(amode);
    };
    inline bool FileIsAtomic(const File &file) {
        int flag;
        MEL_THROW( MPI_File_get_atomicity(file, &flag), "File::GetAtomicity");
        return flag != 0;
    };
    inline void FileSetAtomicity(const File &file, const bool atom) {
        MEL_THROW( MPI_File_set_atomicity(file, atom ? 1 : 0), "File::SetAtomicity");
    };
    inline Offset FileGetByteOffset(const File &file, const Offset offset) {
        Offset byteOffset;
        MEL_THROW( MPI_File_get_byte_offset(file, offset, &byteOffset), "File::GetByteOffset" );
        return byteOffset;
    };
    inline Group FileGetGroup(const File &file) {
        MPI_Group group;
        MEL_THROW( MPI_File_get_group(file, &group), "File::GetGroup");
        return Group(group);
    };
    inline Info FileGetInfo(const File &file) {
        MPI_Info info;
        MEL_THROW( MPI_File_get_info(file, &info), "File::GetInfo");
        return info;
    };
    inline void FileSetInfo(const File &file, const Info &info) {
        MEL_THROW( MPI_File_set_info(file, info), "File::SetInfo");
    };
    inline Offset FileGetPosition(const File &file) {
        Offset offset;
        MEL_THROW( MPI_File_get_position(file, &offset), "File::GetPosition" );
        return offset;
    };
    inline Offset FileGetPositionShared(const File &file) {
        Offset offset;
        MEL_THROW( MPI_File_get_position_shared(file, &offset), "File::GetPositionShared" );
        return offset;
    };
    inline Offset FileGetSize(const File &file) {
        Offset size;
        MEL_THROW( MPI_File_get_size(file, &size), "File::GetSize" );
        return size;
    };
    inline void FileSetSize(const File &file, const Offset size) {
        MEL_THROW( MPI_File_set_size(file, size), "File::SetSize" );
    };
    inline Aint FileGetTypeExtent(const File &file, const Datatype &datatype) {
        Aint size;
        MEL_THROW( MPI_File_get_type_extent(file, (MPI_Datatype) datatype, &size), "File::GetTypeExtent" );
        return size;
    };

    inline File FileOpen(const Comm &comm, const std::string &path, const FileMode amode) {
        MPI_File file;
        MEL_THROW( MPI_File_open((MPI_Comm) comm, path.c_str(), (int) amode, MPI_INFO_NULL, &file), "File::Open");
        MEL_THROW( MPI_File_set_errhandler(file, MPI_ERRORS_RETURN), "File::Open(SetErrorHandler)" );
        return file;
    };

    inline File FileOpenIndividual(const std::string &path, const FileMode amode) {
        return FileOpen(MEL::Comm::SELF, path, amode);
    };

    inline void FileDelete(const std::string &path) {
        MEL_THROW( MPI_File_delete(path.c_str(), MPI_INFO_NULL), "File::Delete");
    };
    inline void FileClose(File &file) {
        MEL_THROW( MPI_File_close(&file), "File::Close");
    };

    inline void FilePreallocate(const File &file, const Offset fileSize) {
        MEL_THROW( MPI_File_preallocate(file, fileSize), "File::Preallocate" );
    };

    inline void FileSeek(const File &file, const Offset offset, const SeekMode seekMode = MEL::SeekMode::SET) {
        MEL_THROW( MPI_File_seek(file, offset, (int) seekMode), "File::Seek" );
    };
    inline void FileSeekShared(const File &file, const Offset offset, const SeekMode seekMode = MEL::SeekMode::SET) {
        MEL_THROW( MPI_File_seek_shared(file, offset, (int) seekMode), "File::SeekShared" );
    };

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

    inline void FileSetView(const File &file, const Offset offset, const Datatype elementaryType, const Datatype fileType, const std::string &datarep = "native") {
        MEL_THROW( MPI_File_set_view(file, offset, (MPI_Datatype) elementaryType, (MPI_Datatype) fileType, datarep.c_str(), MPI_INFO_NULL), "File::SetView" );    
    };
    inline void FileSetView(const File &file, const FileView &view) {
        FileSetView(file, view.offset, view.elementaryType, view.fileType, view.datarep);
    };

    inline void FileGetView(const File &file, Offset &offset, Datatype &elementaryType, Datatype &fileType, std::string &datarep) {
        datarep.resize(BUFSIZ);
        MEL_THROW( MPI_File_get_view(file, &offset, (MPI_Datatype*) &elementaryType, (MPI_Datatype*) &fileType, (char*) &datarep[0]), "File::GetView" ); 
    };
    inline FileView FileGetView(const File &file) {
        FileView view;
        FileGetView(file, view.offset, view.elementaryType, view.fileType, view.datarep);
        return view;
    };

    inline Status FileWrite(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::Write" );
        return status;
    };
    inline Status FileWriteAll(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_all(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAll" );
        return status;
    };
    inline Status FileWriteAt(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_at(file, offset, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAt" );
        return status;
    };
    inline Status FileWriteAtAll(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_at_all(file, offset, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteAtAll" );
        return status;
    };
    inline Status FileWriteOrdered(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_ordered(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteOrdered" );
        return status;
    };
    inline Status FileWriteShared(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_write_shared(file, sptr, snum, (MPI_Datatype) datatype, &status), "File::WriteShared" );
        return status;
    };
    inline Request FileIwrite(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iwrite(file, sptr, snum, (MPI_Datatype) datatype, &request), "File::Iwrite" );
        return Request(request);
    };
    inline Request FileIwriteAt(const File &file, const Offset offset, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW(MPI_File_iwrite_at(file, offset, sptr, snum, (MPI_Datatype) datatype, &request), "File::IwriteAt");
        return Request(request);
    };
    inline Request FileIwriteShared(const File &file, const void *sptr, const int snum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iwrite_shared(file, sptr, snum, (MPI_Datatype) datatype, &request), "File::IwriteShared" );
        return Request(request);
    };

    inline Status FileRead(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::Read" );
        return status;
    };
    inline Status FileReadAll(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_all(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAll" );
        return status;
    };
    inline Status FileReadAt(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_at(file, offset, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAt" );
        return status;
    };
    inline Status FileReadAtAll(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_at_all(file, offset, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadAtAll" );
        return status;
    };
    inline Status FileReadOrdered(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_ordered(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadOrdered" );
        return status;
    };
    inline Status FileReadShared(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Status status;
        MEL_THROW( MPI_File_read_shared(file, rptr, rnum, (MPI_Datatype) datatype, &status), "File::ReadShared" );
        return status;
    };
    inline Request FileIread(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iread(file, rptr, rnum, (MPI_Datatype) datatype, &request), "File::Iread" );
        return Request(request);
    };
    inline Request FileIreadAt(const File &file, const Offset offset, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW(MPI_File_iread_at(file, offset, rptr, rnum, (MPI_Datatype) datatype, &request), "File::IreadAt");
        return Request(request);
    };
    inline Request FileIreadShared(const File &file, void *rptr, const int rnum, const Datatype &datatype) {
        MPI_Request request;
        MEL_THROW( MPI_File_iread_shared(file, rptr, rnum, (MPI_Datatype) datatype, &request), "File::IreadShared" );
        return Request(request);
    };    

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
    

    template<typename T>
    inline Status FileWrite(const File &file, const T *sptr, const int snum) {
        return FileWrite(file, sptr, snum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
    };

    template<typename T>
    inline Status FileRead(const File &file, T *rptr, const int rnum) {
        return FileRead(file, rptr, rnum * sizeof(T), MEL::Datatype::UNSIGNED_CHAR);
    };

    MEL_FILE(wchar_t,                        MPI_WCHAR);

    MEL_FILE(float,                            MPI_FLOAT);
    MEL_FILE(double,                        MPI_DOUBLE);
    MEL_FILE(long double,                    MPI_LONG_DOUBLE);

    MEL_FILE(int8_t,                        MPI_INT8_T);
    MEL_FILE(int16_t,                        MPI_INT16_T);
    MEL_FILE(int32_t,                        MPI_INT32_T);
    MEL_FILE(int64_t,                        MPI_INT64_T);

    MEL_FILE(uint8_t,                        MPI_UINT8_T);
    MEL_FILE(uint16_t,                        MPI_UINT16_T);
    MEL_FILE(uint32_t,                        MPI_UINT32_T);
    MEL_FILE(uint64_t,                        MPI_UINT64_T);

#ifdef MEL_3
    MEL_FILE(std::complex<float>,            MPI_CXX_FLOAT_COMPLEX);
    MEL_FILE(std::complex<double>,            MPI_CXX_DOUBLE_COMPLEX);
    MEL_FILE(std::complex<long double>,        MPI_CXX_LONG_DOUBLE_COMPLEX);
    MEL_FILE(bool,                            MPI_CXX_BOOL);
#endif

#undef MEL_FILE

    /// ----------------------------------------------------------------------------------------------------------
    ///  Point to Point - SEND
    /// ----------------------------------------------------------------------------------------------------------

    inline void Send(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                
        MEL_THROW( MPI_Send(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Send" );                                            
    }                                                                                                                                
    inline void Bsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Bsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Bsend" );                                        
    }                                                                                                                                
    inline void Ssend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Ssend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Ssend" );                                        
    }                                                                                                                                
    inline void Rsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                                
        MEL_THROW( MPI_Rsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm), "Comm::Rsend" );                                        
    }                                                                                                                                
    inline void Isend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Isend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Isend" );                                    
    }                                                                                                                                
    inline Request Isend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Isend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }                                                                                                                                
    inline void Ibsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Ibsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibsend" );                                
    }                                                                                                                                
    inline Request Ibsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Ibsend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }                                                                                                                                
    inline void Issend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Issend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Issend" );                                
    }                                                                                                                                
    inline Request Issend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Issend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }                                                                                                                                
    inline void Irsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {            
        MEL_THROW( MPI_Irsend(ptr, num, (MPI_Datatype) datatype, dst, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irsend" );                                
    }                                                                                                                                
    inline Request Irsend(const void *ptr, const int num, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {                        
        Request rq{};                                                                                                            
        Irsend(ptr, num, datatype, dst, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }

    template<typename T>
    inline void Send(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        Send(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Bsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        Bsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Ssend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        Ssend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Rsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        Rsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }

    template<typename T>
    inline void Isend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {
        Isend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Isend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        return Isend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Ibsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {
        Ibsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Ibsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        return Ibsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Issend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {
        Issend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Issend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        return Issend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }
    template<typename T>
    inline void Irsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm, Request &rq) {
        Irsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Irsend(const std::vector<T> &sbuf, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
        return Irsend(&sbuf[0], sbuf.size(), datatype, dst, tag, comm);
    }

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

    template<typename T>
    inline void Send(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        Send(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Bsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        Bsend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Ssend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        Ssend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Rsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        Rsend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }

    template<typename T>
    inline void Isend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm, Request &rq) {
        Isend(&sbuf[0], sbuf.size(), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Isend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        return Isend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Ibsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm, Request &rq) {
        Ibsend(&sbuf[0], sbuf.size(), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Ibsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        return Ibsend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Issend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm, Request &rq) {
        Issend(&sbuf[0], sbuf.size(), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Issend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        return Issend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }
    template<typename T>
    inline void Irsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm, Request &rq) {
        Irsend(&sbuf[0], sbuf.size(), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Irsend(const std::vector<T> &sbuf, const int dst, const int tag, const Comm &comm) {
        return Irsend(&sbuf[0], sbuf.size(), dst, tag, comm);
    }

    template<typename T>
    inline void Send(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        Send((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Bsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        Bsend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Ssend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        Ssend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Rsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        Rsend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }

    template<typename T>
    inline void Isend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm, Request &rq) {
        Isend((char*) sbuf, snum * sizeof(T), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Isend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        return Isend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Ibsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm, Request &rq) {
        Ibsend((char*) sbuf, snum * sizeof(T), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Ibsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        return Ibsend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Issend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm, Request &rq) {
        Issend((char*) sbuf, snum * sizeof(T), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Issend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        return Issend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }
    template<typename T>
    inline void Irsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm, Request &rq) {
        Irsend((char*) sbuf, snum * sizeof(T), dst, tag, comm, rq);
    }
    template<typename T>
    inline Request Irsend(const T *sbuf, const int snum, const int dst, const int tag, const Comm &comm) {
        return Irsend((char*) sbuf, snum * sizeof(T), dst, tag, comm);
    }

#undef MEL_SEND

    /// ----------------------------------------------------------------------------------------------------------
    ///  Point to Point - RECV
    /// ----------------------------------------------------------------------------------------------------------

    /// Prob incomming message
    inline Status Probe(const int source, const int tag, const Comm &comm) {
        MPI_Status status{};
        MEL_THROW( MPI_Probe(source, tag, (MPI_Comm) comm, &status), "Comm::Probe" );
        return status;
    };
    inline std::pair<bool, Status> Iprobe(const int source, const int tag, const Comm &comm) {
        MPI_Status status{}; int f;
        MEL_THROW( MPI_Iprobe(source, tag, (MPI_Comm) comm, &f, &status), "Comm::Iprobe" );
        return std::make_pair(f != 0, status);
    };

    template<typename T>
    inline int ProbeGetCount(const MPI_Status &status) {
        int c;
        MEL_THROW(MPI_Get_count(&status, MPI_CHAR, &c), "Comm::ProbeGetCount");
        return c / sizeof(T);
    };
    inline int ProbeGetCount(const Datatype &datatype, const Status &status) {
        int c;
        MEL_THROW(MPI_Get_count(&status, (MPI_Datatype) datatype, &c), "Comm::ProbeGetCount");
        return c;
    };
    template<typename T>
    inline int ProbeGetCount(const int src, const int tag, const Comm &comm) {
        MPI_Status status = Probe(src, tag, comm);
        return ProbeGetCount<T>(status);
    };
    inline int ProbeGetCount(const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Status status = Probe(src, tag, comm);
        return ProbeGetCount(datatype, status);
    };

    inline Status Recv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Status status{};                                                                                                        
        MEL_THROW( MPI_Recv(ptr, num, (MPI_Datatype) datatype, src, tag, (MPI_Comm) comm, &status), "Comm::Recv" );                                
        return status;                                                                                                                
    }                                                                                                                                
    inline void Irecv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Irecv(ptr, num, (MPI_Datatype) datatype, src, tag, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Irecv" );                                    
    }                                                                                                                                
    inline Request Irecv(void *ptr, const int num, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        Request rq{};                                                                                                            
        Irecv(ptr, num, datatype, src, tag, comm, rq);                                                                                        
        return rq;                                                                                                                    
    }

    /// Recieve to a vector of known type and length where dbuf is already at least the correct size 
    template<typename T>
    inline Status Recv(std::vector<T> &dbuf, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        return Recv(&dbuf[0], dbuf.size(), datatype, src, tag, comm);
    };
    template<typename T>
    inline Status Recv(std::vector<T> &dbuf, const int dnum, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        return Recv(&dbuf[0], dnum, datatype, src, tag, comm);
    };

    /// Non-blocking recieve to a vector of known type where the vector is at least the correct size
    template<typename T>
    inline Request Irecv(std::vector<T> &dbuf, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        return Irecv(&dbuf[0], dbuf.size(), datatype, src, tag, comm);
    };
    template<typename T>
    inline Request Irecv(std::vector<T> &dbuf, const int dnum, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
        return Irecv(&dbuf[0], dnum, datatype, src, tag, comm);
    };
    template<typename T>
    inline void Irecv(std::vector<T> &dbuf, const Datatype &datatype, const int src, const int tag, const Comm &comm, Request &rq) {
        Irecv(&dbuf[0], dbuf.size(), datatype, src, tag, comm, rq);
    };
    template<typename T>
    inline void Irecv(std::vector<T> &dbuf, const int dnum, const Datatype &datatype, const int src, const int tag, const Comm &comm, Request &rq) {
        Irecv(&dbuf[0], dnum, datatype, src, tag, comm, rq);
    };

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
    /// Recieve to a vector of known type and length where dbuf is already at least the correct size 
    template<typename T>
    inline Status Recv(std::vector<T> &dbuf, const int src, const int tag, const Comm &comm) {
        return Recv(&dbuf[0], dbuf.size(), src, tag, comm);
    };
    template<typename T>
    inline Status Recv(std::vector<T> &dbuf, const int dnum, const int src, const int tag, const Comm &comm) {
        dbuf.resize(dnum);
        return Recv(&dbuf[0], dnum, src, tag, comm);
    };

    /// Dynamic recieve vector of known type with probe and resize
    template<typename T>
    inline Status DynamicRecv(std::vector<T> &dbuf, const int src, const int tag, const Comm &comm) {
        Status status = Probe(src, tag, comm);
        const int dnum = ProbeGetCount<T>(status);
        dbuf.resize(dnum);
        return Recv(dbuf, dnum, status.MPI_SOURCE, status.MPI_TAG, comm);
    };
    template<typename T>
    inline std::vector<T> DynamicRecv(const int src, const int tag, const Comm &comm, Status &status) {
        std::vector<T> dbuf;
        status = DynamicRecv(dbuf, src, tag, comm);
        return dbuf;
    };
    template<typename T>
    inline std::pair<Status, std::vector<T>> DynamicRecv(const int src, const int tag, const Comm &comm) {
        std::vector<T> dbuf;
        Status status = DynamicRecv(dbuf, src, tag, comm);
        return std::make_pair(status, dbuf);
    };

    /// Non-blocking recieve to a vector of known type where the vector is at least the correct size
    template<typename T>
    inline Request Irecv(std::vector<T> &dbuf, const int src, const int tag, const Comm &comm) {
        return Irecv(&dbuf[0], dbuf.size(), src, tag, comm);
    };
    template<typename T>
    inline Request Irecv(std::vector<T> &dbuf, const int dnum, const int src, const int tag, const Comm &comm) {
        dbuf.resize(dnum);
        return Irecv(&dbuf[0], dnum, src, tag, comm);
    };
    template<typename T>
    inline void Irecv(std::vector<T> &dbuf, const int src, const int tag, const Comm &comm, Request &rq) {
        Irecv(&dbuf[0], dbuf.size(), src, tag, comm, rq);
    };
    template<typename T>
    inline void Irecv(std::vector<T> &dbuf, const int dnum, const int src, const int tag, const Comm &comm, Request &rq) {
        dbuf.resize(dnum);
        Irecv(&dbuf[0], dnum, src, tag, comm, rq);
    };

    /// Recieve to a vector of known type and length where dbuf is already at least the correct size 
    template<typename T>
    inline Status Recv(T *dbuf, const int dnum, const int src, const int tag, const Comm &comm) {
        return Recv((char*) dbuf, dnum * sizeof(T), src, tag, comm);
    };
    /// Non-blocking recieve to a vector of known type where the vector is at least the correct size
    template<typename T>
    inline Request Irecv(T *dbuf, const int dnum, const int src, const int tag, const Comm &comm) {
        return Irecv((char*) dbuf, dnum * sizeof(T), src, tag, comm);
    };
    template<typename T>
    inline void Irecv(T *dbuf, const int dnum, const int src, const int tag, const Comm &comm, Request &rq) {
        Irecv((char*) dbuf, dnum * sizeof(T), src, tag, comm, rq);
    };

#undef MEL_RECV

    /// ----------------------------------------------------------------------------------------------------------
    ///  Collectives
    /// ----------------------------------------------------------------------------------------------------------

    /* Bcast */                                                                                                                        
    inline void Bcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm) {                                                                            
        MEL_THROW( MPI_Bcast(ptr, num, (MPI_Datatype) datatype, root, (MPI_Comm) comm), "Comm::Bcast" );                                                                
    }

    /* Scatter / Scatterv */                                                                                                                                
    inline void Scatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Scatter(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Scatter" );                                            
    }        
                                                                                                                                                
    inline void Scatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Scatterv(sptr, snum, displs, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Scatterv" );                                
    }    
                                                                                                                                                    
    /* Gather / Gatherv */                                                                                                                                    
    inline void Gather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Gather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Gather" );                                            
    }    
                                                                                                                                                            
    inline void Gatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm) {
        MEL_THROW( MPI_Gatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displs, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm), "Comm::Gatherv" );                                    
    }
                                                                                                                                                            
    /* Allgather / Allgatherv */                                                                                                                            
    inline void Allgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Allgather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Allgather" );                                            
    }    
                                                                                                                                                            
    inline void Allgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Allgatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displ, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Allgather" );                                    
    }    
                                                                                                                                                
    /* Alltoall / Alltoallv / Alltoallw */                                                                                                                                
    inline void Alltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoall(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Alltoall" );                                                
    }    
                                                                                                                                            
    inline void Alltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoallv(sptr, snum, sdispl, (MPI_Datatype) sdatatype, rptr, rnum, rdispl, (MPI_Datatype) rdatatype, (MPI_Comm) comm), "Comm::Alltoallv" );                            
    }

    inline void Alltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm) {
        MEL_THROW( MPI_Alltoallw(sptr, snum, sdispl, (MPI_Datatype*) sdatatype, rptr, rnum, rdispl, (MPI_Datatype*) rdatatype, (MPI_Comm) comm), "Comm::Alltoallw" );
    }

    /* Reduce / Allreduce */                                                                                                                                
    inline void Reduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        MEL_THROW( MPI_Reduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, root, (MPI_Comm) comm), "Comm::Reduce" );                                                
    }                                                                                                                                                
    inline void Allreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm) {
        MEL_THROW( MPI_Allreduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, (MPI_Comm) comm), "Comm::Allreduce" );                                            
    }
    
#ifdef MEL_3
    inline void Ibcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ibcast(ptr, num, (MPI_Datatype) datatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ibcast");
    }
    inline Request Ibcast(void *ptr, const int num, const Datatype &datatype, const int root, const Comm &comm) {
        Request rq{};
        Ibcast(ptr, num, datatype, root, comm, rq);
        return rq;
    }

    inline void Iscatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iscatter(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatter");
    }
    inline Request Iscatter(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Iscatter(sptr, snum, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    }

    inline void Iscatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iscatterv(sptr, snum, displs, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iscatterv");
    }
    inline Request Iscatterv(void *sptr, const int *snum, const int *displs, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Iscatterv(sptr, snum, displs, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    }

    inline void Igather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Igather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igather");
    }
    inline Request Igather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Igather(sptr, snum, sdatatype, rptr, rnum, rdatatype, root, comm, rq);
        return rq;
    }

    inline void Igatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Igatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displs, (MPI_Datatype) rdatatype, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Igatherv");
    }
    inline Request Igatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displs, const Datatype &rdatatype, const int root, const Comm &comm) {
        Request rq{};
        Igatherv(sptr, snum, sdatatype, rptr, rnum, displs, rdatatype, root, comm, rq);
        return rq;
    }

    inline void Iallgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iallgather(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgather");
    }
    inline Request Iallgather(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Iallgather(sptr, snum, sdatatype, rptr, rnum, rdatatype, comm, rq);
        return rq;
    }

    inline void Iallgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Iallgatherv(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, displ, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallgather");
    }
    inline Request Iallgatherv(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int *rnum, const int *displ, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Iallgatherv(sptr, snum, sdatatype, rptr, rnum, displ, rdatatype, comm, rq);
        return rq;
    }

    inline void Ialltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoall(sptr, snum, (MPI_Datatype) sdatatype, rptr, rnum, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoall");
    }
    inline Request Ialltoall(void *sptr, const int snum, const Datatype &sdatatype, void *rptr, const int rnum, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoall(sptr, snum, sdatatype, rptr, rnum, rdatatype, comm, rq);
        return rq;
    }
    
    inline void Ialltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoallv(sptr, snum, sdispl, (MPI_Datatype) sdatatype, rptr, rnum, rdispl, (MPI_Datatype) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoallv");
    }
    inline Request Ialltoallv(void *sptr, const int *snum, const int *sdispl, const Datatype &sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype &rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoallv(sptr, snum, sdispl, sdatatype, rptr, rnum, rdispl, rdatatype, comm, rq);
        return rq;
    }
    
    inline void Ialltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ialltoallw(sptr, snum, sdispl, (MPI_Datatype*) sdatatype, rptr, rnum, rdispl, (MPI_Datatype*) rdatatype, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ialltoallw");
    }
    inline Request Ialltoallw(void *sptr, const int *snum, const int *sdispl, const Datatype *sdatatype, void *rptr, const int *rnum, const int *rdispl, const Datatype *rdatatype, const Comm &comm) {
        Request rq{};
        Ialltoallw(sptr, snum, sdispl, sdatatype, rptr, rnum, rdispl, rdatatype, comm, rq);
        return rq;
    }
    
    inline void Ireduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm, Request &rq) {
        MEL_THROW(MPI_Ireduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, root, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Ireduce");
    }
    inline Request Ireduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        Request rq{};
        Ireduce(sptr, rptr, num, datatype, op, root, comm, rq);
        return rq;
    }    
    
    inline void Iallreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm, Request &rq) {
        MEL_THROW( MPI_Iallreduce(sptr, rptr, num, (MPI_Datatype) datatype, (MPI_Op) op, (MPI_Comm) comm, (MPI_Request*) &rq), "Comm::Iallreduce" );                            
    }                                                                                                                                                        
    inline Request Iallreduce(void *sptr, void *rptr, const int num, const Datatype &datatype, const Op &op, const Comm &comm) {
        Request rq{};                                                                                                                                        
        Iallreduce(sptr, rptr, num, datatype, op, comm, rq);                                                                                                    
        return rq;                                                                                                                                            
    }
#endif

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

    /// Bcast
    template<typename T>
    inline void Bcast(T *dbuf, const int dnum, const int root, const Comm &comm) {
        Bcast((char*) dbuf, dnum * sizeof(T), root, comm);
    };

    /// Reduce a single value
    template<typename T>
    inline void Reduce(T &val, T &vout, const Op &op, const int root, const Comm &comm) {
        Reduce(&val, &vout, 1, op, root, comm);
    };
    template<typename T>
    inline T Reduce(T &val, const Op &op, const int root, const Comm &comm) {
        T vout;
        Reduce(&val, &vout, 1, op, root, comm);
        return vout;
    };
    template<typename T>
    inline void Allreduce(T &val, T &vout, const Op &op, const Comm &comm) {
        Allreduce(&val, &vout, 1, op, comm);
    };
    template<typename T>
    inline T Allreduce(T &val, const Op &op, const Comm &comm) {
        T vout;
        Allreduce(&val, &vout, 1, op, comm);
        return vout;
    };

    /// Reduce a single value with datatype
    template<typename T>
    inline void Reduce(T &val, T &vout, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        Reduce(&val, &vout, 1, datatype, op, root, comm);
    };
    template<typename T>
    inline T Reduce(T &val, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        T vout;
        Reduce(&val, &vout, 1, datatype, op, root, comm);
        return vout;
    };
    template<typename T>
    inline void Allreduce(T &val, T &vout, const Datatype &datatype, const Op &op, const Comm &comm) {
        Allreduce(&val, &vout, 1, datatype, op, comm);
    };
    template<typename T>
    inline T Allreduce(T &val, const Datatype &datatype, const Op &op, const Comm &comm) {
        T vout;
        Allreduce(&val, &vout, 1, datatype, op, comm);
        return vout;
    };

#ifdef MEL_3
    template<typename T>
    inline Request Ibcast(T *dbuf, const int dnum, const int root, const Comm &comm) {
        return Ibcast((char*) dbuf, dnum * sizeof(T), root, comm);
    };
    template<typename T>
    inline void Ibcast(T *dbuf, const int dnum, const int root, const Comm &comm, Request &rq) {
        Ibcast((char*) dbuf, dnum * sizeof(T), root, comm, rq);
    };
    template<typename T>
    inline Request Ireduce(T &val, T &vout, const Op &op, const int root, const Comm &comm) {
        return Ireduce(&val, &vout, 1, op, root, comm);
    };
    template<typename T>
    inline void Ireduce(T &val, T &vout, const Op &op, const int root, const Comm &comm, Request &rq) {
        Ireduce(&val, &vout, 1, op, root, comm, rq);
    };
    template<typename T>
    inline Request Iallreduce(T &val, T &vout, const Op &op, const Comm &comm) {
        return Iallreduce(&val, &vout, 1, op, comm);
    };
    template<typename T>
    inline void Iallreduce(T &val, T &vout, const Op &op, const Comm &comm, Request &rq) {
        Iallreduce(&val, &vout, 1, op, comm, rq);
    };
    template<typename T>
    inline Request Ireduce(T &val, T &vout, const Datatype &datatype, const Op &op, const int root, const Comm &comm) {
        return Ireduce(&val, &vout, 1, datatype, op, root, comm);
    };
    template<typename T>
    inline void Ireduce(T &val, T &vout, const Datatype &datatype, const Op &op, const int root, const Comm &comm, Request &rq) {
        Ireduce(&val, &vout, 1, datatype, op, root, comm, rq);
    };
    template<typename T>
    inline Request Iallreduce(T &val, T &vout, const Datatype &datatype, const Op &op, const Comm &comm) {
        return Iallreduce(&val, &vout, 1, datatype, op, comm);
    };
    template<typename T>
    inline void Iallreduce(T &val, T &vout, const Datatype &datatype, const Op &op, const Comm &comm, Request &rq) {
        Iallreduce(&val, &vout, 1, datatype, op, comm, rq);
    };
#endif

#undef MEL_COLLECTIVE
#undef MEL_3_COLLECTIVE

    /// ----------------------------------------------------------------------------------------------------------
    ///  Remote Memory Access - RMA
    /// ----------------------------------------------------------------------------------------------------------

    enum class LockType {
        EXCLUSIVE = MPI_LOCK_EXCLUSIVE,
        SHARED = MPI_LOCK_SHARED
    };

    struct Win {
        static const Win WIN_NULL;

        MPI_Win win;

        Win() : win(MPI_WIN_NULL) {};
        explicit Win(const MPI_Win &_e) : win(_e) {};
        inline Win& operator=(const MPI_Win &_e) {
            win = _e;
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

    /// Error handling
    inline ErrorHandler WinCreateErrorHandler(ErrorHandlerFunc func) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Win_create_errhandler((MPI_Win_errhandler_function*) func, &errHndl), "RMA::WinCreateErrorHandler" );
        return ErrorHandler(errHndl);
    };
    inline void WinSetErrorHandler(const Win &win, const ErrorHandler &errHndl) {
        MEL_THROW( MPI_Win_set_errhandler((MPI_Win) win, (MPI_Errhandler) errHndl), "RMA::WinSetErrorHandler" );
    };
    inline void WinSetErrorHandler(const Win &win, ErrorHandlerFunc func) {
        WinSetErrorHandler(win, WinCreateErrorHandler(func));
    };
    inline ErrorHandler WinGetErrorHandler(const Win &win) {
        MPI_Errhandler errHndl;
        MEL_THROW( MPI_Win_get_errhandler((MPI_Win) win, &errHndl), "RMA::WinGetErrorHandler");
        return ErrorHandler(errHndl);
    };

    /// Creation
    inline Win WinCreate(void *ptr, const Aint size, const int disp_unit, const Comm &comm) {
        MPI_Win win;                                                                        
        MEL_THROW( MPI_Win_create(ptr, size, disp_unit, MPI_INFO_NULL, (MPI_Comm) comm, (MPI_Win*) &win), "RMA::WinCreate" );
        MEL_THROW( MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN), "RMA::WinCreate(SetErrorHandler)" );                                                                
        return Win(win);
    };
    template<typename T> 
    inline Win WinCreate(T *ptr, const Aint size, const Comm &comm) {
        return WinCreate(ptr, size, sizeof(T), comm);
    };

    /// Synchronization
    inline void WinFence(const Win &win, const int assert_tag) {
        MEL_THROW( MPI_Win_fence(assert_tag, (MPI_Win) win), "RMA::WinFence" );
    };
    inline void WinFence(const Win &win) {
        WinFence(win, 0);
    };
    inline void WinLock(const Win &win, const int rank, const int assert_tag, const LockType lock_type) {
        MEL_THROW( MPI_Win_lock((int) lock_type, rank, assert_tag, (MPI_Win) win), "RMA::WinLock" );
    };
    inline void WinLock(const Win &win, const int rank, const LockType lock_type) {
        WinLock(win, rank, 0, lock_type);
    };
    inline void WinLockExclusive(const Win &win, const int rank, const int assert_tag) {
        WinLock(win, rank, assert_tag, LockType::EXCLUSIVE);
    };
    inline void WinLockExclusive(const Win &win, const int rank) {
        WinLockExclusive(win, rank, 0);
    };
    inline void WinLockShared(const Win &win, const int rank, const int assert_tag) {
        WinLock(win, rank, assert_tag, LockType::SHARED);
    };
    inline void WinLockShared(const Win &win, const int rank) {
        WinLockShared(win, rank, 0);
    };
    inline void WinUnlock(const Win &win, const int rank) {
        MEL_THROW( MPI_Win_unlock(rank, (MPI_Win) win), "RMA::WinUnlock" );
    };

    /// Put
    inline void Put(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        MEL_THROW( MPI_Put(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win), "RMA::Put" );
    };
    inline void Put(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Put(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win);
    };

    /// Get
    inline void Get(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        MEL_THROW( MPI_Get(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win), "RMA::Get" );
    };
    inline void Get(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Get(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win);
    };

#ifdef MEL_3
    inline void WinLockAll(const Win &win, const int assert_tag) {
        MEL_THROW(MPI_Win_lock_all(assert_tag, (MPI_Win) win), "RMA::WinLockAll");
    };
    inline void WinLockAll(const Win &win) {
        WinLockAll(win, 0);
    };
    inline void WinUnlockAll(const Win &win) {
        MEL_THROW(MPI_Win_unlock_all((MPI_Win) win), "RMA::WinUnlockAll");
    };
    inline void WinFlush(const Win &win, const int rank) {
        MEL_THROW(MPI_Win_flush(rank, (MPI_Win) win), "RMA::WinFlush");
    };
    inline void WinFlushAll(const Win &win) {
        MEL_THROW(MPI_Win_flush_all((MPI_Win) win), "RMA::WinFlushAll");
    };
    inline void WinFlushLocal(const Win &win, const int rank) {
        MEL_THROW(MPI_Win_flush_local(rank, (MPI_Win) win), "RMA::WinFlushLocal");
    };
    inline void WinFlushLocalAll(const Win &win) {
        MEL_THROW(MPI_Win_flush_local_all((MPI_Win) win), "RMA::WinFlushLocalAll");
    };
    inline void WinSync(const Win &win) {
        MEL_THROW(MPI_Win_sync((MPI_Win) win), "RMA::WinSync");
    };

    inline void Rput(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        MEL_THROW(MPI_Rput(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win, (MPI_Request*) &rq), "RMA::Rput");
    };
    inline void Rput(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        Rput(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win, rq);
    };
    inline Request Rput(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Request rq;
        Rput(origin_ptr, origin_num, origin_datatype, target_disp, target_num, target_datatype, target_rank, win, rq);
        return rq;
    };
    inline Request Rput(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win) {
        return Rput(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win);
    };

    inline void Rget(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        MEL_THROW(MPI_Rget(origin_ptr, origin_num, (MPI_Datatype) origin_datatype, target_rank, target_disp, target_num, (MPI_Datatype) target_datatype, (MPI_Win) win, (MPI_Request*) &rq), "RMA::Rget");
    };
    inline void Rget(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win, Request &rq) {
        Rget(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win, rq);
    };
    inline Request Rget(void *origin_ptr, int origin_num, const Datatype &origin_datatype, const Aint target_disp, const int target_num, const Datatype &target_datatype, const int target_rank, const Win &win) {
        Request rq;
        Rget(origin_ptr, origin_num, origin_datatype, target_disp, target_num, target_datatype, target_rank, win, rq);
        return rq;
    };
    inline Request Rget(void *origin_ptr, const Datatype &origin_datatype, const Aint target_disp, const Datatype &target_datatype, const int target_rank, const Win &win) {
        return Rget(origin_ptr, 1, origin_datatype, target_disp, 1, target_datatype, target_rank, win);
    };
#endif

    inline void WinFree(Win &win) {
        if (win != MEL::Win::WIN_NULL)
            MEL_THROW( MPI_Win_free((MPI_Win*) &win), "RMA::FreeWin" );                                                                
    };    

    inline void WinFree(std::vector<Win> &wins) {
        for (auto &w : wins) WinFree(w);
    };

    template<typename T0, typename T1, typename ...Args>
    inline void WinFree(T0 &d0, T1 &d1, Args &&...args) {
        WinFree(d0);
        WinFree(d1, args...);
    };

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

    inline void MutexFree(Mutex &mutex) {
        MEL::Barrier(mutex.comm);
        MEL::WinFree(mutex.win);
        MEL::MemFree(mutex.val);
    };

    
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

    inline bool MutexTest(const Mutex &mutex) {
        return mutex.locked;
    };

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

    template<typename T>
    inline Shared<T> SharedCreate(const int len, const int root, const Comm &comm) {
        return SharedCreate<T>(len, CommRank(comm), CommSize(comm), root, comm);
    };

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

    template<typename T>
    inline void SharedFree(Shared<T> &shared) {
        MEL::Barrier(shared.mutex.comm);
        MEL::WinFree(shared.win);
        MEL::MemFree(shared.ptr);
        MEL::MutexFree(shared.mutex);
        MEL::TypeFree(shared.typeData);
    };

    template<typename T>
    inline bool SharedTest(const Shared<T> &shared) {
        return shared.is_locked();
    };

    template<typename T>
    inline void SharedLock_noget(Shared<T> &shared) {
        SharedLock_noget(shared, 0, shared.len - 1);
    };
    template<typename T>
    inline void SharedLock_noget(Shared<T> &shared, const int start, const int end) {
        MEL::MutexLock(shared.mutex); // , start, end
    };

    template<typename T>
    inline void SharedLock(Shared<T> &shared) {
        SharedLock(shared, 0, shared.len - 1);
    };
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

    template<typename T>
    inline void SharedUnlock_noput(Shared<T> &shared) {
        SharedUnlock_noput(shared, 0, shared.len - 1);
    };
    template<typename T>
    inline void SharedUnlock_noput(Shared<T> &shared, const int start, const int end) {
        MEL::MutexUnlock(shared.mutex); // , start, end
    };

    template<typename T>
    inline void SharedUnlock(Shared<T> &shared) {
        SharedUnlock(shared, 0, shared.len - 1);
    };
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