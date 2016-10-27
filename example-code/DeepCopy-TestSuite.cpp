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

/// Run this test suite using mpirun -n 2 ./DeepCopy-TestSuite
/// It will produce two output files "DeepCopy - Test - Rank <i> of 2.out" and ".err"
/// for each process.

#define  MEL_IMPLEMENTATION
#include "MEL.hpp"
#include "MEL_deepcopy.hpp"

/// This file depends on the "Catch" testing framework
/// available here https://github.com/philsquared/Catch 
/// under the Boost Software License, Version 1.0

#define CATCH_CONFIG_NOSTDOUT
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

struct TestObject {
    std::vector<int> arr;

    TestObject() {};
    TestObject(const int size) : arr(size) {
        for (int i = 0; i < size; ++i) arr[i] = i;
    };

    inline bool operator==(TestObject &rhs) const {
        if (arr.size() != rhs.arr.size()) return false;
        for (int i = 0; i < arr.size(); ++i) 
            if (arr[i] != rhs.arr[i]) return false;
        return true;
    };

    template<typename MSG>
    inline void DeepCopy(MSG &msg) {
        msg & arr;
    };
};

TEST_CASE("Send/Recv", "[Send][Recv]") {
    
    MEL::Comm comm = MEL::Comm::WORLD;
    const int comm_rank = MEL::CommRank(comm),
              comm_size = MEL::CommSize(comm);

    REQUIRE(comm_size == 2);

    SECTION("Non-Deep") {

        SECTION("Send a pointer payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(1);
                *p = 42;
                MEL::Deep::Send(p, 1, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                int *p = nullptr;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(*p == 42);
                MEL::MemFree(p);
            }
        }

        SECTION("Send a pointer/len payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(10);
                for (int i = 0; i < 10; ++i) p[i] = i;
                MEL::Deep::Send(p, 10, 1, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                int *p = nullptr;
                MEL::Deep::Recv(p, 10, 0, 0, comm);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
                MEL::MemFree(p);
            }
        }

        SECTION("Send an object payload") {
            if (comm_rank == 0) {
                int p = 42;
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                int p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p == 42);
            }
        }

        SECTION("Send a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<int> p(10);
                for (int i = 0; i < 10; ++i) p[i] = i;
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                std::vector<int> p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
            }
        }

        SECTION("Send a std::list payload") {
            if (comm_rank == 0) {
                std::list<int> p;
                for (int i = 0; i < 10; ++i) p.push_back(i);
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                std::list<int> p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == i); }
            }
        }
    }

    SECTION("Deep") {

        SECTION("Send a pointer payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemConstruct<TestObject>(10);
                MEL::Deep::Send(p, 1, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                TestObject *p = nullptr;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(*p == TestObject(10));
                MEL::MemFree(p);
            }
        }

        SECTION("Send a pointer/len payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemAlloc<TestObject>(10);
                for (int i = 0; i < 10; ++i) new (&p[i]) TestObject(i);
                MEL::Deep::Send(p, 10, 1, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                TestObject *p = nullptr;
                MEL::Deep::Recv(p, 10, 0, 0, comm);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
                MEL::MemFree(p);
            }
        }

        SECTION("Send an object payload") {
            if (comm_rank == 0) {
                TestObject p(42);
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                TestObject p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p == TestObject(42));
            }
        }

        SECTION("Send a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<TestObject> p(10);
                for (int i = 0; i < 10; ++i) p[i] = TestObject(i);
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                std::vector<TestObject> p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
            }
        }

        SECTION("Send a std::list payload") {
            if (comm_rank == 0) {
                std::list<TestObject> p;
                for (int i = 0; i < 10; ++i) p.push_back(TestObject(i));
                MEL::Deep::Send(p, 1, 0, comm);
            }
            else if (comm_rank == 1) {
                std::list<TestObject> p;
                MEL::Deep::Recv(p, 0, 0, comm);
                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == TestObject(i)); }
            }
        }
    }
}

TEST_CASE("Bcast", "[Bcast]") {

    MEL::Comm comm = MEL::Comm::WORLD;
    const int comm_rank = MEL::CommRank(comm),
              comm_size = MEL::CommSize(comm);

    REQUIRE(comm_size == 2);

    SECTION("Non-Deep") {

        SECTION("Bcast a pointer payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(1);
                *p = 42;
                MEL::Deep::Bcast(p, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                int *p = nullptr;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(*p == 42);
                MEL::MemFree(p);
            }
        }

        SECTION("Bcast a pointer/len payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(10);
                for (int i = 0; i < 10; ++i) p[i] = i;
                MEL::Deep::Bcast(p, 10, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                int *p = nullptr;
                MEL::Deep::Bcast(p, 10, 0, comm);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
                MEL::MemFree(p);
            }
        }

        SECTION("Bcast an object payload") {
            if (comm_rank == 0) {
                int p = 42;
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                int p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p == 42);
            }
        }

        SECTION("Bcast a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<int> p(10);
                for (int i = 0; i < 10; ++i) p[i] = i;
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                std::vector<int> p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
            }
        }

        SECTION("Bcast a std::list payload") {
            if (comm_rank == 0) {
                std::list<int> p;
                for (int i = 0; i < 10; ++i) p.push_back(i);
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                std::list<int> p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == i); }
            }
        }
    }

    SECTION("Deep") {

        SECTION("Bcast a pointer payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemConstruct<TestObject>(10);
                MEL::Deep::Bcast(p, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                TestObject *p = nullptr;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(*p == TestObject(10));
                MEL::MemFree(p);
            }
        }

        SECTION("Bcast a pointer/len payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemAlloc<TestObject>(10);
                for (int i = 0; i < 10; ++i) new (&p[i]) TestObject(i);
                MEL::Deep::Bcast(p, 10, 0, comm);
                MEL::MemFree(p);
            }
            else if (comm_rank == 1) {
                TestObject *p = nullptr;
                MEL::Deep::Bcast(p, 10, 0, comm);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
                MEL::MemFree(p);
            }
        }

        SECTION("Bcast an object payload") {
            if (comm_rank == 0) {
                TestObject p(42);
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                TestObject p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p == TestObject(42));
            }
        }

        SECTION("Bcast a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<TestObject> p(10);
                for (int i = 0; i < 10; ++i) p[i] = TestObject(i);
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                std::vector<TestObject> p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
            }
        }

        SECTION("Bcast a std::list payload") {
            if (comm_rank == 0) {
                std::list<TestObject> p;
                for (int i = 0; i < 10; ++i) p.push_back(TestObject(i));
                MEL::Deep::Bcast(p, 0, comm);
            }
            else if (comm_rank == 1) {
                std::list<TestObject> p;
                MEL::Deep::Bcast(p, 0, comm);
                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == TestObject(i)); }
            }
        }
    }
}

TEST_CASE("MEL::File", "[MEL File]") {

    MEL::Comm comm = MEL::Comm::WORLD;
    const int comm_rank = MEL::CommRank(comm),
              comm_size = MEL::CommSize(comm);

    REQUIRE(comm_size == 2);

    SECTION("Non-Deep") {

        MEL::Barrier(comm);

        SECTION("MEL::File a pointer payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(1);
                *p = 42;

                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm); 
                int *p = nullptr;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(*p == 42);
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File a pointer/len payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(10);
                for (int i = 0; i < 10; ++i) p[i] = i;

                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, 10, file);
                MEL::FileClose(file);

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                int *p = nullptr;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, 10, file);
                MEL::FileClose(file);

                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File an object payload") {
            if (comm_rank == 0) {
                int p = 42;

                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                int p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p == 42);
            }
        }

        MEL::Barrier(comm);
        
        SECTION("MEL::File a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<int> p(10);
                for (int i = 0; i < 10; ++i) p[i] = i;

                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::vector<int> p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File a std::list payload") {
            if (comm_rank == 0) {
                std::list<int> p;
                for (int i = 0; i < 10; ++i) p.push_back(i);
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::list<int> p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == i); }
            }
        }

        MEL::Barrier(comm);

    }

    SECTION("Deep") {

        MEL::Barrier(comm);

        SECTION("MEL::File a pointer payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemConstruct<TestObject>(10);
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject *p = nullptr;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(*p == TestObject(10));
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File a pointer/len payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemAlloc<TestObject>(10);
                for (int i = 0; i < 10; ++i) new (&p[i]) TestObject(i);
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, 10, file);
                MEL::FileClose(file);

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject *p = nullptr;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, 10, file);
                MEL::FileClose(file);

                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File an object payload") {
            if (comm_rank == 0) {
                TestObject p(42);
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p == TestObject(42));
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<TestObject> p(10);
                for (int i = 0; i < 10; ++i) p[i] = TestObject(i);
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::vector<TestObject> p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
            }
        }

        MEL::Barrier(comm);

        SECTION("MEL::File a std::list payload") {
            if (comm_rank == 0) {
                std::list<TestObject> p;
                for (int i = 0; i < 10; ++i) p.push_back(TestObject(i));
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
                MEL::Deep::FileWrite(p, file);
                MEL::FileClose(file);

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::list<TestObject> p;
                
                MEL::File file = MEL::FileOpenIndividual("test.tmp", MEL::FileMode::DELETE_ON_CLOSE | MEL::FileMode::RDONLY);
                MEL::Deep::FileRead(p, file);
                MEL::FileClose(file);

                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == TestObject(i)); }
            }
        }

        MEL::Barrier(comm);

    }
}

TEST_CASE("STL File", "[STL File]") {

    MEL::Comm comm = MEL::Comm::WORLD;
    const int comm_rank = MEL::CommRank(comm),
              comm_size = MEL::CommSize(comm);

    REQUIRE(comm_size == 2);

    SECTION("Non-Deep") {

        MEL::Barrier(comm);

        SECTION("STL File a pointer payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(1);
                *p = 42;

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                int *p = nullptr;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(*p == 42);
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a pointer/len payload") {
            if (comm_rank == 0) {
                int *p = MEL::MemAlloc<int>(10);
                for (int i = 0; i < 10; ++i) p[i] = i;

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, 10, file);
                file.close();

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                int *p = nullptr;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, 10, file);
                file.close();

                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File an object payload") {
            if (comm_rank == 0) {
                int p = 42;

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                int p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p == 42);
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<int> p(10);
                for (int i = 0; i < 10; ++i) p[i] = i;

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::vector<int> p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == i); }
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a std::list payload") {
            if (comm_rank == 0) {
                std::list<int> p;
                for (int i = 0; i < 10; ++i) p.push_back(i);

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::list<int> p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == i); }
            }
        }

        MEL::Barrier(comm);

    }

    SECTION("Deep") {

        MEL::Barrier(comm);

        SECTION("STL File a pointer payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemConstruct<TestObject>(10);

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject *p = nullptr;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(*p == TestObject(10));
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a pointer/len payload") {
            if (comm_rank == 0) {
                TestObject *p = MEL::MemAlloc<TestObject>(10);
                for (int i = 0; i < 10; ++i) new (&p[i]) TestObject(i);

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, 10, file);
                file.close();

                MEL::MemFree(p);
                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject *p = nullptr;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, 10, file);
                file.close();

                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
                MEL::MemFree(p);
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File an object payload") {
            if (comm_rank == 0) {
                TestObject p(42);

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                TestObject p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p == TestObject(42));
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a std::vector payload") {
            if (comm_rank == 0) {
                std::vector<TestObject> p(10);
                for (int i = 0; i < 10; ++i) p[i] = TestObject(i);

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::vector<TestObject> p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p.size() == 10);
                for (int i = 0; i < 10; ++i) { REQUIRE(p[i] == TestObject(i)); }
            }
        }

        MEL::Barrier(comm);

        SECTION("STL File a std::list payload") {
            if (comm_rank == 0) {
                std::list<TestObject> p;
                for (int i = 0; i < 10; ++i) p.push_back(TestObject(i));

                std::ofstream file("test.tmp", std::ios::out | std::ios::binary);
                file.clear();
                MEL::Deep::FileWrite(p, file);
                file.close();

                MEL::Barrier(comm);
            }
            else if (comm_rank == 1) {
                MEL::Barrier(comm);
                std::list<TestObject> p;

                std::ifstream file("test.tmp", std::ios::in | std::ios::binary);
                MEL::Deep::FileRead(p, file);
                file.close();

                REQUIRE(p.size() == 10);
                auto it = p.begin();
                for (int i = 0; i < 10; ++i) { REQUIRE(*it++ == TestObject(i)); }
            }
        }

        MEL::Barrier(comm);

    }
}

std::ofstream localOut, localErr;

std::ostream& Catch::cout() {
    return localOut;
};
std::ostream& Catch::cerr() {
    return localErr;
};

int main(int argc, char *argv[]) {
    MEL::Init(argc, argv);

    MEL::Comm comm = MEL::Comm::WORLD;
    const int comm_rank = MEL::CommRank(comm),
              comm_size = MEL::CommSize(comm);

    if (comm_rank == 0) {
        int tmp;
        //std::cin >> tmp;
    }

    std::stringstream outstr;
    outstr << "DeepCopy - Test - Rank " << comm_rank << " of " << comm_size << ".out";
    localOut = std::ofstream(outstr.str(), std::ios::out | std::ios::binary);
    localOut.clear();

    std::stringstream errstr;
    errstr << "DeepCopy - Test - Rank " << comm_rank << " of " << comm_size << ".err";
    localErr = std::ofstream(errstr.str(), std::ios::out | std::ios::binary);
    localErr.clear();

    int result = Catch::Session().run(argc, argv);
    localOut.close();
    localErr.close();

    MEL::Finalize();
    return result;
}