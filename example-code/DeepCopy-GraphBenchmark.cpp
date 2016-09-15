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
#define  MEL_IMPLEMENTATION
#include "MEL.hpp"
#include "MEL_deepcopy.hpp"

#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <stack>
#include <string>

#ifdef FILE_TEST
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/utility.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/shared_ptr.hpp"
#endif

// Older GCC doesn't define this
namespace std {
    template<typename T>
    std::string to_string(const T& n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

inline void RM(const std::string &path) {
    std::system(("rm \"" + path + "\"").c_str());
};

#ifdef FILE_TEST
template<typename T>
struct BoostDiGraphNode {
    T value;
    std::vector<std::shared_ptr<BoostDiGraphNode<T>>> edges;

    BoostDiGraphNode() {};
    explicit BoostDiGraphNode(const T &_value) : value(_value) {};

    template<class Archive>
    void serialize(Archive& ar, const unsigned version) {
        ar & value;
        ar & edges;
    };
};

inline std::shared_ptr<BoostDiGraphNode<int>> MakeBoostBTreeGraph(const int numNodes) {
    /// BTree Graph
    std::vector<std::shared_ptr<BoostDiGraphNode<int>>> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = std::make_shared<BoostDiGraphNode<int>>(i);
    }

    if (numNodes > 1) nodes[0]->edges.push_back(nodes[1]);

    for (int i = 1; i < numNodes; ++i) {
        const int j = ((i - 1) * 2) + 2;
        nodes[i]->edges.reserve(2);
        if (j < numNodes)		nodes[i]->edges.push_back(nodes[j]);
        if ((j + 1) < numNodes) nodes[i]->edges.push_back(nodes[j + 1]);
    }
    return nodes[0];
};

inline std::shared_ptr<BoostDiGraphNode<int>> MakeBoostRingGraph(const int numNodes) {
    /// Ring Graph
    std::vector<std::shared_ptr<BoostDiGraphNode<int>>> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = std::make_shared<BoostDiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        nodes[i]->edges.reserve(2);
        nodes[i]->edges.push_back(nodes[(i + 1) % numNodes]);
        nodes[i]->edges.push_back(nodes[(i == 0) ? (numNodes - 1) : (i - 1)]);
    }
    return nodes[0];
};

inline std::shared_ptr<BoostDiGraphNode<int>> MakeBoostRandomGraph(const int numNodes) {
    /// Random Graph
    std::vector<std::shared_ptr<BoostDiGraphNode<int>>> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = std::make_shared<BoostDiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        const int numEdges = (rand() % (numNodes));
        nodes[i]->edges.reserve(numEdges);
        for (int j = 0; j < numEdges; ++j) {
            nodes[i]->edges.push_back(nodes[rand() % numNodes]);
        }
    }
    return nodes[0];
};

inline std::shared_ptr<BoostDiGraphNode<int>> MakeBoostFullyConnectedGraph(const int numNodes) {
    /// Fully Connected Graph
    std::vector<std::shared_ptr<BoostDiGraphNode<int>>> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = std::make_shared<BoostDiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        nodes[i]->edges.reserve(numNodes);
        for (int j = 0; j < numNodes; ++j) {
            nodes[i]->edges.push_back(nodes[j]);
        }
    }

    return nodes[0];
};

inline void VisitGraph(std::shared_ptr<BoostDiGraphNode<int>> &root, std::function<void(std::shared_ptr<BoostDiGraphNode<int>> &node)> func) {
    std::unordered_set<std::shared_ptr<BoostDiGraphNode<int>>> pointerMap;
    std::stack<std::shared_ptr<BoostDiGraphNode<int>>> stack;

    stack.push(root);
    while (!stack.empty()) {
        std::shared_ptr<BoostDiGraphNode<int>> node = stack.top();
        stack.pop();

        // If node has not been visited
        if (pointerMap.find(node) == pointerMap.end()) {
            pointerMap.insert(node);
            for (auto e : node->edges) stack.push(e);
            func(node);
        }
    }
};


template<std::shared_ptr<BoostDiGraphNode<int>>(*T)(const int)>
inline void RunBoostVerification(const std::string &outputName, const int numInstances, const MEL::Comm &comm, const std::string &layout) {
    const int rank = MEL::CommRank(comm),
        size = MEL::CommSize(comm);

    if (rank > 1) return;

    for (int i = 0; i <= numInstances; ++i) {
        // Load the graph on the root process
        std::shared_ptr<BoostDiGraphNode<int>> graph;
        if (rank == 0) {
            graph = T(1 << i);

            std::ofstream graphFile(outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::out | std::ios::binary);
            if (graphFile.is_open()) {
                boost::archive::binary_oarchive oa(graphFile);
                oa << graph;
                graphFile.close();
            }
        }

        MEL::Barrier(comm);

        if (rank == 1) {
            std::ifstream graphFile(outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::in | std::ios::binary);
            if (graphFile.is_open()) {
                boost::archive::binary_iarchive ia(graphFile);
                ia >> graph;
                graphFile.close();
            }
        }

        auto GetNodeName = [](std::shared_ptr<BoostDiGraphNode<int>> &node) -> std::string {
            return "Node_" + std::to_string((unsigned long long)(node.get()));
        };

        const std::string dotFileName = outputName + "-" + std::to_string((unsigned long long)(i)) + "-node-" + std::to_string((unsigned long long)(rank));

        std::ofstream dotFile(dotFileName + ".dot", std::ios::out);
        if (dotFile.is_open()) {

            dotFile << "digraph graph_" + std::to_string((unsigned long long)(i)) + " {" << std::endl;

            dotFile << "\t" << layout << std::endl;

            VisitGraph(graph, [&](std::shared_ptr<BoostDiGraphNode<int>> &node) -> void {
                dotFile << "\t" << GetNodeName(node) << " [label=\"" << node->value << "\"]" << std::endl;

                for (int j = 0; j < node->edges.size(); ++j) {
                    dotFile << "\t" << GetNodeName(node) << " -> " << GetNodeName(node->edges[j]) << std::endl;
                }
            });

            dotFile << "}" << std::endl;

            dotFile.close();
        }

        std::system(("dot -Tpng \"" + dotFileName + ".dot\" > \"" + dotFileName + ".png\"").c_str());
    }
};

template<std::shared_ptr<BoostDiGraphNode<int>>(*T)(const int)>
inline void RunBoostFileBenchmarks(const std::string &outputName, const int numRuns, const int numInstances, const MEL::Comm &comm) {
    const int rank = MEL::CommRank(comm),
        size = MEL::CommSize(comm);

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "Boost File Write Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // Boost FileWrite Test                                                                          //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {
            // Load the graph on the root process
            std::shared_ptr<BoostDiGraphNode<int>> graph = T(1 << i);

            double startTime, endTime, deltaTime;

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                RM("/tmp/csjoss/" + outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph");

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream graphFile("/tmp/csjoss/" + outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::out | std::ios::binary);
                if (graphFile.is_open()) {
                    boost::archive::binary_oarchive oa(graphFile);
                    oa << graph;
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "Boost Binary File Write in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-Boost-Deep-FileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }
        }
    }

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "Boost File Read Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // Boost FileRead Test                                                                                 //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {

            double startTime, endTime, deltaTime;

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                std::shared_ptr<BoostDiGraphNode<int>> graph;

                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream graphFile("/tmp/csjoss/" + outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::in | std::ios::binary);
                if (graphFile.is_open()) {
                    boost::archive::binary_iarchive ia(graphFile);
                    ia >> graph;
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            RM("/tmp/csjoss/" + outputName + "-Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".graph");

            std::cout << "Boost Binary File Read in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-Boost-Deep-FileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }
        }
    }

    RM("/tmp/csjoss/*.graph");
};
#endif

template<typename T>
struct DiGraphNode {
    T value;
    std::vector<DiGraphNode<T>*> edges;

    DiGraphNode() {};
    explicit DiGraphNode(const T &_value) : value(_value) {};

    template<typename MSG>
    inline void DeepCopy(MSG &msg) {
        msg & edges;
        for (auto &e : edges) msg.packSharedPtr(e);
    };
};

inline DiGraphNode<int>* MakeBTreeGraph(const int numNodes) {
    /// BTree Graph
    std::vector<DiGraphNode<int>*> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = MEL::MemConstruct<DiGraphNode<int>>(i);
    }

    if (numNodes > 1) nodes[0]->edges.push_back(nodes[1]);

    for (int i = 1; i < numNodes; ++i) {
        const int j = ((i - 1) * 2) + 2;
        nodes[i]->edges.reserve(2);
        if (j < numNodes)		nodes[i]->edges.push_back(nodes[j]);
        if ((j + 1) < numNodes) nodes[i]->edges.push_back(nodes[j + 1]);
    }
    return nodes[0];
};

inline DiGraphNode<int>* MakeRingGraph(const int numNodes) {
    /// Ring Graph
    std::vector<DiGraphNode<int>*> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = MEL::MemConstruct<DiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        nodes[i]->edges.reserve(2);
        nodes[i]->edges.push_back(nodes[(i + 1) % numNodes]);
        nodes[i]->edges.push_back(nodes[(i == 0) ? (numNodes - 1) : (i - 1)]);
    }
    return nodes[0];
};

inline DiGraphNode<int>* MakeRandomGraph(const int numNodes) {
    /// Random Graph
    std::vector<DiGraphNode<int>*> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = MEL::MemConstruct<DiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        const int numEdges = (rand() % (numNodes));
        nodes[i]->edges.reserve(numEdges);
        nodes[i]->edges.push_back(nodes[(i + 1) % numNodes]);
        for (int j = 1; j < numEdges; ++j) {
            nodes[i]->edges.push_back(nodes[rand() % numNodes]);
        }
    }
    return nodes[0];
};

inline DiGraphNode<int>* MakeFullyConnectedGraph(const int numNodes) {
    /// Fully Connected Graph
    std::vector<DiGraphNode<int>*> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = MEL::MemConstruct<DiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        nodes[i]->edges.reserve(numNodes);
        for (int j = 0; j < numNodes; ++j) {
            nodes[i]->edges.push_back(nodes[j]);
        }
    }

    return nodes[0];
};

inline void VisitGraph(DiGraphNode<int> *&root, std::function<void(DiGraphNode<int> *&node)> func) {
    std::unordered_set<DiGraphNode<int>*> pointerMap;
    std::stack<DiGraphNode<int>*> stack;

    stack.push(root);
    while (!stack.empty()) {
        DiGraphNode<int> *node = stack.top();
        stack.pop();

        // If node has not been visited
        if (pointerMap.find(node) == pointerMap.end()) {
            pointerMap.insert(node);
            for (auto e : node->edges) stack.push(e);
            func(node);
        }
    }
};

inline void DestructGraph(DiGraphNode<int> *&root) {
    VisitGraph(root, [](DiGraphNode<int> *&node) -> void {
        MEL::MemDestruct(node);
    });
};

template<DiGraphNode<int>*(*T)(const int)>
inline void RunVerification(const std::string &outputName, const int numInstances, const MEL::Comm &comm, const std::string &layout) {
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);
    
    for (int i = 0; i <= numInstances; ++i) {
        if (rank == 0) {
            // Load the graph on the root process
            DiGraphNode<int> *graph = T(1 << i);

            MEL::Deep::Send(graph, 1, 0, comm);

            DestructGraph(graph);
        }
        else if (rank == 1) {
            DiGraphNode<int> *graph = nullptr;
            MEL::Deep::Recv(graph, 0, 0, comm);

            auto GetNodeName = [](const DiGraphNode<int> *node) -> std::string {
                return "Node_" + std::to_string((size_t) node);
            };
        
            const std::string dotFileName = outputName + "-" + std::to_string((unsigned long long)(i));

            std::ofstream dotFile(dotFileName + ".dot", std::ios::out);
            if (dotFile.is_open()) {
            
                dotFile << "digraph graph_" + std::to_string((unsigned long long)(i)) + " {" << std::endl;

                dotFile << "\t" << layout << std::endl;

                VisitGraph(graph, [&](DiGraphNode<int> *&node) -> void {
                    dotFile << "\t" << GetNodeName(node) << " [label=\"" << node->value << "\"]" << std::endl;

                    for (int j = 0; j < node->edges.size(); ++j) {
                        dotFile << "\t" << GetNodeName(node) << " -> " << GetNodeName(node->edges[j]) << std::endl;
                    }
                });

                dotFile << "}" << std::endl;

                dotFile.close();
            }

            std::system(("dot -Tpng \"" + dotFileName + ".dot\" > \"" + dotFileName + ".png\"").c_str());

            DestructGraph(graph);
        }
    }
};

template<DiGraphNode<int>*(*T)(const int)>
inline void RunFileBenchmarks(const std::string &outputName, const int numRuns, const int numInstances, const MEL::Comm &comm) {
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);
    
    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "File Write Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // FileWrite Test                                                                                 //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {
            // Load the graph on the root process
            DiGraphNode<int> *graph = nullptr;
            if (rank == 0) {
                graph = T(1 << i);
            }

            double startTime, endTime, deltaTime;

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                RM("/tmp/csjoss/" + outputName + "-MEL-Deep-STLFile-" + std::to_string(static_cast<long long>(i)) + ".graph");

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream graphFile("/tmp/csjoss/" + outputName + "-MEL-Deep-STLFile-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::out | std::ios::binary);
                if (graphFile.is_open()) {
                    MEL::Deep::FileWrite(graph, graphFile);
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "MEL::Deep::STLFileWrite in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-NonBuffered-STLFileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                RM("/tmp/csjoss/" + outputName + "-MEL-Deep-STLBufferedFile-" + std::to_string(static_cast<long long>(i)) + ".graph");

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream graphFile("/tmp/csjoss/" + outputName + "-MEL-Deep-STLBufferedFile-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::out | std::ios::binary);
                if (graphFile.is_open()) {
                    MEL::Deep::BufferedFileWrite(graph, graphFile);
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "MEL::Deep::STLBufferedFileWrite in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-Buffered-STLFileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            DestructGraph(graph);
        }
    }

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "File Read Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // FileRead Test                                                                                 //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {
            DiGraphNode<int> *graph = nullptr;

            double startTime, endTime, deltaTime;
            
            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream graphFile("/tmp/csjoss/" + outputName + "-MEL-Deep-STLFile-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::in | std::ios::binary);
                if (graphFile.is_open()) {
                    MEL::Deep::FileRead(graph, graphFile);
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;

                // Clean up scene on root
                DestructGraph(graph);
            }

            RM("/tmp/csjoss/" + outputName + "-MEL-Deep-STLFile-" + std::to_string(static_cast<long long>(i)) + ".graph");

            std::cout << "MEL::Deep::STLFileRead in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-NonBuffered-STLFileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream graphFile("/tmp/csjoss/" + outputName + "-MEL-Deep-STLBufferedFile-" + std::to_string(static_cast<long long>(i)) + ".graph", std::ios::in | std::ios::binary);
                if (graphFile.is_open()) {
                    MEL::Deep::BufferedFileRead(graph, graphFile);
                    graphFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;

                // Clean up scene on root
                DestructGraph(graph);
            }

            RM("/tmp/csjoss/" + outputName + "-MEL-Deep-STLBufferedFile-" + std::to_string(static_cast<long long>(i)) + ".graph");

            std::cout << "MEL::Deep::STLBufferedFileRead in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-Buffered-STLFileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }
        }
    }

    RM("/tmp/csjoss/*.graph");
};

template<DiGraphNode<int>*(*T)(const int)>
inline void RunBcastBenchmarks(const std::string &outputName, const int numRuns, const int numInstances, const MEL::Comm &comm) {
    const int rank = MEL::CommRank(comm),
        size = MEL::CommSize(comm);

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "Bcast Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // Bcast Test                                                                                 //
    //-----------------------------------------------------------------------------------------------//

    for (int i = 0; i <= numInstances; ++i) {
        // Load the graph on the root process
        DiGraphNode<int> *graph = nullptr;
        if (rank == 0) {
            graph = T(1 << i);
        }

        double startTime, endTime, deltaTime;

        //---------------------------------------------------------------------------------------//
        deltaTime = 0.;
        for (int k = 0; k < numRuns; ++k) {
            MEL::Barrier(comm);
            startTime = MEL::Wtime(); // Start the clock!

            MEL::Deep::Bcast(graph, 0, comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) DestructGraph(graph);
        }

        if (rank == 0) {
            deltaTime /= (double) numRuns;
            std::cout << "MEL::Deep::Bcast in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            std::ofstream outFile(outputName + "-MEL-NonBuffered-Bcast-" + std::to_string(size) + ".time", std::ios::out | std::ios::app);
            if (outFile.is_open()) {
                if (i == 0) outFile.clear();
                outFile << i << " " << deltaTime << std::endl;
            }
        }

        //---------------------------------------------------------------------------------------//
        deltaTime = 0.;
        for (int k = 0; k < numRuns; ++k) {
            MEL::Barrier(comm);
            startTime = MEL::Wtime(); // Start the clock!

            MEL::Deep::BufferedBcast(graph, 0, comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) DestructGraph(graph);
        }

        if (rank == 0) {
            deltaTime /= (double) numRuns;
            std::cout << "MEL::Deep::BufferedBcast in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            std::ofstream outFile(outputName + "-MEL-Buffered-Bcast-" + std::to_string(size) + ".time", std::ios::out | std::ios::app);
            if (outFile.is_open()) {
                if (i == 0) outFile.clear();
                outFile << i << " " << deltaTime << std::endl;
            }
        }

        /*
        //---------------------------------------------------------------------------------------//
        for (int mag = 2; mag <= 4; ++mag) {
            for (int inc = 1; inc < 10; inc++) {
                
                const int bufferSize = std::pow(10., mag) * inc;
                
                deltaTime = 0.;
                for (int k = 0; k < numRuns; ++k) {
                    MEL::Barrier(comm);
                    startTime = MEL::Wtime(); // Start the clock!

                    MEL::Deep::BcastStream(graph, 0, comm, bufferSize);

                    MEL::Barrier(comm);
                    endTime = MEL::Wtime(); // Stop the clock!
                    deltaTime += endTime - startTime;

                    // Clean up for next test
                    if (rank != 0) DestructGraph(graph);
                }

                if (rank == 0) {
                    deltaTime /= (double) numRuns;
                    std::cout << "MEL::Deep::BcastStream in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
                    std::ofstream outFile(outputName + "-MEL-Bcast-Stream-" + std::to_string(size) + "-buffersize-" + std::to_string(bufferSize) + ".time", std::ios::out | std::ios::app);
                    if (outFile.is_open()) {
                        if (i == 0) outFile.clear();
                        outFile << i << " " << deltaTime << std::endl;
                    }
                }
            }
        }
        */
    }

    MEL::Barrier(comm);
};

int main(int argc, char *argv[]) {
    MEL::Init(argc, argv);

    MEL::Comm comm = MEL::Comm::WORLD;
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);

    std::cout << "Running Benchmarks on " << size << " nodes..." << std::endl;

    if (argc != 2) {
        if (rank == 0) std::cout << "Wrong number of parameters..." << std::endl;
        MEL::Exit(-2);
    }

    srand(time(nullptr));

    const int numRuns = 10;
    const int numInstances = std::stoi(argv[1]); // 2^n instances
    
    /*
    RunVerification<MakeBTreeGraph>							("BTree-Graph",					5, comm, "");
    RunVerification<MakeRingGraph>							("Ring-Graph",					5, comm, "layout=circo");
    RunVerification<MakeRandomGraph>						("Random-Graph",				5, comm, "layout=circo");
    RunVerification<MakeFullyConnectedGraph>				("Fully-Connected-Graph",		5, comm, "layout=circo");

#ifdef FILE_TEST
    RunBoostVerification<MakeBoostBTreeGraph>				("Boost-BTree-Graph",			5, comm, "");
    RunBoostVerification<MakeBoostRingGraph>				("Boost-Ring-Graph",			5, comm, "layout=circo");
    RunBoostVerification<MakeBoostRandomGraph>				("Boost-Random-Graph",			5, comm, "layout=circo");
    RunBoostVerification<MakeBoostFullyConnectedGraph>		("Boost-Fully-Connected-Graph",	5, comm, "layout=circo");
#endif
    */

#ifdef FILE_TEST
    RM("/tmp/csjoss/*.graph");

    RunFileBenchmarks<MakeBTreeGraph>						("Binary Tree Graph",			numRuns, numInstances, comm);
    RunFileBenchmarks<MakeRingGraph>						("Ring Graph",					numRuns, numInstances, comm);
    RunFileBenchmarks<MakeRandomGraph>						("Random Graph",				numRuns, numInstances, comm);
    RunFileBenchmarks<MakeFullyConnectedGraph>				("Fully Connected Graph",		numRuns, numInstances, comm);

    RunBoostFileBenchmarks<MakeBoostBTreeGraph>				("Binary Tree Graph",			numRuns, numInstances, comm);
    RunBoostFileBenchmarks<MakeBoostRingGraph>				("Ring Graph",					numRuns, numInstances, comm);
    RunBoostFileBenchmarks<MakeBoostRandomGraph>			("Random Graph",				numRuns, numInstances, comm);
    RunBoostFileBenchmarks<MakeBoostFullyConnectedGraph>	("Fully Connected Graph",		numRuns, numInstances, comm);
#else
    for (int i = 2; i <= size; i *= 2) {
        MEL::Comm subComm = MEL::CommSplit(comm, rank < i ? 0 : 1);
        
        if (rank < i) {
            RunBcastBenchmarks<MakeBTreeGraph>						("Binary Tree Graph",			numRuns, numInstances, subComm);
            RunBcastBenchmarks<MakeRingGraph>						("Ring Graph",					numRuns, numInstances, subComm);
            RunBcastBenchmarks<MakeRandomGraph>						("Random Graph",				numRuns, numInstances, subComm);
            RunBcastBenchmarks<MakeFullyConnectedGraph>				("Fully Connected Graph",		numRuns, numInstances, subComm);
        }

        MEL::CommFree(subComm);
    }
#endif

    MEL::Finalize();
    return 0;
};