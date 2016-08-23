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
#include "MEL_deepcopy_experimental.hpp"

#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <stack>
#include <unordered_set>

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
        nodes[i]->edges.reserve(1);
        nodes[i]->edges.push_back(nodes[(i + 1) % numNodes]);
    }
    return nodes[0];
};

inline DiGraphNode<int>* MakeRandomGraph(const int numNodes) {
    srand(1234567);

    /// Random Graph
    std::vector<DiGraphNode<int>*> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i] = MEL::MemConstruct<DiGraphNode<int>>(i);
    }

    for (int i = 0; i < numNodes; ++i) {
        const int numEdges = rand() % numNodes;
        nodes[i]->edges.reserve(numEdges);
        for (int j = 0; j < numEdges; ++j) {
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

inline void DestructGraph(DiGraphNode<int> *&root) {
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
            MEL::MemDestruct(node);
        }
    }
};

int main(int argc, char *argv[]) {
    MEL::Init(argc, argv);

    MEL::Comm comm = MEL::Comm::WORLD;
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);

    if (argc != 3) {
        if (rank == 0)
            std::cout << "Wrong number of parameters..." << std::endl;
        MEL::Exit(-1);
    }

    const int numNodes  = 1 << std::stoi(argv[1]), // 2^n nodes
              graphType = std::stoi(argv[2]);


    DiGraphNode<int> *graph = nullptr;
    if (rank == 0) {
        switch (graphType) {
        case 0:
            graph = MakeBTreeGraph(numNodes);
            break;
        case 1:
            graph = MakeRingGraph(numNodes);
            break;
        case 2:
            graph = MakeRandomGraph(numNodes);
            break;
        case 3:
            graph = MakeFullyConnectedGraph(numNodes);
            break;
        }

        std::cout << "Graph size " << MEL::Deep::BufferSize(graph) << " bytes..." << std::endl;
    }

    MEL::Barrier(comm);
    auto startTime = MEL::Wtime(); // Start the clock!

    // Deep copy the graph to all nodes
    MEL::Deep::Bcast(graph, 0, comm);

    MEL::Barrier(comm);
    auto endTime = MEL::Wtime(); // Stop the clock!

    if (rank == 0) {
        std::cout << "Broadcast Graph in " << (endTime - startTime) << " seconds..." << std::endl;
    }

    MEL::Barrier(comm);

    // File name for output
    std::stringstream sstr;
    sstr << "rank=" << rank << " type=" << graphType << " nodes=" << numNodes << ".graph";

    // Save the output to disk from each node
    std::ofstream graphFile(sstr.str(), std::ios::out | std::ios::binary);
    if (graphFile.is_open()) {
        MEL::Deep::FileWrite(graph, graphFile);
        graphFile.close();
    }

    MEL::Barrier(comm);

    DestructGraph(graph);

    if (rank == 0) std::cout << "Done." << std::endl;

    MEL::Finalize();
    return 0;
}