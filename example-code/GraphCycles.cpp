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

template<typename T>
struct DiGraphNode {
	/// Members
	T value;
	std::vector<DiGraphNode<T>*> edges;

	DiGraphNode() {};
	DiGraphNode(const T &_v) : value(_v) {};

	inline int getOutDegree() const {
		return (int) edges.size();
	};

	inline void addEdge(DiGraphNode<T> *node) {
		edges.push_back(node);
	};

    template<typename MSG>
	inline void DeepCopy(MSG &msg) {
		msg & edges;

		for (auto it = edges.begin(); it != edges.end(); ++it) {
			msg.packSharedPtr(*it);
		}
	};
};

template<typename T>
struct DiGraph {
	/// Members
	std::vector<DiGraphNode<T>*> nodes;

	DiGraph() {};
	~DiGraph() {
		for (auto it = nodes.begin(); it != nodes.end(); ++it) {
			MEL::MemDestruct(*it);
		}
	};

	inline void addNode(const T &value) {
		nodes.push_back(MEL::MemConstruct<DiGraphNode<T>>(value));
	};

	inline DiGraphNode<T>* getNode(const int nodeId) const {
		return nodes[nodeId];
	};

    template<typename MSG>
	inline void DeepCopy(MSG &msg) {
		msg & nodes;

		for (auto it = nodes.begin(); it != nodes.end(); ++it) {
			msg.packSharedPtr(*it);
		}
	};

	inline void visit(const int startNodeId, std::function<void(const T &)> func) const {
		std::unordered_set<const DiGraphNode<T>*> visited;
		std::stack<const DiGraphNode<T>*>		  nodeStack;

		nodeStack.push(getNode(startNodeId));

		while (!nodeStack.empty()) {
			const DiGraphNode<T> *currentNode = nodeStack.top();
			nodeStack.pop();

			if (visited.find(currentNode) != visited.end()) continue;

			visited.insert(currentNode);
			func(currentNode->value);

			for (auto it = currentNode->edges.rbegin(); it != currentNode->edges.rend(); ++it) {
				nodeStack.push(it->node);
			}
		}
	};
};

inline DiGraph<int>* MakeBTreeGraph(const int numNodes) {
	/// Ring Graph
	DiGraph<int>* graph = MEL::MemConstruct<DiGraph<int>>();

	/// Create graph nodes
	for (int i = 0; i < numNodes; ++i) {
		graph->addNode(i);
	}

	if (numNodes > 1) graph->getNode(0)->addEdge(graph->getNode(1));

	for (int i = 1; i < numNodes; ++i) {
		const int j = ((i - 1) * 2) + 2;

		if (j < numNodes)		graph->getNode(i)->addEdge(graph->getNode(j));
		if ((j + 1) < numNodes) graph->getNode(i)->addEdge(graph->getNode(j + 1));
	}
	return graph;
};

inline DiGraph<int>* MakeRingGraph(const int numNodes) {
	/// Ring Graph
	DiGraph<int>* graph = MEL::MemConstruct<DiGraph<int>>();

	/// Create graph nodes
	for (int i = 0; i < numNodes; ++i) {
		graph->addNode(i);
	}

	/// Connect each node to its right neighbour
	for (int i = 0; i < numNodes; ++i) {
		graph->getNode(i)->addEdge(graph->getNode((i + 1) % numNodes));
	}
	return graph;
};

inline DiGraph<int>* MakeRandomGraph(const int numNodes) {
	srand(1234567);

	/// Random Graph
	auto graph = MEL::MemConstruct<DiGraph<int>>();

	/// Create graph nodes
	for (int i = 0; i < numNodes; ++i) {
		graph->addNode(i);
	}

	/// Connect each node to a random set of neighbours
	for (int i = 0; i < numNodes; ++i) {
		const int numEdges = rand() % numNodes;

		DiGraphNode<int> *node = graph->getNode(i);
		for (int j = 0; j < numEdges; ++j) {
			node->addEdge(graph->getNode(rand() % numNodes));
		}
	}
	return graph;
};

inline DiGraph<int>* MakeFullyConnectedGraph(const int numNodes) {
	/// Fully Connected Graph
	DiGraph<int>* graph = MEL::MemConstruct<DiGraph<int>>();

	/// Create graph nodes
    std::cout << "Adding nodes..." << std::endl;

	for (int i = 0; i < numNodes; ++i) {
		graph->addNode(i);
	}

	/// Connect each node to all neighbours
    std::cout << "Linking nodes..." << std::endl;

	for (int i = 0; i < numNodes; ++i) {
		DiGraphNode<int> *node = graph->getNode(i);
		for (int j = 0; j < numNodes; ++j) {
			node->addEdge(graph->getNode(j));
		}
	}

    std::cout << "Done..." << std::endl;
	return graph;
};

//------------------------------------------------------------------------------------------------------------//
// Example Usage: mpirun -n [number of processes] ./GraphCycles [num nodes: 0 <= n] [graph type: 0 <= t <= 3] //
//                mpirun -n 8 ./GraphCycles 11 0                                                              //
//------------------------------------------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
	MEL::Init(argc, argv);

	MEL::Comm comm = MEL::Comm::WORLD;
	const int rank = MEL::CommRank(comm),
		      size = MEL::CommSize(comm);

    //TEST_STREAMS();

    if (argc != 3) {
		if (rank == 0)
			std::cout << "Wrong number of parameters..." << std::endl;
		MEL::Exit(-1);
	}

	const int numNodes = 1 << std::stoi(argv[1]), // 2^n nodes
		     graphType = std::stoi(argv[2]);

	DiGraph<int> *graph = nullptr;
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
    MEL::Deep::BcastStream(graph, 0, comm, 4 * 1024 * 1024);

	MEL::Barrier(comm);
	auto endTime = MEL::Wtime(); // Stop the clock!

	if (rank == 0) {
		std::cout << "Broadcast Graph in " << (endTime - startTime) << " seconds..." << std::endl;
	}

	// File name for output
	std::stringstream sstr;
	sstr << "rank=" << rank << " type=" << graphType << " nodes=" << numNodes << ".graph";

	// Save the output to disk from each node
	std::ofstream graphFile(sstr.str(), std::ios::out | std::ios::binary);
	if (graphFile.is_open()) {
		MEL::Deep::FileWrite(graph, graphFile);
		graphFile.close();
	}

	MEL::MemDestruct(graph);

	MEL::Finalize();
	return 0;
}