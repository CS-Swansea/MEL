#define  MEL_IMPLEMENTATION
#include "MEL.hpp"
#include "MEL_deepcopy.hpp"

#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <stack>

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

constexpr double INF = 1e9;

struct Vec {
    double x, y, z;

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned version) {
        ar & x & y & z;
    };
#endif

    inline Vec operator+(const Vec &rhs) const { return{ x + rhs.x, y + rhs.y, z + rhs.z }; };
    inline Vec operator-(const Vec &rhs) const { return{ x - rhs.x, y - rhs.y, z - rhs.z }; };
    inline Vec operator*(const Vec &rhs) const { return{ x * rhs.x, y * rhs.y, z * rhs.z }; };
    inline Vec operator/(const Vec &rhs) const { return{ x / rhs.x, y / rhs.y, z / rhs.z }; };
    inline Vec operator+(const double rhs) const { return{ x + rhs, y + rhs, z + rhs }; };
    inline Vec operator*(const double rhs) const { return{ x * rhs, y * rhs, z * rhs }; };
    inline Vec operator/(const double rhs) const { double d = 1.0 / rhs; return{ x * d, y * d, z * d }; };
    inline friend Vec operator/(const double lhs, const Vec &rhs) { return{ lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; };

    /// Length & Magnitude
    inline double length() const { return std::sqrt((x * x) + (y * y) + (z * z)); };
    inline double length2() const { return (x * x) + (y * y) + (z * z); };
    inline Vec inv() const { return{ 1.0 / x, 1.0 / y, 1.0 / z }; };
    inline Vec normal() const {
        const double e = ((x * x) + (y * y) + (z * z));
        const double d = 1.0 / std::sqrt(e);
        return{ x * d, y * d, z * d };
    };

    /// Dot & Cross Products
    inline double operator|(const Vec &rhs) const { return (x * rhs.x) + (y * rhs.y) + (z * rhs.z); };
    inline Vec operator^(const Vec &rhs) const { return{ y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x }; };

    /// Min/Max Channels
    inline double min() const { return std::fmin(x, std::fmin(y, z)); };
    inline double max() const { return std::fmax(x, std::fmax(y, z)); };
    inline Vec min(const Vec &b) const { return{ std::fmin(x, b.x), std::fmin(y, b.y), std::fmin(z, b.z) }; };
    inline Vec max(const Vec &b) const { return{ std::fmax(x, b.x), std::fmax(y, b.y), std::fmax(z, b.z) }; };
    inline int maxAxis() const { return (x > y && x > z) ? 0 : (y > z) ? 1 : 2; };
};

struct Triangle {
    Vec v0, v1, v2;
    int material;

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned version) {
        ar & v0 & v1 & v2 & material;
    };
#endif

    inline Vec min() const { return v0.min(v1.min(v2)); };
    inline Vec max() const { return v0.max(v1.max(v2)); };
    inline Vec centroid() const { return (max() - min()) * .5; };

    //inline bool intersect(const Ray &ray, Intersection &isect) const;
};

struct Material {
    Vec kd, ke;

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned version) {
        ar & kd & ke;
    };
#endif  
};

struct Camera {
    Vec		pos, dir, u, v;
    int		w, h;

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned version) {
        ar & pos & dir & u & v & w & h;
    };
#endif
    
    Camera() {};
    Camera(Vec _p, Vec _d, double _f, int _w, int _h)
        : pos(_p), dir(_d), w(_w), h(_h) {
        double fw = (2.0 * std::tan((_f * 0.0174532925) / 2.0));
        u = dir ^ Vec{ 0, 1, 0 };
        v = u   ^ dir;
        u = u * fw;
        v = v * fw;
    };

    //inline Ray getRay(const double x, const double y) const;
};

struct TreeNode {
    int			startElem, endElem;
    Vec			v0, v1;
    TreeNode	*leftChild, *rightChild;

    TreeNode() : startElem(0), endElem(0), leftChild(nullptr), rightChild(nullptr), v0{ INF, INF, INF }, v1{ -INF, -INF, -INF } {};
    TreeNode(const int _s, const int _e) : startElem(_s), endElem(_e), leftChild(nullptr), rightChild(nullptr), v0{ INF, INF, INF }, v1{ -INF, -INF, -INF } {};
    ~TreeNode() {
        MEL::MemDestruct(leftChild);
        MEL::MemDestruct(rightChild);
    };

    //inline bool intersect(const Ray &rayInv, double &tmin, const double dist) const;

    template<typename MSG>
    inline void DeepCopy(MSG &msg) {
        msg.packPtr( leftChild);
        msg.packPtr(rightChild);
    };

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void save(Archive& ar, const unsigned version) const {
        ar & startElem & endElem;
        ar & v0 & v1;

        bool hasLeftTree = leftChild != nullptr;
        bool hasRightTree = rightChild != nullptr;
        ar & hasLeftTree & hasRightTree;

        if (hasLeftTree)  ar & *leftChild;
        if (hasRightTree) ar & *rightChild;
    };

    template<class Archive>
    void load(Archive& ar, const unsigned version) {
        ar & startElem & endElem;
        ar & v0 & v1;

        bool hasLeftTree, hasRightTree;
        ar & hasLeftTree & hasRightTree;

        if (hasLeftTree) {
            MEL::MemDestruct(leftChild);
            leftChild = MEL::MemAlloc<TreeNode>(1, TreeNode());
            ar & *leftChild;
        }
        if (hasRightTree) {
            MEL::MemDestruct(rightChild);
            rightChild = MEL::MemAlloc<TreeNode>(1, TreeNode());
            ar & *rightChild;
        }
    };

    BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif
};

struct Scene {
    std::vector<Material> materials;
    std::vector<Triangle> mesh;
    TreeNode *rootNode;
    Camera camera;

    Scene() : rootNode(nullptr) {};
    Scene(const Scene &old) = delete;					// Remove CopyConstructor
    inline Scene& operator=(const Scene &old) = delete; // Remove CopyAssignment
    Scene(Scene &&old) : mesh(std::move(old.mesh)), materials(std::move(old.materials)),
        camera(old.camera), rootNode(old.rootNode) {
        old.mesh.clear();
        old.materials.clear();
        old.rootNode = nullptr;
    };
    inline Scene& operator=(Scene &&old) {
        mesh = std::move(old.mesh);
        materials = std::move(old.materials);
        rootNode = old.rootNode;
        camera = old.camera;
        old.mesh.clear();
        old.materials.clear();
        old.rootNode = nullptr;
        return *this;
    };
    ~Scene() {
        MEL::MemDestruct(rootNode);
    };

    //inline bool intersect(const Ray &ray, Intersection &isect) const;

    /// MEL Deep Copy
    template<typename MSG>
    inline void DeepCopy(MSG &msg) {
        msg & mesh & materials;
        msg.packPtr(rootNode);
    };

#ifdef FILE_TEST
    /// Boost Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void save(Archive& ar, const unsigned version) const {
        ar & mesh;
        ar & materials;
        ar & camera;

        bool hasTree = rootNode != nullptr;
        ar & hasTree;

        if (hasTree) ar & *rootNode;
    };

    template<class Archive>
    void load(Archive& ar, const unsigned version) {
        ar & mesh;
        ar & materials;
        ar & camera;

        bool hasTree;
        ar & hasTree;

        if (hasTree) {
            MEL::MemDestruct(rootNode);
            rootNode = MEL::MemAlloc<TreeNode>(1, TreeNode());
            ar & *rootNode;
        }
    };

    BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif   
};

inline Scene* loadScene(const std::string &meshPath, const int instances) {
    Scene *scene = MEL::MemConstruct<Scene>();

    /// Helpers
    auto hasPrefix = [](const std::string &str, const std::string &prefix) -> bool {
        return (str.size() >= prefix.size()) && (std::mismatch(prefix.begin(), prefix.end(), str.begin()).first == prefix.end());
    };
    auto split = [](const std::string &str, const char delim) -> std::vector<std::string> {
        std::vector<std::string> result; result.reserve(16);
        size_t start = 0, end;
        while ((end = str.find(delim, start)) != std::string::npos) {
            if (str[start] != ' ') result.push_back(str.substr(start, (end - start)));
            start = end + 1;
        }
        result.push_back(str.substr(start));
        return result;
    };

    /// Load .obj file
    std::vector<Vec> vertices;
    std::vector<Triangle> mesh;
    std::ifstream meshFile(meshPath, std::ios::in);
    if (meshFile.is_open()) {

        std::string meshLine;
        while (std::getline(meshFile, meshLine)) {
            /// Vertex Declaration
            if (hasPrefix(meshLine, "v ")) {
                std::vector<std::string> sm = split(meshLine, ' ');
                Vec v{ std::stod(sm[1]), std::stod(sm[2]), std::stod(sm[3]) };
                vertices.push_back(v);
                continue;
            }

            /// Face Declaration
            if (hasPrefix(meshLine, "f ")) {
                std::vector<std::string> fm = split(meshLine, ' '), sm;
                sm = split(fm[1], '/'); int v0 = std::stoi(sm[0]);
                sm = split(fm[2], '/'); int v1 = std::stoi(sm[0]);
                sm = split(fm[3], '/'); int v2 = std::stoi(sm[0]);
                Vec V0{}, V1{}, V2{};
                V0 = (v0 > 0) ? vertices[v0 - 1] : (v0 < 0) ? vertices[vertices.size() + v0] : Vec{};
                V1 = (v1 > 0) ? vertices[v1 - 1] : (v1 < 0) ? vertices[vertices.size() + v1] : Vec{};
                V2 = (v2 > 0) ? vertices[v2 - 1] : (v2 < 0) ? vertices[vertices.size() + v2] : Vec{};
                mesh.push_back(Triangle{ V0, V1, V2, 0 });
                continue;
            }
        }

        meshFile.close();
        std::cout << "Successfully loaded: " << meshPath << std::endl;
    }
    else {
        std::cout << "Error loading: " << meshPath << std::endl;
        std::exit(-1);
    }

    /// Instance the mesh n times in random locations within a cube
    srand(12345);
    auto rng = []() -> double {
        return ((double) rand() / (double) INT_MAX);
    };

    const int meshSize = (int) mesh.size();
    scene->mesh.resize(instances * meshSize);
    for (int i = 0; i < instances; ++i) {
        Vec r{ ((rng() - .5) * 1000.), ((rng()) * 1000.), ((rng() - .5) * 1000.) };

        for (int j = 0; j < meshSize; ++j) {
            scene->mesh[j + (i * meshSize)] = Triangle{ mesh[j].v0, mesh[j].v1, mesh[j].v2, mesh[j].material };
        }
    }

    auto start = MEL::Wtime();

    // Create root node
    int numNodes = 1;
    scene->rootNode = MEL::MemConstruct<TreeNode>(0, scene->mesh.size());

    // Stack based traversal
    // We track tree nodes and their depth in the tree
    std::stack<TreeNode*> treeStack;
    treeStack.push(scene->rootNode); // <- Depth 0

    // While the stack is not empty there is work to be done
    while (!treeStack.empty()) {
        // Get the current node to traverse
        TreeNode *currentNode = treeStack.top();
        treeStack.pop();

        const int numGeom = currentNode->endElem - currentNode->startElem;

        // Compute the nodes bounding box
        Vec b0{ INF, INF, INF }, b1{ -INF, -INF, -INF };
        for (int i = currentNode->startElem; i < currentNode->endElem; ++i) {
            currentNode->v0 = currentNode->v0.min(scene->mesh[i].min());
            currentNode->v1 = currentNode->v1.max(scene->mesh[i].max());
            const Vec c = scene->mesh[i].centroid();
            b0 = b0.min(c); b1 = b1.max(c);
        }

        // Is it worth splitting?
        if (numGeom <= 1) {
            continue;
        }

        // Mid index for partitioning
        int midElem = -1;
        typename std::vector<Triangle>::iterator start = scene->mesh.begin() + currentNode->startElem,
            end = scene->mesh.begin() + currentNode->endElem,
            mid;

        midElem = (currentNode->startElem + (numGeom / 2));
        mid = scene->mesh.begin() + midElem;

        // Sort the tree such that all nodes before the split plane are in the first half of the vector
        const int splitAxis = (b1 - b0).maxAxis();
        std::nth_element(start, mid, end, [&](const Triangle &a, const Triangle &b) -> bool {
            const Vec ac = a.centroid(), bc = b.centroid();
            switch (splitAxis) {
            case 0:
                return ac.x < bc.x;
            case 1:
                return ac.y < bc.y;
            case 2:
                return ac.z < bc.z;
            };
            return false;
        });

        // Create child nodes based on partition
        numNodes += 2;
        currentNode->leftChild  = MEL::MemConstruct<TreeNode>(currentNode->startElem, midElem);
        currentNode->rightChild = MEL::MemConstruct<TreeNode>(midElem, currentNode->endElem);

        // Push new nodes onto the working stack
        treeStack.push(currentNode->rightChild);
        treeStack.push(currentNode->leftChild);
    }
    auto end = MEL::Wtime();
    std::cout << "BVH Tree constructed of ( " << numNodes << " ) nodes in " << std::setprecision(4) << (end - start) << "s" << std::endl;

    return scene;
};

inline void MPI_NonBufferedBcast_Scene(Scene *&scene, const int rank, const int root, const MPI_Comm comm) {
    /// Receiving nodes allocate space for scene
    if (rank != root) {
        MPI_Alloc_mem(sizeof(Scene), MPI_INFO_NULL, &scene);
        new (scene) Scene();
    }

    /// Bcast the camera struct
    MPI_Bcast(&(scene->camera), sizeof(Camera), MPI_CHAR, root, comm);

    /// Bcast the vector sizes
    int sizes[2];
    if (rank == root) {
        sizes[0] = (int) scene->mesh.size();
        sizes[1] = (int) scene->materials.size();
    }
    MPI_Bcast(sizes, 2, MPI_INT, root, comm);

    /// 'Allocate' space for vectors
    if (rank != root) {
        scene->mesh.resize(sizes[0]);
        scene->materials.resize(sizes[1]);
    }

    /// Bcast the vectors
    MPI_Bcast(&(scene->mesh[0]), sizeof(Triangle) * sizes[0], MPI_CHAR, root, comm);
    MPI_Bcast(&(scene->materials[0]), sizeof(Material) * sizes[1], MPI_CHAR, root, comm);

    /// Receiving nodes allocate space for rootNode
    if (rank != root) {
        MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(scene->rootNode));
        new (scene->rootNode) TreeNode();
    }

    // While the stack is not empty there is work to be done
    std::stack<TreeNode*> treeStack;
    treeStack.push(scene->rootNode);
    while (!treeStack.empty()) {
        // Get the current node to traverse
        TreeNode *currentNode = treeStack.top();
        treeStack.pop();

        /// Bcast the current node's values
        MPI_Bcast((currentNode), sizeof(TreeNode), MPI_CHAR, root, comm);

        /// Do we need to send children?
        bool hasChildren = (currentNode->leftChild != nullptr);

        if (hasChildren) {
            /// Allocate space for child nodes on receiving process
            if (rank != root) {
                MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(currentNode->leftChild));
                MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(currentNode->rightChild));
                new (currentNode->leftChild)  TreeNode();
                new (currentNode->rightChild) TreeNode();
            }

            /// Push children onto the stack so they get processed
            treeStack.push(currentNode->leftChild);
            treeStack.push(currentNode->rightChild);
        }
    }
};

inline void MPI_BufferedBcast_Scene(Scene *&scene, const int rank, const int root, const MPI_Comm comm) {
    /// Receiving nodes allocate space for scene
    if (rank != root) {
        MPI_Alloc_mem(sizeof(Scene), MPI_INFO_NULL, &scene);
        new (scene) Scene();
    }

    /// Calculate the byte size of the tree
    int packed_size = 0;
    if (rank == root) {
        packed_size += sizeof(Camera);
        packed_size += sizeof(int) +((int) scene->mesh.size()        * sizeof(Triangle));
        packed_size += sizeof(int) +((int) scene->materials.size()    * sizeof(Material));

        // While the stack is not empty there is work to be done
        std::stack<TreeNode*> treeStack;
        treeStack.push(scene->rootNode);
        while (!treeStack.empty()) {
            // Get the current node to traverse
            TreeNode *currentNode = treeStack.top();
            treeStack.pop();

            packed_size += sizeof(TreeNode);

            /// Do we need to send children?
            bool hasChildren = (currentNode->leftChild != nullptr);

            if (hasChildren) {
                /// Push children onto the stack so they get processed
                treeStack.push(currentNode->leftChild);
                treeStack.push(currentNode->rightChild);
            }
        }
    }

    /// Share the buffer size
    MPI_Bcast(&packed_size, 1, MPI_INT, root, comm);

    /// Allocate the buffer
    int position = 0;
    char *buffer;
    MPI_Alloc_mem(packed_size, MPI_INFO_NULL, &(buffer));

    if (rank == root) {
        /// Pack the camera struct
        MPI_Pack(&(scene->camera), sizeof(Camera), MPI_CHAR, buffer, packed_size, &position, comm);

        int mesh_size = scene->mesh.size(),
            materials_size = scene->materials.size();

        /// Pack the mesh vector
        MPI_Pack(&(mesh_size), 1, MPI_INT, buffer, packed_size, &position, comm);
        MPI_Pack(&(scene->mesh[0]), mesh_size * sizeof(Triangle), MPI_CHAR, buffer, packed_size, &position, comm);

        /// Pack the materials vector
        MPI_Pack(&(materials_size), 1, MPI_INT, buffer, packed_size, &position, comm);
        MPI_Pack(&(scene->materials[0]), materials_size * sizeof(Material), MPI_CHAR, buffer, packed_size, &position, comm);

        // While the stack is not empty there is work to be done
        std::stack<TreeNode*> treeStack;
        treeStack.push(scene->rootNode);
        while (!treeStack.empty()) {
            // Get the current node to traverse
            TreeNode *currentNode = treeStack.top();
            treeStack.pop();

            /// Pack the current node
            MPI_Pack(currentNode, sizeof(TreeNode), MPI_CHAR, buffer, packed_size, &position, comm);

            /// Do we need to send children?
            bool hasChildren = (currentNode->leftChild != nullptr);

            if (hasChildren) {
                /// Push children onto the stack so they get processed
                treeStack.push(currentNode->leftChild);
                treeStack.push(currentNode->rightChild);
            }
        }

        /// Send the buffer
        MPI_Bcast(buffer, packed_size, MPI_CHAR, root, comm);
    }
    else {
        /// Receive the packed buffer
        MPI_Bcast(buffer, packed_size, MPI_CHAR, root, comm);

        MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(scene->rootNode));
        new (scene->rootNode) TreeNode();

        /// Unpack the camera struct
        int mesh_size, materials_size;
        MPI_Unpack(buffer, packed_size, &position, &(scene->camera), sizeof(Camera), MPI_CHAR, comm);

        /// Unpack mesh vector
        MPI_Unpack(buffer, packed_size, &position, &(mesh_size), 1, MPI_INT, comm);
        scene->mesh.resize(mesh_size);
        MPI_Unpack(buffer, packed_size, &position, &(scene->mesh[0]), mesh_size * sizeof(Triangle), MPI_CHAR, comm);

        /// Unpack materials vector
        MPI_Unpack(buffer, packed_size, &position, &(materials_size), 1, MPI_INT, comm);
        scene->materials.resize(materials_size);
        MPI_Unpack(buffer, packed_size, &position, &(scene->materials[0]), materials_size * sizeof(Material), MPI_CHAR, comm);

        // While the stack is not empty there is work to be done
        std::stack<TreeNode*> treeStack;
        treeStack.push(scene->rootNode);
        while (!treeStack.empty()) {
            // Get the current node to traverse
            TreeNode *currentNode = treeStack.top();
            treeStack.pop();

            /// Unpack the current node
            MPI_Unpack(buffer, packed_size, &position, currentNode, sizeof(TreeNode), MPI_CHAR, comm);

            /// Do we need to send children?
            bool hasChildren = (currentNode->leftChild != nullptr);

            if (hasChildren) {
                /// Allocate space for child nodes on receiving process
                MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(currentNode->leftChild));
                MPI_Alloc_mem(sizeof(TreeNode), MPI_INFO_NULL, &(currentNode->rightChild));
                new (currentNode->leftChild)  TreeNode();
                new (currentNode->rightChild) TreeNode();

                /// Push children onto the stack so they get processed
                treeStack.push(currentNode->leftChild);
                treeStack.push(currentNode->rightChild);
            }
        }
    }

    /// Clean up
    MPI_Free_mem(buffer);
};

inline void RunBcastBenchmarks(const std::string &outputName, const std::string &meshPath, const int numRuns, const int numInstances, const MEL::Comm &comm) {
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "Bcast Test - nodes: " << size << " instances: " << numInstances << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // Bcast Test                                                                                    //
    //-----------------------------------------------------------------------------------------------//
    for (int i = 0; i <= numInstances; ++i) {
        // Load the scene on the root process
        Scene *scene = nullptr;
        if (rank == 0) {
            std::cout << "Loading scene..." << std::endl;
            scene = loadScene(meshPath, 1 << i); // load 2^n meshes
        }

        double startTime, endTime, deltaTime;

        //---------------------------------------------------------------------------------------//
        deltaTime = 0.;
        for (int k = 0; k < numRuns; ++k) {
            MEL::Barrier(comm);
            startTime = MEL::Wtime(); // Start the clock!

            MEL::Deep::Bcast(scene, 0, comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) MEL::MemDestruct(scene);
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

            MEL::Deep::BufferedBcast(scene, 0, comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) MEL::MemDestruct(scene);
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

        //---------------------------------------------------------------------------------------//
        deltaTime = 0.;
        for (int k = 0; k < numRuns; ++k) {
            MEL::Barrier(comm);
            startTime = MEL::Wtime(); // Start the clock!

            MPI_NonBufferedBcast_Scene(scene, rank, 0, (MPI_Comm) comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) MEL::MemDestruct(scene);
        }

        if (rank == 0) {
            deltaTime /= (double) numRuns;
            std::cout << "MPI_NonBufferedBcast_Scene in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            std::ofstream outFile(outputName + "-MPI-NonBuffered-Bcast-" + std::to_string(size) + ".time", std::ios::out | std::ios::app);
            if (outFile.is_open()) {
                if (i == 0) outFile.clear();
                outFile << i << " " << deltaTime << std::endl;
            }

            // Clean up for next test
            if (rank != 0) MEL::MemDestruct(scene);
        }

        //---------------------------------------------------------------------------------------//
        deltaTime = 0.;
        for (int k = 0; k < numRuns; ++k) {
            MEL::Barrier(comm);
            startTime = MEL::Wtime(); // Start the clock!

            MPI_BufferedBcast_Scene(scene, rank, 0, (MPI_Comm) comm);

            MEL::Barrier(comm);
            endTime = MEL::Wtime(); // Stop the clock!
            deltaTime += endTime - startTime;

            // Clean up for next test
            if (rank != 0) MEL::MemDestruct(scene);
        }

        if (rank == 0) {
            deltaTime /= (double) numRuns;
            std::cout << "MPI_BufferedBcast_Scene in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            std::ofstream outFile(outputName + "-MPI-Buffered-Bcast-" + std::to_string(size) + ".time", std::ios::out | std::ios::app);
            if (outFile.is_open()) {
                if (i == 0) outFile.clear();
                outFile << i << " " << deltaTime << std::endl;
            }
        }

        // Clean up scene on root
        MEL::MemDestruct(scene);
    }

    MEL::Barrier(comm);
};

#ifdef FILE_TEST
inline void RunFileBenchmarks(const std::string &outputName, const std::string &meshPath, const int numRuns, const int numInstances, const MEL::Comm &comm) {
    const int rank = MEL::CommRank(comm),
        size = MEL::CommSize(comm);

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "File Write Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // FileWrite Test                                                                                //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {
            // Load the scene on the root process
            Scene *scene = nullptr;
            std::cout << "Loading scene..." << std::endl;
            scene = loadScene(meshPath, 1 << i); // load 2^n meshes

            double startTime, endTime, deltaTime;
            //MEL::File treeFile;

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                std::system(("rm /tmp/csjoss/MEL-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree").c_str());

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream stdFile("/tmp/csjoss/MEL-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::out | std::ios::binary);
                if (stdFile.is_open()) {
                    MEL::Deep::FileWrite(scene, stdFile);
                    stdFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "MEL::Deep::FileWrite in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-NonBuffered-FileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                std::system(("rm /tmp/csjoss/MEL-Deep-BufferedFile-" + std::to_string(static_cast<long long>(i)) + ".tree").c_str());

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream stdFile("/tmp/csjoss/MEL-Deep-BufferedFile-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::out | std::ios::binary);
                if (stdFile.is_open()) {
                    MEL::Deep::BufferedFileWrite(scene, stdFile);
                    stdFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "MEL::Deep::BufferedFileWrite in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-Buffered-FileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                std::system(("rm /tmp/csjoss/Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree").c_str());

                startTime = MEL::Wtime(); // Start the clock!

                std::ofstream stdTreeFile("/tmp/csjoss/Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::out | std::ios::binary);
                if (stdTreeFile.is_open()) {
                    boost::archive::binary_oarchive oa(stdTreeFile);
                    oa << *scene;
                    stdTreeFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;
            }

            std::cout << "Boost binary_oarchive in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-Boost-Deep-FileWrite.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            // Clean up scene on root
            MEL::MemDestruct(scene);
        }
    }

    MEL::Barrier(comm);
    if (rank == 0) std::cout << std::endl << "File Read Test" << std::endl;

    //-----------------------------------------------------------------------------------------------//
    // FileRead Test                                                                                 //
    //-----------------------------------------------------------------------------------------------//
    if (rank == 0) {
        for (int i = 0; i <= numInstances; ++i) {
            Scene *scene = nullptr;

            double startTime, endTime, deltaTime;
            //MEL::File treeFile;

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream stdFile("/tmp/csjoss/MEL-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::in | std::ios::binary);
                if (stdFile.is_open()) {
                    MEL::Deep::FileRead(scene, stdFile);
                    stdFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;

                // Clean up scene on root
                MEL::MemDestruct(scene);
            }

            std::cout << "MEL::Deep::FileRead in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-NonBuffered-FileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream stdFile("/tmp/csjoss/MEL-Deep-BufferedFile-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::in | std::ios::binary);
                if (stdFile.is_open()) {
                    MEL::Deep::BufferedFileRead(scene, stdFile);
                    stdFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;

                // Clean up scene on root
                MEL::MemDestruct(scene);
            }

            std::cout << "MEL::Deep::BufferedFileRead in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-MEL-Buffered-FileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }

            //---------------------------------------------------------------------------------------//
            deltaTime = 0.;
            for (int j = 0; j < numRuns; ++j) {
                startTime = MEL::Wtime(); // Start the clock!

                std::ifstream stdTreeFile("/tmp/csjoss/Boost-Deep-File-" + std::to_string(static_cast<long long>(i)) + ".tree", std::ios::in | std::ios::binary);
                if (stdTreeFile.is_open()) {
                    boost::archive::binary_iarchive ia(stdTreeFile);
                    scene = MEL::MemConstruct<Scene>();
                    ia >> *scene;
                    stdTreeFile.close();
                }

                endTime = MEL::Wtime(); // Stop the clock!
                deltaTime += (endTime - startTime) / (double) numRuns;

                // Clean up scene on root
                MEL::MemDestruct(scene);
            }

            std::cout << "Boost binary_iarchive in " << std::setprecision(30) << deltaTime << " seconds..." << std::endl;
            {
                std::ofstream outFile(outputName + "-Boost-Deep-FileRead.time", std::ios::out | std::ios::app);
                if (outFile.is_open()) {
                    if (i == 0) outFile.clear();
                    outFile << i << " " << deltaTime << std::endl;
                }
            }
        }
    }

    MEL::Barrier(comm);
};
#endif

int main(int argc, char *argv[]) {
    MEL::Init(argc, argv);

    MEL::Comm comm = MEL::Comm::WORLD;
    const int rank = MEL::CommRank(comm),
              size = MEL::CommSize(comm);

    if (argc != 3) {
        if (rank == 0) std::cout << "Wrong number of parameters..." << std::endl;
        MEL::Exit(-1);
    }

    const int         numRuns = 5;
    const std::string meshPath = std::string(argv[1]);
    const int		  numInstances = std::stoi(argv[2]); // 2^n instances
    
    MEL::Barrier(comm);

#ifdef FILE_TEST
    RunFileBenchmarks("Ray", meshPath, numRuns, numInstances, comm);
#else
    for (int i = 2; i <= size; i *= 2) {
        MEL::Comm subComm = MEL::CommSplit(comm, rank < i ? 0 : 1);

        if (rank < i) RunBcastBenchmarks("Ray", meshPath, numRuns, numInstances, subComm);

        MEL::CommFree(subComm);
    }
#endif

    MEL::Finalize();
    return 0;
};