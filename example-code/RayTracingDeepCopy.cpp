#define MEL_IMPLEMENTATION
#include "MEL.hpp"
#include "MEL_deepcopy.hpp"
#include <omp.h>

#include <ctime>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stack>
#include <algorithm>
#include <random>

/// Number of openmp threads
constexpr unsigned int NUM_THREADS = 16;

/// C++11 RNG
struct RNG {
	std::uniform_real_distribution<double> dist;
	std::mt19937 eng;
	RNG() : dist(0., 1.), eng() {};
	inline void seed(const unsigned long s) { eng.seed(s); };
	inline double operator()() { return dist(eng); };
};

/// C++11 Multi-Threaded RNG
struct MT_RNG {
	RNG rngs[NUM_THREADS]{};
	MT_RNG(const unsigned long s) {
		for (int i = 0; i < NUM_THREADS; ++i) rngs[i].seed(time(nullptr) + s + i);
	};
	inline double operator()() { return rngs[omp_get_thread_num()](); };
};

/// Colour Correction
inline unsigned char ColourCorrect(const double x) {
	auto Gamma_Uncharted = [](const double x) {
		constexpr double A = 0.15, B = 0.5, C = 0.1, D = 0.2, E = 0.02, F = 0.30;
		return ((x * (A * x + C * B) + D * E) / (x * (A * x * B) + D * F)) - E / F;
	};
	auto clamp = [](const double x) {
		return std::fmax(0., std::fmin(1., x));
	};
	constexpr double gamma = (1.0 / 2.2), exposure = 1.0, exposureBias = 2.0, whitePoint = 11.2;
	double y = clamp(std::pow((Gamma_Uncharted(x * exposure * exposureBias) / Gamma_Uncharted(whitePoint)), gamma));
	return (unsigned char) (y * 255.);
};

/// Ray Tracing Constants
constexpr double INF = 1e9, EPS = 0.000001;

struct Vec {
	double x, y, z;

	inline Vec operator+(const Vec &rhs) const { return{ x + rhs.x, y + rhs.y, z + rhs.z }; };
	inline Vec operator-(const Vec &rhs) const { return{ x - rhs.x, y - rhs.y, z - rhs.z }; };
	inline Vec operator*(const Vec &rhs) const { return{ x * rhs.x, y * rhs.y, z * rhs.z }; };
	inline Vec operator/(const Vec &rhs) const { return{ x / rhs.x, y / rhs.y, z / rhs.z }; };
	inline Vec operator+(const double rhs) const { return{ x + rhs, y + rhs, z + rhs }; };
	inline Vec operator*(const double rhs) const { return{ x * rhs, y * rhs, z * rhs }; };
	inline Vec operator/(const double rhs) const { double d = 1.0 / rhs; return{ x * d, y * d, z * d }; };
	inline friend Vec operator/(const double lhs, const Vec &rhs) { return{ lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; };

	inline Vec normal() const {
		const double e = ((x * x) + (y * y) + (z * z));
		const double d = 1.0 / std::sqrt(e);
		return{ x * d, y * d, z * d };
	};

	/// Length & Magnitude
	inline double length() const { return std::sqrt((x * x) + (y * y) + (z * z)); };
	inline double length2() const { return (x * x) + (y * y) + (z * z); };

	/// Dot & Cross Products
	inline double operator|(const Vec &rhs) const { return (x * rhs.x) + (y * rhs.y) + (z * rhs.z); };
	inline Vec operator^(const Vec &rhs) const { return{ y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x }; };
	inline Vec inv() const { return{ 1.0 / x, 1.0 / y, 1.0 / z }; };

	/// Min/Max Channels
	inline double min() const { return std::fmin(x, std::fmin(y, z)); };
	inline double max() const { return std::fmax(x, std::fmax(y, z)); };
	inline Vec min(const Vec &b) const { return{ std::fmin(x, b.x), std::fmin(y, b.y), std::fmin(z, b.z) }; };
	inline Vec max(const Vec &b) const { return{ std::fmax(x, b.x), std::fmax(y, b.y), std::fmax(z, b.z) }; };
	inline int maxAxis() const { return (x > y && x > z) ? 0 : (y > z) ? 1 : 2; };
};

struct Ray { Vec o, d; };

struct Intersection {
	double	distance;
	int		material;
	Vec		normal, pos;
	Intersection() : distance(INF) {};
};

struct Triangle {
	Vec v0, v1, v2;
	int material;

	inline Vec min() const { const Vec EPSV{ EPS, EPS, EPS }; return v0.min(v1.min(v2)) - EPSV; };
	inline Vec max() const { const Vec EPSV{ EPS, EPS, EPS }; return v0.max(v1.max(v2)) - EPSV; };
	inline Vec centroid() const { return (max() - min()) * .5; };

	inline bool intersect(const Ray &ray, Intersection &isect) const {
		/// Möller–Trumbore Ray-Triangle intersection algorithm
		Vec P, Q, T, E1 = v1 - v0, E2 = v2 - v0;
		double det, inv_det, u, v, t;
		P = ray.d ^ E2;
		det = E1 | P;
		if (det > -EPS && det < EPS) return false;

		inv_det = 1.0 / det;
		T = ray.o - v0;
		u = (T | P) * inv_det;
		if (u < 0.0 || u > 1.0) return false;

		Q = T ^ E1;
		v = (ray.d | Q) * inv_det;
		if (v < 0.0 || u + v > 1.0) return false;

		t = ((E2 | Q) * inv_det);
		if (t > EPS && t < isect.distance) {
			Vec norm = (E1 ^ E2).normal();
			/// Back-face culling
			if ((ray.d | norm) > 0.) return false;

			isect.distance = t;
			isect.material = material;
			isect.normal = norm;
			isect.pos = (ray.o + (ray.d * t));
			return true;
		}
		return false;
	};
};

struct Material { Vec kd, ke; };

struct Camera {
	Vec		pos, dir, u, v;
	int		w, h;

	Camera() {};
	Camera(Vec _p, Vec _d, double _f, int _w, int _h)
		: pos(_p), dir(_d), w(_w), h(_h) {
		double fw = (2.0 * std::tan((_f * 0.0174532925) / 2.0));
		u = dir ^ Vec{ 0, 1, 0 };
		v = u   ^ dir;
		u = u * fw;
		v = v * fw;
	};

	inline Ray getRay(const double x, const double y) const {
		const double px = 0.5 * ((2.0 * (x)) + 1.0 - (double) w),
					 py = 0.5 * ((2.0 * (y)) + 1.0 - (double) h);
		return{ pos, (dir + (u * (1. / (double) w) * px) + (v * (1. / (double) w) * py)).normal() };
	};
};

struct TreeNode {
	int			startElem,	endElem;
	Vec			v0, v1;
	TreeNode	*leftChild, *rightChild;
	
	TreeNode() : startElem(0), endElem(0), leftChild(nullptr), rightChild(nullptr), v0{ INF, INF, INF }, v1{ -INF, -INF, -INF } {};
	TreeNode(const int _s, const int _e) : startElem(_s), endElem(_e), leftChild(nullptr), rightChild(nullptr), v0{ INF, INF, INF }, v1{ -INF, -INF, -INF } {};
	~TreeNode() {
		MEL::MemDestruct( leftChild);
		MEL::MemDestruct(rightChild);
	};

	inline bool intersect(const Ray &rayInv, double &tmin, const double dist) const {
		// Check local AABB
		double tmax;
		Vec t0 = (v0 - rayInv.o) * rayInv.d;
		Vec t1 = (v1 - rayInv.o) * rayInv.d;
		tmin = std::fmax(std::fmax(std::fmin(t0.x, t1.x), std::fmin(t0.y, t1.y)), std::fmin(t0.z, t1.z));
		tmax = std::fmin(std::fmin(std::fmax(t0.x, t1.x), std::fmax(t0.y, t1.y)), std::fmax(t0.z, t1.z));
		return !(tmax < EPS || tmin > tmax || tmin > dist);
	};

	inline void DeepCopy(MEL::Deep::Message &msg) {
		msg.packPtr( leftChild);
		msg.packPtr(rightChild);
	};
};

struct Scene {
	std::vector<Material> materials;
	std::vector<Triangle> mesh;
	TreeNode *rootNode;
	Camera camera;

	Scene() : rootNode(nullptr) {};
	Scene(const Scene &old)					  = delete; // Remove CopyConstructor
	inline Scene& operator=(const Scene &old) = delete; // Remove CopyAssignment
	Scene(Scene &&old) : mesh(std::move(old.mesh)), materials(std::move(old.materials)), 
						 camera(old.camera), rootNode(old.rootNode) {
		old.mesh.clear();
		old.materials.clear();
		old.rootNode	= nullptr;
	};
	inline Scene& operator=(Scene &&old) {
		mesh			= std::move(old.mesh);
		materials		= std::move(old.materials);
		rootNode		= old.rootNode;
		camera			= old.camera;
		old.mesh.clear();
		old.materials.clear();
		old.rootNode	= nullptr;
		return *this;
	};
	~Scene() {
		MEL::MemDestruct(rootNode);
	};

	inline bool intersect(const Ray &ray, Intersection &isect) const {
		const Ray invRay{ ray.o, ray.d.inv() };

		// Stack based traversal
		std::stack<std::pair<const TreeNode*, double>> treeStack;
		double rootDist;
		if (rootNode->intersect(invRay, rootDist, isect.distance)) treeStack.push({ rootNode, rootDist });

		// While the stack is not empty there is work to be done
		bool found = false;
		while (!treeStack.empty()) {
			// Get the current node to traverse
			auto top = treeStack.top();
			const TreeNode *currentNode = top.first;
			const double	bboxDist = top.second;
			treeStack.pop();

			// Is it still worth traversing this node?
			if (bboxDist >= isect.distance) continue;

			// Intersect triangles within leaf node
			if (currentNode->leftChild == nullptr) {
				for (int i = currentNode->startElem; i < currentNode->endElem; ++i) {
					found |= mesh[i].intersect(ray, isect);
				}
				continue;
			}

			// Intersect child nodes, closest first
			double lDist = INF, rDist = INF;
			const bool lHit = currentNode->leftChild->intersect( invRay, lDist, isect.distance);
			const bool rHit = currentNode->rightChild->intersect(invRay, rDist, isect.distance);
			if (lHit && rHit) {
				if (lDist < rDist) {
					treeStack.push({ currentNode->rightChild, rDist });
					treeStack.push({ currentNode->leftChild,  lDist });
				}
				else {
					treeStack.push({ currentNode->leftChild,  lDist });
					treeStack.push({ currentNode->rightChild, rDist });
				}
			}
			else if (lHit) {
				treeStack.push({ currentNode->leftChild,  lDist });
			}
			else if (rHit) {
				treeStack.push({ currentNode->rightChild, rDist });
			}
		}
		return found;
	};

	inline void DeepCopy(MEL::Deep::Message &msg) {
		msg & mesh & materials;
		msg.packPtr(rootNode);
	};
};

inline Scene* loadScene(const std::string &scenePath) {
	Scene *scene = MEL::MemConstruct<Scene>();
	
	/// Helpers
	auto hasPrefix = [](const std::string &str, const std::string &prefix) {
		return (str.size() >= prefix.size()) && (std::mismatch(prefix.cbegin(), prefix.cend(), str.cbegin()).first == prefix.cend());
	};
	auto split = [](const std::string &str, const char delim) {
		std::vector<std::string> result; result.reserve(16);
		size_t start{}, end;
		while ((end = str.find(delim, start)) != std::string::npos) {
			if (str[start] != ' ') result.push_back(str.substr(start, (end - start)));
			start = end + 1;
		}
		result.push_back(str.substr(start));
		return result;
	};

	/// Load .scene file
	std::ifstream sceneFile(scenePath, std::ios::in);
	if (sceneFile.is_open()) {
		std::string sceneLine;
		while (std::getline(sceneFile, sceneLine)) {
			/// Material Declaration
			if (hasPrefix(sceneLine, "m ")) {
				std::vector<std::string> sm = split(sceneLine, ' ');
				scene->materials.push_back(Material{
						{ std::stod(sm[1]), std::stod(sm[2]), std::stod(sm[3]) },
						{ std::stod(sm[4]), std::stod(sm[5]), std::stod(sm[6]) }
				});
				continue;
			}
			/// Camera Declaration
			if (hasPrefix(sceneLine, "c ")) {
				std::vector<std::string> sm = split(sceneLine, ' ');
				scene->camera = Camera(
					Vec{ std::stod(sm[1]), std::stod(sm[2]), std::stod(sm[3]) },
					Vec{ std::stod(sm[4]), std::stod(sm[5]), std::stod(sm[6]) }.normal(),
					std::stod(sm[7]), std::stoi(sm[8]), std::stoi(sm[9])
				);
				continue;
			}
			/// Mesh Declaration
			if (hasPrefix(sceneLine, "v ")) {
				std::vector<std::string> sm = split(sceneLine, ' ');
				auto rtrim = [](std::string &str) {
					size_t endpos = str.find_last_not_of("\r\n");
					if (endpos != std::string::npos) {
						return str.substr(0, endpos+1);
					}
					return str;
				};
				/// Load .obj file
				const int material = std::stoi(sm[1]) - 1;
				const std::string meshPath = rtrim(sm[2]);
				std::vector<Vec> vertices;
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
							scene->mesh.push_back(Triangle{ V0, V1, V2, material });
							continue;
						}
					}
					meshFile.close();
				}
				else {
					std::cout << "Error loading: " << meshPath << std::endl;
					std::exit(-1);
				}
				continue;
			}
		}
	}
	else {
		std::cout << "Error loading scene: " << scenePath << std::endl;
		std::exit(-1);
	}

	std::cout << "Building BVH Tree with Median Splits" << std::endl;
	auto start = MEL::Wtime();

	// Create root node
	scene->mesh.shrink_to_fit(); // Optimize the vector that was passed
	int numNodes = 1;
	scene->rootNode = MEL::MemConstruct<TreeNode>(0, scene->mesh.size());

	// Stack based traversal
	// We track tree nodes and their depth in the tree
	std::stack<std::pair<TreeNode*, int>> treeStack;
	treeStack.push({ scene->rootNode, 0 }); // <- Depth 0

	// While the stack is not empty there is work to be done
	while (!treeStack.empty()) {
		// Get the current node to traverse
		auto top = treeStack.top();
		TreeNode *currentNode = top.first;
		const int depth = top.second;
		treeStack.pop();

		const int numGeom = currentNode->endElem - currentNode->startElem;

		// Compute the nodes bounding box
		Vec b0{ INF, INF, INF }, b1{ -INF, -INF, -INF };
		for (int i = currentNode->startElem; i < currentNode->endElem; ++i) {
			currentNode->v0 = currentNode->v0.min(scene->mesh[i].min());
			currentNode->v1 = currentNode->v1.max(scene->mesh[i].max());
			const Vec c = scene->mesh[i].centroid();
			b0 = b0.min(c);
			b1 = b1.max(c);
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
		currentNode->rightChild = MEL::MemConstruct<TreeNode>(midElem,   currentNode->endElem);

		// Push new nodes onto the working stack
		treeStack.push({ currentNode->rightChild, depth + 1 });
		treeStack.push({ currentNode->leftChild,  depth + 1 });
	}
	auto end = MEL::Wtime();
	std::cout << "BVH Tree constructed of ( " << numNodes <<  " ) nodes in " 
			  << std::setprecision(4) << (end - start) << "s" << std::endl;
	return scene;
};

inline Vec render(const Scene *scene, const Ray &ray) {
	Intersection isect;
	if (scene->intersect(ray, isect)) {
		return scene->materials[isect.material].kd * std::fabs(ray.d | isect.normal);
	}
	return{}; 
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
	MPI_Bcast(&(scene->mesh[0]),		sizeof(Triangle) * sizes[0], MPI_CHAR, root, comm); 
	MPI_Bcast(&(scene->materials[0]),	sizeof(Material) * sizes[1], MPI_CHAR, root, comm); 

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
		/// !!!NOTE!!! The current value of the leftChild and rightChild pointers are dangling 
		/// references to the memory on the sending process. We can use whether they are nullptr
		/// or not to determine if we need to allocate space for the real nodes but we cannot 
		/// safely dereference them!
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
		packed_size += sizeof(int) + ((int) scene->mesh.size()		* sizeof(Triangle));
		packed_size += sizeof(int) + ((int) scene->materials.size()	* sizeof(Material));
			
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

		int mesh_size		= scene->mesh.size(),
			materials_size	= scene->materials.size();

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

int main(int argc, char *argv[]) {
	MEL::Init(argc, argv);

	/// Who are we?
	MEL::Comm comm = MEL::Comm::WORLD;
	const int rank = MEL::CommRank(comm), 
			  size = MEL::CommSize(comm);
	
	/// Check for correct params
	if (argc != 3) {
		if (rank == 0) std::cout << "Usage: mpirun ./RayTracingDeepCopy [scene_file:] [bcast_method_id: (0..3)]" << std::endl;
		std::exit(-1);
	}

	/// Parse cmd args
	const std::string scenePath = std::string(argv[1]);
	const int		  method	= std::stoi(argv[2]);

	/// Check if method is valid
	if (method < 0 || method > 3) {
		if (rank == 0) std::cout << "Invalid Method Id: Must be in range (0..3) | Saw: " << method << std::endl;
		std::exit(-1);
	}

	/// ****************************************** ///
	/// Load the scene on the root process         ///
	/// ****************************************** ///
	Scene *scene = nullptr;
	if (rank == 0) {
		scene = loadScene(scenePath);
		std::cout << "Rank: " << std::setw(4) << rank 
				  << " Loading Scene: " << scenePath << std::endl;
	}
	
	/// ****************************************** ///
	/// Broadcast the scene object to all nodes    ///
	/// ****************************************** ///
	MEL::Barrier(comm);
	auto startTime = MEL::Wtime();

	switch (method) {
	case 0:
		MEL::Deep::Bcast(scene, 0, comm);
		break;
	case 1:
		MEL::Deep::BufferedBcast(scene, 0, comm);
		break;
	case 2:
		MPI_NonBufferedBcast_Scene(scene, rank, 0, (MPI_Comm) comm);
		break;
	case 3:
		MPI_BufferedBcast_Scene(scene, rank, 0, (MPI_Comm) comm);
		break;
	};

	MEL::Barrier(comm);
	auto endTime = MEL::Wtime();
	
	if (rank == 0) {
		std::cout << "Rank: " << std::setw(4) << rank << " Scene Bcast in " 
				  << std::fixed << (endTime - startTime) << "s" << std::endl;
	}
	
	/// ****************************************** ///
	/// EVERYTHING FROM HERE ON IS RENDERING STUFF ///
	/// ****************************************** ///
	
	/// Allocate image plane
	const int w = scene->camera.w, h = scene->camera.h;
	/// Use wR for image width as BMP files require the
	/// width be padded to a multiple of four bytes
	const int R = (w % 4), wR = w + R;

	char *filmPtr = nullptr;
	MEL::Win filmWin;
	if (rank == 0) {
		/// Root has the main film and exposes it to the workers
		filmPtr = MEL::MemAlloc<char>(wR * h * 3);
		filmWin = MEL::WinCreate(filmPtr, wR * h * 3, comm);
	}
	else {
		/// Workers don't expose anything
		filmWin = MEL::WinCreate(filmPtr, 0, comm);
	}

	auto typeColour		= MEL::TypeCreateContiguous(MEL::Datatype::UNSIGNED_CHAR, 3);
	auto typeFilm		= MEL::TypeCreateContiguous(typeColour, wR * h);
	auto sharedIndex	= MEL::SharedCreate<int>(1, rank, size, 0, comm);

	/// Work distribution by blocks
	const int blockSize = 1 << 6, // 64
			  blockSize2 = blockSize * blockSize,
			  uBlocks = ((w + blockSize - 1) & ~(blockSize - 1)) / blockSize,
			  vBlocks = ((h + blockSize - 1) & ~(blockSize - 1)) / blockSize,
			  tBlocks = uBlocks * vBlocks;
	
	/// ****************************************** ///
	/// Render the image block by block            ///
	/// ****************************************** ///
	while (true) {
		/// Get the next block (dynamic load balancing)
		MEL::SharedLock(sharedIndex);
		const int localIndex = (*sharedIndex)++;
		MEL::SharedUnlock(sharedIndex);

		/// No more work to do
		if (localIndex >= tBlocks) break;
		std::cout << "Rank: " << std::setw(4) << rank << " Starting block " 
							  << std::setw(4) << (localIndex + 1) << " of " 
							  << std::setw(4) << tBlocks << std::endl;

		/// Where is the local block in the global image
		const int bx = (localIndex % uBlocks) * blockSize, 
				  by = (localIndex / uBlocks) * blockSize, 
				  bw = std::min(blockSize, w - bx), 
				  bh = std::min(blockSize, h - by);

		/// Helper types for moving data
		auto typeGlobalBlock = MEL::TypeCreateSubArray2D(typeColour, bx, by, bw, bh, wR, h);
		auto typeLocalBlock  = MEL::TypeCreateContiguous(typeColour, bw * bh);

		/// Allocate local image block
		unsigned char *blockPtr = MEL::MemAlloc<unsigned char>(bw * bh * 3);

		/// Use openmp to render pixels within block
		#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic) 
		for (int i = 0; i < bw * bh; ++i) {
			const int x = i % bw, y = (i - x) / bw, j = i * 3;
			const Vec col = render(scene, scene->camera.getRay(bx + x, by + y));
			blockPtr[j]   = ColourCorrect(col.z);
			blockPtr[j+1] = ColourCorrect(col.y);
			blockPtr[j+2] = ColourCorrect(col.x);
		}

		/// Write local block to global image on root process
		MEL::WinLockShared(filmWin, 0);
		MEL::Put(blockPtr, typeLocalBlock, 0, typeGlobalBlock, 0, filmWin);
		MEL::WinUnlock(filmWin, 0);

		/// Clean up
		MEL::MemFree(blockPtr);
		MEL::TypeFree(typeGlobalBlock, typeLocalBlock);
	}
	
	MEL::SharedFree(sharedIndex);
	MEL::Barrier(comm);

	/// ****************************************** ///
	/// Save the output as a BMP 24-bpp            ///
	/// ****************************************** ///
	if (rank == 0) {
		std::cout << "Rank: " << std::setw(4) << rank << " Saving image to output.bmp" << std::endl;
		auto file = MEL::FileOpenIndividual("output.bmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
		
		/// BMP 24-bpp Header
		const int R = (w % 4), wR = w + R, fileSize = 0x36 + (wR * h);
		unsigned char hdr[0x36] = { 66, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0x36, 0, 0, 0, 
									40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24 };
		*((int*) (hdr + 0x02)) = fileSize;  /// Total file size
		*((int*) (hdr + 0x12)) = w;			/// Width
		*((int*) (hdr + 0x16)) = h;			/// Height
		MEL::FileSetSize(file, fileSize);
		MEL::FileWrite(file, hdr, 0x36);
		
		// Write pixel data
		MEL::FileSetView(file, 0x36, MEL::Datatype::UNSIGNED_CHAR, typeFilm);
		MEL::FileWrite(file, filmPtr, 1, typeFilm);
		MEL::FileClose(file);
	}

	/// Clean up
	MEL::TypeFree(typeColour, typeFilm);
	MEL::WinFree(filmWin);
	MEL::MemFree(filmPtr);
	MEL::MemDestruct(scene);

	MEL::Finalize();
	return 0;
};