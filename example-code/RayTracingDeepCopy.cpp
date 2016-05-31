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

/// C++11 RNG
struct RNG {
#if !defined(USE_C_RNG)
	std::uniform_real_distribution<double> dist;
    std::mt19937 eng;
	RNG() : RNG(0) {};
	RNG(const unsigned long s) : dist(0., 1.), eng() { seed(s); };
    inline void seed(const unsigned long s) { eng.seed(s); };
    inline double operator()() { return dist(eng); };
#else
	RNG() {};
	RNG(const unsigned long s) { seed(s); };
	inline void seed(const unsigned long s) { srand(s); };
	inline double operator()() { 
		double r;
		#pragma omp critical
		r = (double) rand() / (double) RAND_MAX;
		return r; 
	};
#endif
};

/// C++11 Multi-Threaded RNG
struct MT_RNG {
    RNG *rngs = nullptr;
	int num_threads = 0;
	MT_RNG(const unsigned long s) {
		num_threads = omp_get_max_threads();
		rngs = new RNG[num_threads];
		const unsigned long t = time(nullptr);
        for (int i = 0; i < num_threads; ++i) rngs[i] = RNG(t + s + i);
    };
	MT_RNG(const MT_RNG &old) = delete;
	MT_RNG operator=(const MT_RNG &old) = delete;
	MT_RNG(MT_RNG &&old)                = delete;
	MT_RNG operator=(MT_RNG &&old)      = delete;
	~MT_RNG() {
		delete [] rngs;
	};
    inline double operator()() { return rngs[omp_get_thread_num()](); };
};

/// Colour Correction
inline unsigned char ColourCorrect(const double x) {
	const auto Gamma_Uncharted = [](const double x) -> double {
        constexpr double A = 0.15, B = 0.5, C = 0.1, D = 0.2, E = 0.02, F = 0.30;
        return ((x * (A * x + C * B) + D * E) / (x * (A * x * B) + D * F)) - E / F;
    };
    constexpr double gamma = (1.0 / 2.2), exposure = 1.0, exposureBias = 2.0, whitePoint = 11.2;
	const double y = std::fmax(0., std::fmin(1., (std::pow((Gamma_Uncharted(x * exposure * exposureBias) / Gamma_Uncharted(whitePoint)), gamma))));
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
    double    distance;
    int        material;
    Vec        normal, pos;
    Intersection() : distance(INF) {};
};

struct Triangle {
    Vec v0, v1, v2, n0, n1, n2;
    int material;

    inline Vec min() const { const Vec EPSV{ EPS, EPS, EPS }; return v0.min(v1.min(v2)) - EPSV; };
    inline Vec max() const { const Vec EPSV{ EPS, EPS, EPS }; return v0.max(v1.max(v2)) - EPSV; };
    inline Vec centroid() const { return (max() + min()) * .5; };

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
			/// Face Normal by Winding
            //Vec norm = (E1 ^ E2).normal();
			/// Interpolated Normal from Vertex Normals
			Vec norm = ((n1 * u) + (n2 * v) + (n0 * (1.0 - u - v))).normal(); 

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

struct Material { 
	Vec kd, ke; 
	Material() {};
	Material(const Vec &_kd, const Vec &_ke) : kd(_kd), ke(_ke) {};
};

struct Camera {
    Vec        pos, dir, u, v;
    int        w, h;

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
    int          startElem, endElem;
    Vec          v0, v1;
    TreeNode    *leftChild, *rightChild;
    
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
    Scene(const Scene &old)                   = delete; // Remove CopyConstructor
    inline Scene& operator=(const Scene &old) = delete; // Remove CopyAssignment
    Scene(Scene &&old) : mesh(std::move(old.mesh)), materials(std::move(old.materials)), 
                         camera(old.camera), rootNode(old.rootNode) {
        old.mesh.clear();
        old.materials.clear();
        old.rootNode    = nullptr;
    };
    inline Scene& operator=(Scene &&old) {
        mesh            = std::move(old.mesh);
        materials        = std::move(old.materials);
        rootNode        = old.rootNode;
        camera            = old.camera;
        old.mesh.clear();
        old.materials.clear();
        old.rootNode    = nullptr;
        return *this;
    };
    ~Scene() {
        MEL::MemDestruct(rootNode);
    };

	inline void DeepCopy(MEL::Deep::Message &msg) {
		msg & mesh & materials;
		msg.packPtr(rootNode);
	};

	// Ray BVH-Tree(Triangle) intersection
    inline bool intersect(const Ray &ray, Intersection &isect) const {
        const Ray invRay{ ray.o, ray.d.inv() };

        // Stack based traversal
        std::stack<std::pair<const TreeNode*, double>> treeStack;
        double rootDist;
        if (rootNode->intersect(invRay, rootDist, isect.distance)) treeStack.emplace( rootNode, rootDist );

        // While the stack is not empty there is work to be done
        bool found = false;
        while (!treeStack.empty()) {
            // Get the current node to traverse
            auto top = treeStack.top();
            const TreeNode *currentNode = top.first;
            const double    bboxDist = top.second;
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
                    treeStack.emplace( currentNode->rightChild, rDist );
					treeStack.emplace( currentNode->leftChild, lDist );
                }
                else {
					treeStack.emplace( currentNode->leftChild, lDist );
					treeStack.emplace( currentNode->rightChild, rDist );
                }
            }
            else if (lHit) {
				treeStack.emplace( currentNode->leftChild, lDist );
            }
            else if (rHit) {
				treeStack.emplace( currentNode->rightChild, rDist );
            }
        }
        return found;
    };

	// Scene loading helpers
	template<typename ...Args>
	inline void setCamera(Args &&...args) {
		camera = Camera(args...);
	};
	template<typename ...Args>
	inline void addMaterial(Args &&...args) {
		materials.emplace_back(args...);
	};
	inline void addObj(const int material, const std::string &meshPath) {
		/// Helpers
		const auto hasPrefix = [](const std::string &str, const std::string &prefix) -> bool {
			return (str.size() >= prefix.size()) && (std::mismatch(prefix.begin(), prefix.end(), str.begin()).first == prefix.end());
		};
		const auto split = [](const std::string &str, const char delim) -> std::vector<std::string> {
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
		std::vector<Vec> vertices, normals;
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
				/// Vertex Normal Declaration
				if (hasPrefix(meshLine, "vn ")) {
					std::vector<std::string> sm = split(meshLine, ' ');
					Vec v{ std::stod(sm[1]), std::stod(sm[2]), std::stod(sm[3]) };
					normals.push_back(v.normal());
					continue;
				}
				/// Face Declaration
				if (hasPrefix(meshLine, "f ")) {
					std::vector<std::string> fm = split(meshLine, ' '), sm;
					sm = split(fm[1], '/'); int v0 = std::stoi(sm[0]), n0 = std::stoi(sm[2]);
					sm = split(fm[2], '/'); int v1 = std::stoi(sm[0]), n1 = std::stoi(sm[2]);
					sm = split(fm[3], '/'); int v2 = std::stoi(sm[0]), n2 = std::stoi(sm[2]);
					Vec V0{}, V1{}, V2{}, N0{}, N1{}, N2{};
					V0 = (v0 > 0) ? vertices[v0 - 1] : (v0 < 0) ? vertices[vertices.size() + v0] : Vec{};
					V1 = (v1 > 0) ? vertices[v1 - 1] : (v1 < 0) ? vertices[vertices.size() + v1] : Vec{};
					V2 = (v2 > 0) ? vertices[v2 - 1] : (v2 < 0) ? vertices[vertices.size() + v2] : Vec{};
					N0 = (n0 > 0) ? normals[n0 - 1] : (n0 < 0) ? normals[normals.size() + n0] : Vec{};
					N1 = (n1 > 0) ? normals[n1 - 1] : (n1 < 0) ? normals[normals.size() + n1] : Vec{};
					N2 = (n2 > 0) ? normals[n2 - 1] : (n2 < 0) ? normals[normals.size() + n2] : Vec{};
					mesh.push_back(Triangle{ V0, V1, V2, N0, N1, N2, material });
					continue;
				}
			}
			meshFile.close();
			std::cout << "Successfully loaded: " << meshPath << std::endl;
		}
		else {
			std::cout << "Error loading: " << meshPath << std::endl;
			MEL::Exit(-1);
		}
	};
	inline void buildBVHTree() {
		std::cout << "Building BVH Tree with SAH Splits" << std::endl;
		auto startTime = MEL::Wtime();

		// Clear root node if it already exists
		MEL::MemDestruct(rootNode);
		// Create root node
		int numNodes = 1;
		rootNode = MEL::MemConstruct<TreeNode>(0, mesh.size());

		// Stack based traversal
		// We track tree nodes and their depth in the tree
		std::stack<std::pair<TreeNode*, int>> treeStack;
		treeStack.emplace( rootNode, 0 ); // <- Depth 0

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
				currentNode->v0 = currentNode->v0.min(mesh[i].min());
				currentNode->v1 = currentNode->v1.max(mesh[i].max());
				const Vec c = mesh[i].centroid();
				b0 = b0.min(c);
				b1 = b1.max(c);
			}

			// Mid index for partitioning
			int midElem = -1;
			typename std::vector<Triangle>::iterator start = mesh.begin() + currentNode->startElem,
													 end   = mesh.begin() + currentNode->endElem,
													 mid;

			// Is it worth splitting?
			if (numGeom <= 1) {
				continue;
			}
			else if (numGeom <= 4) {
				// Median Splits
				midElem = (currentNode->startElem + (numGeom / 2));
				mid = mesh.begin() + midElem;

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
			}
			else {
				// SAH Splits
				const int numBuckets = 8;
				struct SAH_Bucket {
					int count;
					Vec b0, b1;
					SAH_Bucket() : count(0), b0{ INF, INF, INF }, b1{ -INF, -INF, -INF } {};
				} bucketsX[numBuckets], bucketsY[numBuckets], bucketsZ[numBuckets];

				// Current node data
				const Vec bboxMin = currentNode->v0;
				const Vec bboxMax = currentNode->v1;

				// Compute the nodes bounding box
				Vec dBox0{ INF, INF, INF }, dBox1{ -INF, -INF, -INF };
				for (int i = currentNode->startElem; i < currentNode->endElem; ++i) {
					// Geom elem data
					const Triangle &elem = mesh[i];
					const Vec elemBox0 = elem.min(), elemBox1 = elem.max();
					const Vec elemCentroid = elem.centroid();
					dBox0 = dBox0.min(elemCentroid);
					dBox1 = dBox1.max(elemCentroid);

					// Work out which bucket the current elems goes in
					int bX = (int) floor((double) numBuckets * ((elemCentroid.x - bboxMin.x) / (bboxMax.x - bboxMin.x)));
					int bY = (int) floor((double) numBuckets * ((elemCentroid.y - bboxMin.y) / (bboxMax.y - bboxMin.y)));
					int bZ = (int) floor((double) numBuckets * ((elemCentroid.z - bboxMin.z) / (bboxMax.z - bboxMin.z)));
					bX = ((bX < numBuckets) ? bX : (numBuckets - 1));
					bY = ((bY < numBuckets) ? bY : (numBuckets - 1));
					bZ = ((bZ < numBuckets) ? bZ : (numBuckets - 1));

					// Update the buckets
					bucketsX[bX].count++; bucketsX[bX].b0 = bucketsX[bX].b0.min(elemBox0); bucketsX[bX].b1 = bucketsX[bX].b1.max(elemBox1);
					bucketsY[bY].count++; bucketsY[bY].b0 = bucketsY[bY].b0.min(elemBox0); bucketsY[bY].b1 = bucketsY[bY].b1.max(elemBox1); 
					bucketsZ[bZ].count++; bucketsZ[bZ].b0 = bucketsZ[bZ].b0.min(elemBox0); bucketsZ[bZ].b1 = bucketsZ[bZ].b1.max(elemBox1);
				}

				// A neat structure to make calculating relative costs easier
				struct SAH_CostBucket {
					Vec b0b0, b0b1, b1b0, b1b1;
					int c0, c1;
					SAH_CostBucket() : c0(0), c1(0), 
					                   b0b0{ INF, INF, INF }, b0b1{ -INF, -INF, -INF }, 
									   b1b0{ INF, INF, INF }, b1b1{ -INF, -INF, -INF } {};
				};

				// Initial cost values
				double cXCost = INF, cYCost = INF, cZCost = INF;
				int cXi = 0, cYi = 0, cZi = 0;

				// Calculate costs for each bucket and track smallest cost indices
				for (int i = 0; i < numBuckets; ++i) {
					// Cost data
					SAH_CostBucket cX, cY, cZ;

					// Left node sweep
					for (int j = 0; j <= i; ++j) {
						cX.b0b0 = cX.b0b0.min(bucketsX[j].b0); cX.b0b1 = cX.b0b1.max(bucketsX[j].b1); cX.c0 += bucketsX[j].count;
						cY.b0b0 = cY.b0b0.min(bucketsY[j].b0); cY.b0b1 = cY.b0b1.max(bucketsY[j].b1); cY.c0 += bucketsY[j].count;
						cZ.b0b0 = cZ.b0b0.min(bucketsZ[j].b0); cZ.b0b1 = cZ.b0b1.max(bucketsZ[j].b1); cZ.c0 += bucketsZ[j].count;
					}

					// Right node sweep
					for (int j = (i + 1); j < numBuckets; ++j) {
						cX.b1b0 = cX.b1b0.min(bucketsX[j].b0); cX.b1b1 = cX.b1b1.max(bucketsX[j].b1); cX.c1 += bucketsX[j].count;
						cY.b1b0 = cY.b1b0.min(bucketsY[j].b0); cY.b1b1 = cY.b1b1.max(bucketsY[j].b1); cY.c1 += bucketsY[j].count;
						cZ.b1b0 = cZ.b1b0.min(bucketsZ[j].b0); cZ.b1b1 = cZ.b1b1.max(bucketsZ[j].b1); cZ.c1 += bucketsZ[j].count;
					}

					auto surfaceArea = [](Vec v0, Vec v1) -> double {
						const Vec s = (v1 - v0); return ((s.x * s.y) + (s.x * s.z) + (s.y * s.z)) * 2.0;
					};

					// Calculate cost per axis
					const double SA = surfaceArea(currentNode->v0, currentNode->v1);
					const double costX = .125 * ((((double) cX.c0 * surfaceArea(cX.b0b0, cX.b0b1)) + ((double) cX.c1 * surfaceArea(cX.b1b0, cX.b1b1))) / SA);
					const double costY = .125 * ((((double) cY.c0 * surfaceArea(cY.b0b0, cY.b0b1)) + ((double) cY.c1 * surfaceArea(cY.b1b0, cY.b1b1))) / SA);
					const double costZ = .125 * ((((double) cZ.c0 * surfaceArea(cZ.b0b0, cZ.b0b1)) + ((double) cZ.c1 * surfaceArea(cZ.b1b0, cZ.b1b1))) / SA);

					// Update costs if less than current minimums
					if (i == 0 || costX < cXCost) { cXCost = costX; cXi = i; }
					if (i == 0 || costY < cYCost) { cYCost = costY; cYi = i; }
					if (i == 0 || costZ < cZCost) { cZCost = costZ; cZi = i; }
				}

				Vec dim = dBox1 - dBox0;
				const double DTHRESH = (EPS * 2.);
				if (dim.x < DTHRESH) cXCost = INF;
				if (dim.y < DTHRESH) cYCost = INF;
				if (dim.z < DTHRESH) cZCost = INF;

				// Select the best axis and use std::partition to split the current 
				// sub-section of the original vector into a "left" and "right" sub-list
				if (cXCost < cYCost && cXCost < cZCost) {
					mid = std::partition(start, end, [&](const Triangle &a) -> bool {
						int bX = (int) floor((double) numBuckets * ((a.centroid().x - bboxMin.x) / (bboxMax.x - bboxMin.x)));
						bX = ((bX < numBuckets) ? bX : (numBuckets - 1));
						return (bX <= cXi);
					});
					midElem = std::distance(mesh.begin(), mid);
				}
				else if (cYCost < cXCost && cYCost < cZCost) {
					mid = std::partition(start, end, [&](const Triangle &a) -> bool {
						int bY = (int) floor((double) numBuckets * ((a.centroid().y - bboxMin.y) / (bboxMax.y - bboxMin.y)));
						bY = ((bY < numBuckets) ? bY : (numBuckets - 1));
						return (bY <= cYi);
					});
					midElem = std::distance(mesh.begin(), mid);
				}
				else if (cZCost < cXCost && cZCost < cYCost) {
					mid = std::partition(start, end, [&](const Triangle &a) -> bool {
						int bZ = (int) floor((double) numBuckets * ((a.centroid().z - bboxMin.z) / (bboxMax.z - bboxMin.z)));
						bZ = ((bZ < numBuckets) ? bZ : (numBuckets - 1));
						return (bZ <= cZi);
					});
					midElem = std::distance(mesh.begin(), mid);
				}
				else {
					// Median Splits
					midElem = (currentNode->startElem + (numGeom / 2));
					mid = mesh.begin() + midElem;

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
				}
			}        

			// Create child nodes based on partition
			numNodes += 2;
			currentNode->leftChild  = MEL::MemConstruct<TreeNode>(currentNode->startElem, midElem);
			currentNode->rightChild = MEL::MemConstruct<TreeNode>(midElem,   currentNode->endElem);

			// Push new nodes onto the working stack
			treeStack.emplace( currentNode->rightChild, depth + 1 );
			treeStack.emplace( currentNode->leftChild,  depth + 1 );
		}
		auto endTime = MEL::Wtime();
		std::cout << "BVH Tree constructed of ( " << numNodes <<  " ) nodes in " 
				  << std::setprecision(4) << (endTime - startTime) << "s" << std::endl;
	};
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

inline Vec render(MT_RNG &rng, const Scene *scene, const int x, const int y, const int spp) {
    Vec colour{};
    
    for (int i = 0; i < spp; ++i) {
        // Jitter sample on image plane
        const double jx = rng() - .5, jy = rng() - .5;
        Ray ray = scene->camera.getRay((double) x + jx, (double) y + jy);

        // Compute sample value
        Vec cL{};
        Intersection isect;
        if (scene->intersect(ray, isect)) {
            cL = scene->materials[isect.material].kd * std::fabs(ray.d | isect.normal);
        }

        // Accumulate sample
        colour = colour + cL;
    }

    return colour / (double) spp;
};

int main(int argc, char *argv[]) {
    MEL::Init(argc, argv); 

    /// Who are we?
    MEL::Comm comm = MEL::Comm::WORLD;
    const int rank = MEL::CommRank(comm), 
              size = MEL::CommSize(comm);
    
    /// Check for correct params
    if (argc != 3) {
        if (rank == 0) std::cout << "Usage: mpirun ./RayTracingDeepCopy [bcast_method_id: (0..3)] [samples_per_pixel: (1..)]" << std::endl;
        std::exit(-1);
    }

    /// Parse cmd args
    const int method = std::stoi(argv[1]),
                 spp = std::stoi(argv[2]);

    /// Check if method is valid
    if (method < 0 || method > 3) {
        if (rank == 0) std::cout << "Invalid Method Id: Must be in range (0..3) | Saw: " << method << std::endl;
        std::exit(-1);
    }

    /// Check if spp is valid
    if (spp < 0) {
        if (rank == 0) std::cout << "Invalid Samples per Pixel: Must be greater then 1 | Saw: " << spp << std::endl;
        std::exit(-1);
    }

    /// ****************************************** ///
    /// Load the scene on the root process         ///
    /// ****************************************** ///
    Scene *scene = nullptr;
    if (rank == 0) {
		scene = MEL::MemConstruct<Scene>();

		// Set the camera setCamera(pos, dir, fov, width, height)
		scene->setCamera(Vec{ 0., 500., -1700. }, Vec{ 0., 0., 1. }.normal(), 42.501, 1024, 1024);

		// Add materials addMaterial(kd, ke)
		scene->addMaterial(Vec{  .9,  .9,  .9 }, Vec{ 0, 0, 0 }); // White
		scene->addMaterial(Vec{ .81, .23, .14 }, Vec{ 0, 0, 0 }); // Red
		scene->addMaterial(Vec{ .23, .41, .24 }, Vec{ 0, 0, 0 }); // Green
		scene->addMaterial(Vec{ .62, .71, .13 }, Vec{ 0, 0, 0 }); // Yellow
		scene->addMaterial(Vec{   0.,  0.,  0.}, Vec{ 100, 100, 100 }); // Light Source

		// Load Meshes addObj(material_index, mesh_path)
		scene->addObj(0, "assets/cornellbox-white.obj");
		scene->addObj(1, "assets/cornellbox-red.obj");
		scene->addObj(2, "assets/cornellbox-green.obj");
		scene->addObj(3, "assets/bunny.obj");

		scene->buildBVHTree();
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
    
	/// Multi-Threaded Random Number Generator
	MT_RNG rng(rank); // Seed with MPI rank

    /// Allocate image plane
    const int w = scene->camera.w, h = scene->camera.h;
    /// Use wR for image width as BMP files require the
    /// width be padded to a multiple of four bytes
    const int R = ((w * 3) % 4), wR = (w * 3) + (R == 0 ? 0 : (4 - R));

    char *filmPtr = nullptr;
    MEL::Win filmWin;
    if (rank == 0) {
        /// Root has the main film and exposes it to the workers
        filmPtr = MEL::MemAlloc<char>(wR * h);
        filmWin = MEL::WinCreate(filmPtr, wR * h, comm);
    }
    else {
        /// Workers don't expose anything
        filmWin = MEL::WinCreate(filmPtr, 0, comm);
    }

    auto typeColour  = MEL::TypeCreateContiguous(MEL::Datatype::UNSIGNED_CHAR, 3);
    auto typeFilm    = MEL::TypeCreateContiguous(MEL::Datatype::UNSIGNED_CHAR, wR * h);
    auto sharedIndex = MEL::SharedCreate<int>(1, rank, size, 0, comm);

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
        auto typeGlobalBlock = MEL::TypeCreateSubArray2D(MEL::Datatype::UNSIGNED_CHAR, bx * 3, by, bw * 3, bh, wR, h);
        auto typeLocalBlock  = MEL::TypeCreateContiguous(typeColour, bw * bh);

        /// Allocate local image block
        unsigned char *blockPtr = MEL::MemAlloc<unsigned char>(bw * bh * 3);

        /// Use openmp to render pixels within block
		#pragma omp parallel for schedule(dynamic) shared(rng)
        for (int i = 0; i < bw * bh; ++i) {
            const int x = i % bw, y = (i - x) / bw, j = i * 3;
            const Vec col = render(rng, scene, (bx + x), (by + y), spp);
            blockPtr[j]   = ColourCorrect(col.z);
            blockPtr[j+1] = ColourCorrect(col.y);
            blockPtr[j+2] = ColourCorrect(col.x);
        }

        /// Write local block to global image on root process
        MEL::WinLockShared(filmWin, 0);
        MEL::Put(blockPtr, 1, typeLocalBlock, 0, 1, typeGlobalBlock, 0, filmWin);
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
        std::cout << "Rank: " << std::setw(4) << rank << " Saving image to output-RayTracingDeepCopy.bmp" << std::endl;
        auto file = MEL::FileOpenIndividual("output-RayTracingDeepCopy.bmp", MEL::FileMode::CREATE | MEL::FileMode::WRONLY);
        
        /// BMP 24-bpp Header
        const int fileSize = 0x36 + (wR * h);
        unsigned char hdr[0x36] = { 66, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0x36, 0, 0, 0, 
                                    40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24 };
        *((int*) (hdr + 0x02)) = fileSize;  /// Total file size
        *((int*) (hdr + 0x12)) = w;         /// Width
        *((int*) (hdr + 0x16)) = h;         /// Height
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