#include <KDTree.hpp>
#include <PointDescriptor.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <ShapeModel.hpp>
#include <ControlPoint.hpp>



#define KDTTREE_DEBUG_FLAG 0
#define KDTREE_BUILD_DEBUG 0



template <template<class> class ContainerType, class PointType>
KDTree<ContainerType,PointType>::KDTree(ContainerType<PointType> * owner) {
this -> owner = owner;
}

template <template<class> class ContainerType, class PointType>
void KDTree<ContainerType,PointType>::set_depth(int depth) {
this -> depth = depth;
}

template <template<class> class ContainerType, class PointType>
double KDTree<ContainerType,PointType>::get_value() const {
return this -> value;
}

template <template<class> class ContainerType, class PointType>
unsigned int KDTree<ContainerType,PointType>::get_axis() const {
return this -> axis;
}

template <template<class> class ContainerType, class PointType>
void KDTree<ContainerType,PointType>::set_value(double value) {
this -> value = value;
}
template <template<class> class ContainerType, class PointType> void KDTree<ContainerType,PointType>::set_axis(unsigned int axis) {
this -> axis = axis;
}


template <template<class> class ContainerType, class PointType>
void KDTree<ContainerType,PointType>::closest_point_search(const arma::vec & test_point,
const std::shared_ptr<KDTree> & node,
int & best_guess_index,
double & distance) const {

	// If the left child of this node is nullptr, so is the right
	// and we have a leaf node
	if (node -> left == nullptr) {
		KDTree<ContainerType,PointType>::search_node(test_point,node,best_guess_index,distance);
	}

	else {

		bool search_left_first;

		if (test_point(node -> get_axis()) <= node -> get_value()) {
			search_left_first = true;
		}
		else {
			search_left_first = false;
		}

		if (search_left_first) {



			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess_index,
					distance);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess_index,
					distance);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess_index,
					distance);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess_index,
					distance);
			}

		}

	}


}

template <template<class> class ContainerType, class PointType> 
void KDTree<ContainerType,PointType>::closest_N_point_search(const arma::vec & test_point,
	const unsigned int & N_points,
	const std::shared_ptr<KDTree> & node,
	double & distance,
	std::map<double,int > & closest_points) const{

	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << "Depth: " << this -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> indices.size() << std::endl;
	std::cout << "Points found so far : " << closest_points.size() << std::endl;
	#endif

	// DEPRECATED
	if (node -> left == nullptr) {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Leaf node\n";
		#endif

		KDTree<ContainerType,PointType>::search_node(test_point,N_points,node,distance,closest_points);

	}

	else {

		bool search_left_first;


		if (test_point(node -> get_axis()) <= node -> get_value()) {
			search_left_first = true;
		}
		else {
			search_left_first = false;
		}

		if (search_left_first) {



			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> left,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> right,
					distance,
					closest_points);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> right,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> left,
					distance,
					closest_points);
			}

		}

	}

	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << " Leaving "<<  std::endl; ;
	std::cout << "Points found : " << std::endl;
	for (auto it = closest_points.begin(); it != closest_points.end(); ++it){
		std::cout << it -> first << " : " << this -> owner -> get_point_coordinates(it -> second).t().t();
	}
	#endif


}


template <template<class> class ContainerType, class PointType> 
void KDTree<ContainerType,PointType>::radius_point_search(const arma::vec & test_point,
	const std::shared_ptr<KDTree> & node,
	const double & distance,
	std::vector< int > & closest_points_indices) {



	double node_value = node -> get_value();
	double test_value = test_point(node -> get_axis());


	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << "Depth: " << node -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> indices.size() << std::endl;
	std::cout << "Points found so far : " << closest_points_indices.size() << std::endl;
	#endif

	if (node -> left == nullptr ) {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Leaf node\n";
		#endif

		KDTree<ContainerType,PointType>::search_node(test_point,node,distance,closest_points_indices);

	}

	else {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Fork node\n";
		#endif

		

		if (test_value <= node_value) {

			#if KDTTREE_DEBUG_FLAG
			std::cout << "Searching left first\n";
			#endif


			if (test_value - distance <= node_value) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points_indices);
			}

			if (test_value + distance >= node_value) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points_indices);
			}

		}

		else {

			#if KDTTREE_DEBUG_FLAG
			std::cout << "Searching right first\n";
			#endif
			if (test_value + distance >= node_value) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points_indices);
			}

			if (test_value - distance <= node_value) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points_indices);
			}

		}

	}

}

template <template<class> class ContainerType, class PointType>
void KDTree<ContainerType,PointType>::build(const std::vector< int > & indices, int depth) {

	this -> indices = indices;
	this -> left = nullptr;
	this -> right = nullptr;
	this -> set_depth(depth);

	#if KDTREE_BUILD_DEBUG
	std::cout << "Points in node: " << indices.size() <<  std::endl;
	#endif

	if (static_cast<int>(this -> indices.size()) == 0) {
		#if KDTREE_BUILD_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return;
	}
	else if (static_cast<int>(this -> indices.size()) == 1){
		#if KDTREE_BUILD_DEBUG
		std::cout << "Trivial node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return;
	}
	else if (static_cast<int>(this -> indices.size()) < this -> get_min_indices_per_node()){
		#if KDTREE_BUILD_DEBUG
		std::cout << "Node contains less indices ("  <<static_cast<int>(this -> indices.size()) << ") than the prescribed number ";
		std::cout << this -> min_indices_per_node << " . Node depth was " << depth << std::endl;
		#endif
		return ;
	}
	else if (this -> depth == this -> get_max_depth()){
		
		#if KDTREE_BUILD_DEBUG
		std::cout << "Max depth (" << this -> get_max_depth()  << ") reached\n";
		std::cout << "Node contains "  <<static_cast<int>(this -> indices.size()) << " indices \n";

		#endif
		return ;

	}

	else {

		this -> left = std::make_shared<KDTree<ContainerType,PointType>>( KDTree<ContainerType,PointType>(this -> owner) );
		this -> right = std::make_shared<KDTree<ContainerType,PointType>>( KDTree<ContainerType,PointType>(this -> owner) );

		this -> left -> indices = std::vector<int >();
		this -> right -> indices = std::vector<int >();

	}

	arma::vec midpoint = arma::zeros<arma::vec>(this -> owner -> get_point_coordinates(indices[0]).size());
	const arma::vec & start_point = this -> owner -> get_point_coordinates(indices[0]);

	arma::vec min_bounds = start_point;
	arma::vec max_bounds = start_point;

	// Could multithread here
	for (unsigned int i = 0; i < indices.size(); ++i) {

		const arma::vec & point = this -> owner -> get_point_coordinates(indices[i]);

		max_bounds = arma::max(max_bounds,point);
		min_bounds = arma::min(min_bounds,point);

		// The midpoint of all the facets is found
		midpoint += (point / indices.size());
		
	}



	arma::vec bounding_box_lengths = max_bounds - min_bounds;

	// Facets to be assigned to the left and right nodes
	std::vector < int > left_points;
	std::vector < int > right_points;

	#if KDTREE_BUILD_DEBUG
	std::cout << "Midpoint: " << midpoint.t() << std::endl;
	std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	#endif


	if (arma::norm(bounding_box_lengths) == 0) {
		#if KDTREE_BUILD_DEBUG
		std::cout << "Cluttered node" << std::endl;
		#endif

		this -> left = nullptr;
		this -> right = nullptr;

		return;
	}

	int longest_axis = bounding_box_lengths.index_max();

	for (unsigned int i = 0; i < indices.size() ; ++i) {

		if (midpoint(longest_axis) >= this -> owner -> get_point_coordinates(indices[i]).at(longest_axis)) {
			left_points.push_back(indices[i]);
		}

		else {
			right_points.push_back(indices[i]);
		}

	}

	this -> set_axis(longest_axis);
	this -> set_value(midpoint(longest_axis));

	// I guess this could be avoided
	if (left_points.size() == 0 && right_points.size() > 0) {
		left_points = right_points;
	}

	if (right_points.size() == 0 && left_points.size() > 0) {
		right_points = left_points;
	}


	// Recursion continues
	this -> left -> build(left_points, depth + 1);
	this -> right -> build(right_points, depth + 1);

}

template <template<class> class ContainerType, class PointType>
unsigned int KDTree<ContainerType,PointType>::size() const {
return static_cast<int>(this -> indices.size());
}


template <>
double KDTree<PointCloud,PointNormal>::distance(const PointNormal & point_in_pc,
	const arma::vec & point) const{

	return arma::norm(point_in_pc.get_point_coordinates() - point);

}



template <>
double KDTree<PointCloud,PointDescriptor>::distance(const PointDescriptor & point_in_pc,
	const arma::vec & point) const{

	return point_in_pc.distance_to_descriptor(point);

}

template <>
double KDTree<ShapeModel,ControlPoint>::distance(const ControlPoint & point_in_shape,
	const arma::vec & point) const{

	return arma::norm(point_in_shape.get_point_coordinates() - point);

}

template <template<class> class ContainerType, class PointType> 
void KDTree<ContainerType,PointType>::search_node(const arma::vec & test_point,
	const std::shared_ptr<KDTree> & node,
	int & best_guess_index,
	double & distance) const{

	for (int i = 0; i < node -> indices.size(); ++i){
		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[i]),test_point);
		if (new_distance < distance) {
			distance = new_distance;
			best_guess_index = node -> indices[i];
		}
	}

}

template <template<class> class ContainerType, class PointType> 
void KDTree<ContainerType,PointType>::search_node(const arma::vec & test_point,
	const std::shared_ptr<KDTree> & node,
	const double & distance,
	std::vector< int > & closest_points_indices) const{


	for (int i = 0; i < node -> indices.size(); ++i){
		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[i]),test_point);

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (new_distance < distance ) {
			closest_points_indices.push_back(node -> indices[i]);
			#if KDTTREE_DEBUG_FLAG
			std::cout << "Found closest point " << node -> indices[i] << " with distance = " + std::to_string(distance)<< " \n" << std::endl;
			#endif
		}

	}


}

template <template<class> class ContainerType, class PointType>
void KDTree<ContainerType,PointType>::search_node(const arma::vec & test_point,
	const unsigned int & N_points,
	const std::shared_ptr<KDTree> & node,
	double & distance,
	std::map<double,int > & closest_points) const{

	for (int i =0 ; i < node -> indices.size(); ++i){
		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[i]),test_point);


		#if KDTTREE_DEBUG_FLAG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (closest_points.size() < N_points){
			closest_points[new_distance] = node -> indices[i];
		}
		else{

			unsigned int size_before = closest_points.size(); // should always be equal to N_points

			closest_points[new_distance] = node -> indices[i];

			unsigned int size_after = closest_points.size(); // should always be equal to N_points + 1, unless new_distance was already in the map

			if (size_after == size_before + 1){

				// Remove last element in map
				closest_points.erase(--closest_points.end());

				// Set the distance to that between the query point and the last element in the map
				distance = (--closest_points.end()) -> first;

			}


		}

	}


}



template < template<class> class ContainerType,class PointType>
void KDTree<ContainerType,PointType>::set_max_depth(int max_depth){
	this -> max_depth = max_depth;
}

template < template<class> class ContainerType,class PointType>
int KDTree<ContainerType,PointType>::get_max_depth() const{
	return this -> max_depth;
}

template < template<class> class ContainerType,class PointType>
void KDTree<ContainerType,PointType>::set_min_indices_per_node(int min_indices_per_node){
	this -> min_indices_per_node = min_indices_per_node;
}

template < template<class> class ContainerType,class PointType>
int KDTree<ContainerType,PointType>::get_min_indices_per_node() const{
	return this -> min_indices_per_node;
}


// Explicit instantiations
template class KDTree<PointCloud,PointNormal> ;
template class KDTree<PointCloud,PointDescriptor> ;
template class KDTree<ShapeModel,ControlPoint> ;



