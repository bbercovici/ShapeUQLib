#ifndef HEADER_KDTree
#define HEADER_KDTree

#include <memory>
#include <vector>
#include <armadillo>

// template <class T> class PointCloud;

template <template<class> class ContainerType, class PointType>  class KDTree {

public:
	std::shared_ptr<KDTree> left;
	std::shared_ptr<KDTree> right;
	std::vector<int> indices;

	KDTree(ContainerType<PointType> * owner);

	void build(const std::vector< int > & indices, int depth) ;

	void closest_point_search(const arma::vec & test_point,
		const std::shared_ptr<KDTree> & node,
		int & best_guess_index,
		double & distance) const;

	void closest_N_point_search(const arma::vec & test_point,
		const unsigned int & N_points,
		const std::shared_ptr<KDTree> & node,
		double & distance,
		std::map<double, int > & closest_points) const;

	void radius_point_search(const arma::vec & test_point,
		const std::shared_ptr<KDTree> & node,
		const double & distance,
		std::vector< int > & closest_points);

	int get_depth() const;
	void set_depth(int depth);

	unsigned int size() const;

	double get_value() const;
	unsigned int get_axis() const;

	void set_value(double value) ;
	void set_axis(unsigned int axis);

	void set_max_depth(int max_depth);
	int get_max_depth() const;


	void set_min_indices_per_node(int min_indices_per_node);
	int get_min_indices_per_node() const;



protected:

	double distance(const PointType & point_in_pc, const arma::vec & point) const;

	void search_node(const arma::vec & test_point,
		const unsigned int & N_points,
		const std::shared_ptr<KDTree> & node,
		double & distance,
		std::map<double,int > & closest_points) const;

	void search_node(const arma::vec & test_point,
		const std::shared_ptr<KDTree> & node,
		const double & distance,
		std::vector< int > & closest_points_indices) const;

	void search_node(const arma::vec & test_point,
		const std::shared_ptr<KDTree> & node,
		int & best_guess_index,
		double & distance) const;

	int depth;
	int max_depth = 1000;
	int min_indices_per_node = -1;

	double value;
	unsigned int axis = 0;

	ContainerType<PointType> * owner;

};


#endif