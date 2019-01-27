#ifndef HEADER_CONTROL_POINT
#define HEADER_CONTROL_POINT
#include <armadillo>
#include "Element.hpp"
#include <memory>

#include <set>

class Element;
class ControlPoint;
template <class PointType> class ShapeModel;

class ControlPoint {

public:


	ControlPoint(ShapeModel<ControlPoint> * owning_shape = nullptr);


	/**
	Getter to the vertex's coordinates
	@return coordinates vertex coordinates
	*/
	const arma::vec::fixed<3> & get_point_coordinates() const;


	/**
	Setter to the vertex's coordinates
	@param coordinates vertex coordinates
	*/
	void set_point_coordinates(arma::vec::fixed<3> coordinates);


	/**
	Adds $facet to the vector of std::shared_ptr<Element>  that own this vertex.
	Nothing happens if the facet is already listed
	@param facet index to the element owning this vertex
	*/
	void add_ownership(int el_index);


	arma::vec::fixed<3> get_normal_coordinates(bool bezier) const;


	std::set< int >  common_elements(int control_point_index) const;

	/**
	Determines if $this is owned by $facet
	@param facet Facet whose relationship with the facet is to be tested
	@return true is $this is owned by $facet, false otherwise
	*/
	bool is_owned_by( int el_index) const;



	/**
	Delete $facet from the list of Element * owning $this
	Nothing happens if the facet was not listed (maybe throw a warning)>
	@param facet Pointer to the facet owning this vertex
	*/
	void remove_ownership(int el_index);

	/**
	Removes all ownership relationships 
	*/
	void reset_ownership();


	/**
	Returns the owning elements
	@return Owning elements
	*/
	std::set< int  > get_owning_elements() const;

	/**
	Sets the owning elements
	@param Owning elements
	*/
	void set_owning_elements(std::set< int  > & owning_elements);

	/**
	Returns point covariance
	@return point covariance
	*/
	arma::mat get_covariance() const;

	/**
	@param element owning element
	@param local_indices triplet of indices numbering this control point within the owning element
	*/
	void add_local_numbering(int element,const arma::uvec & local_indices);

	/**
	Returns the local numbering of this control point within the specified element
	@param element pointer to element to consider
	*/
	arma::uvec get_local_numbering(int element) const;

	/**
	Sets the control point covariance
	@param P covariance
	*/
	void set_covariance(arma::mat P);

	void set_deviation(arma::vec d) { this -> deviation = d;}
	arma::vec get_deviation() const { return this -> deviation; }

	/**
	Returns the number of facets owning this vertex
	@return N number of owning of facets
	*/
	unsigned int get_number_of_owning_elements() const ;

	/**
	Get global index (shape wise)
	@return global index
	*/
	int get_global_index() const;


	/**
	Set global index (shape wise)
	@param global index
	*/
	void set_global_index(int index);

protected:
	arma::vec::fixed<3> coordinates;
	arma::vec mean_coordinates;

	std::set<int> owning_elements;
	std::map<int,arma::uvec> local_numbering;
	arma::mat covariance = arma::zeros<arma::mat>(3,3);
	arma::vec deviation = arma::zeros<arma::vec>(3);

	int global_index;
	ShapeModel<ControlPoint> * owning_shape;

};


#endif