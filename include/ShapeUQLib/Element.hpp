#ifndef HEADER_ELEMENT
#define HEADER_ELEMENT

#include <armadillo>
#include <iostream>
#include <memory>
#include <set>

template <class PointType> 
class ShapeModel;

class ControlPoint;
class Element {

public:

	/**
	Constructor
	*/
	Element(std::vector<int> & control_points,ShapeModel<ControlPoint> * owning_shape);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbors
	@return Pointer to neighboring elements, plus the calling element
	*/
	virtual std::set < int > get_neighbors(bool all_neighbors) const = 0;

	/**
	Recomputes element-specific values
	*/
	void update() ;

	const arma::vec::fixed<3> & get_point_coordinates(int point_index) const;

	/**
	Get element normal. 
	Definition varies depending upon Element type: 
	- the facet normal if the element is a triangular facet
	- the normal evaluated at the center of the element if the element is a Bezier patch
	@return element normal. 
	*/
	arma::vec::fixed<3> get_normal_coordinates() const;

	/**
	Get element center. 
	Definition varies depending upon Element type:
	- vertices average position if the element is a triangular facet
	- Bezier patch evaluated at u == v == w == 1./3 if the element is a Bezier patch
	@return element center
	*/
	arma::vec::fixed<3> get_center() const;


	/**
	Return the control points owned by this element
	@return owned control points
	*/	
	const std::vector<int > & get_points() const;

	void set_control_points(std::vector<int > & control_points);

	void set_owning_shape(ShapeModel<ControlPoint> * owning_shape);

	/**
	Returns number of control points in owning shape
	@return number of control points in owning shape
	*/

	int get_NControlPoints_in_owning_shape() const;

	
	/**
	Gets an eventual super element corresponding to the present element
	@return super element
	*/
	int get_super_element() const;

	/**
	Sets an eventual super element corresponding to the present element
	@param super_element super element to assign
	*/
	void set_super_element(int super_element);

	/**
	Return surface area of element
	@return surface area of element
	*/
	double get_area() const;

	void set_global_index(int i );

	int get_global_index() const;


protected:

	virtual void compute_center() = 0;
	virtual void compute_normal() = 0;
	virtual void compute_area() = 0;


	std::vector<int>  control_points;

	// Element * super_element = nullptr;
	int super_element = -1;

	arma::vec::fixed<3> normal;
	arma::vec::fixed<3> center;

	double area;
	int global_index;

	ShapeModel<ControlPoint> * owning_shape;
};











#endif 