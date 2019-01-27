#ifndef HEADER_FACET
#define HEADER_FACET
#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"

#include <memory>
#include <iostream>

#include <set>

class ControlPoint;
template <class PointType> class ShapeModel;

class Facet : public Element{

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Facet( std::vector<int> & vertices,ShapeModel<ControlPoint> * owning_shape);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbords
	@return Pointer to neighboring facets, plus the calling facet
	*/
	virtual std::set< int > get_neighbors(bool all_neighbors) const;

	const arma::vec::fixed<3> & get_normal_coordinates() const;

	/**
	Returns pointer to the first vertex owned by $this that is
	neither $v0 and $v1. When $v0 and $v1 are on the same edge,
	this method returns a pointer to the vertex of $this that is not
	on the edge but still owned by $this
	@param v0 Pointer to first vertex to exclude
	@param v1 Pointer to first vertex to exclude
	@return Pointer to the first vertex of $this that is neither $v0 and $v1
	*/
	int vertex_not_on_edge(int v0,
		int v1) const ;


protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();


	unsigned int split_counter = 0;
	unsigned int hit_count = 0;

};
#endif