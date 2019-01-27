
#ifndef HEADER_SHAPEMODELTRI
#define HEADER_SHAPEMODELTRI

#include <string>
#include <string>
#include <iostream>
#include <armadillo>
#include <set>
#include <map>
#include <limits>
#include <cassert>
#include <ShapeModel.hpp>
#include <Facet.hpp>

class Ray;


/**
Declaration of the ShapeModelTri class. Specialized
implementation storing an explicit facet/vertex shape model
*/
template <class PointType>
class ShapeModelTri : public ShapeModel<PointType> {

public:

	
	/**
	Constructor
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModelTri(std::string ref_frame_name,
		FrameGraph * frame_graph) : ShapeModel<PointType>(ref_frame_name,frame_graph){};


	ShapeModelTri(){};

	ShapeModelTri(const std::vector<std::vector<int> > & vertices,
		const std::vector<int> & super_elements,
		const std::vector<PointType> & control_points);

	/**
	Constructs the KDTree holding the shape model for ray-casting purposes
	@param verbose true will save the bounding boxes to a file and display
	kd tree construction details
	*/
	virtual void construct_kd_tree_shape();

	/**
	Determines whether the provided point lies inside or outside the shape model.
	The shape model must have a closed surface for this method to be trusted
	@param point coordinates of the point to be tested expressed in the shape model frame
	@param tol numerical tolerance ,i.e value under which the lagrangian of the "surface field"
		below which the point is considered outside
	@return true if point is contained inside the shape, false otherwise
	*/
	bool contains(double * point, double tol = 1e-6) ;


	/**
	Checks that the normals were consistently oriented. If not,
	the ordering of the vertices in the provided shape model file is incorrect
	@param tol numerical tolerance (if consistent: norm(Sum(oriented_surface_area)) / average_facet_surface_area << tol)
	*/
	void check_normals_consistency(double tol = 1e-3) const;



	/**
	Saves the shape model in the form of an .obj file
	@param path Location of the saved file
	@param X translation component to apply
	@param M rotational component to apply
	*/
	void save(std::string path,
		const arma::vec & X = arma::zeros<arma::vec>(3),
		const arma::mat & M = arma::eye<arma::mat>(3,3)) const;

	/**
	Samples N points over each facet of the shape model
	@param N number of samples per facet
	@param points reference to matrix holding points coordinates
	@param normals reference to matrix holding normals coordinates
	*/
	void random_sampling(unsigned int N,arma::mat & points, arma::mat & normals) const;

	virtual unsigned int get_NElements() const;

	/**
	Updates the values of the center of mass, volume, surface area
	*/
	virtual void update_mass_properties();

	/**
	Update all the facets of the shape model
	*/
	void update_facets() ;

	void add_element(Facet & el);
	void set_elements(std::vector<Facet> elements);


	virtual void clear();




	/**
	Updates the specified facets of the shape model. Ensures consistency between the vertex coordinates
	and the facet surface area, normals and centers.
	@param facets Facets to be updated
	@param compute_dyad true if the facet dyad needs to be computed/updated
	*/
	void update_facets(std::set<Facet *> & facets);


	

	/**
	Computes the surface area of the shape model
	*/
	virtual void compute_surface_area();
	/**
	Computes the volume of the shape model
	*/
	virtual void compute_volume();
	/**
	Computes the center of mass of the shape model
	*/
	virtual void compute_center_of_mass();
	/**
	Computes the inertia tensor of the shape model
	*/
	virtual void compute_inertia();


	Facet & get_element(int e);


	/**
	Finds the intersect between the provided ray and the shape model
	@param ray pointer to ray. If a hit is found, the ray's internal is changed to store the range to the hit point
	*/
	virtual bool ray_trace(Ray * ray,bool outside = true);


	virtual const std::vector<int> & get_element_control_points(int e) const;
	virtual arma::vec::fixed<3> get_point_normal_coordinates(unsigned int i) const;




protected:
	
	std::vector<Facet> elements;



};

#endif