#ifndef HEADER_SHAPEMODEL
#define HEADER_SHAPEMODEL

#include <string>
#include <iostream>
#include <armadillo>
#include <set>
#include <map>
#include <limits>
#include <OMP_flags.hpp>

#include <FrameGraph.hpp>
// #include <KDTreeControlPoints.hpp>
// #include <KDTreeShape.hpp>


class Ray ;
class Element;
class KDTreeShape;
template <template<class> class ContainerType, class PointType>  class KDTree ;

/**
Declaration of the ShapeModel class. Base class for 
the implementation of shape model
*/
template <class PointType>
class ShapeModel {

public:

	/**
	Constructor
	*/
	ShapeModel();


	/**
	Constructor
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModel(std::string ref_frame_name,
		FrameGraph * frame_graph);

	virtual void construct_kd_tree_shape() = 0;


	/**
	Returns the dimensions of the bounding box
	@param Bounding box dimension to be computed (xmin,ymin,zmin,xmax,ymax,zmax)
	*/
	void get_bounding_box(double * bounding_box,arma::mat M = arma::eye<arma::mat>(3,3)) const;


	/**
	Translates the shape model by x
	@param x translation vector applied to the coordinates of each control point
	*/
	void translate(const arma::vec::fixed<3> &);

	/**
	Rotates the shape model by 
	@param M rotation matrix
	*/
	void rotate(const arma::mat::fixed<3,3> & M);

	
	/**
	Returns pointer to KDTreeShape member.
	@return pointer to KDTreeShape
	*/
	std::shared_ptr<KDTreeShape> get_KDTreeShape() const ;


	/**
	Shifts the coordinates of the shape model
	so as to have (0,0,0) aligned with its barycenter
	The resulting barycenter coordinates are (0,0,0)
	*/
	void shift_to_barycenter();

	/**
	Applies a rotation that aligns the body
	with its principal axes.
	This assumes that the body has been shifted so
	that (0,0,0) lies at its barycenter
	The resulting inertia tensor is diagonal
	Undefined behavior if
	the inertia tensor has not been computed beforehand
	*/
	void align_with_principal_axes();

	


	/**
	Returns the volume of the provided shape model
	@return volume (U^2 where U is the unit of the shape coordinates)
	*/
	double get_volume() const;


	/**
	Returns the principal axes and principal moments of the shape model
	@param axes M as in X = MX' where X' is a position expressed in the principal frame
	@param moments dimensionless inertia moments in ascending order
	*/	
	void get_principal_inertias(arma::mat & axes,arma::vec & moments) const;



	/**
	Defines the reference frame attached to the shape model
	@param ref_frame Pointer to the reference frame attached
	to the shape model
	*/
	void set_ref_frame_name(std::string ref_frame_name);

	/**
	Returns the name of the reference frame attached to this
	ref frame
	@return name of reference frame
	*/
	std::string get_ref_frame_name() const;


	PointType & get_point(unsigned int i) ;
	const arma::vec::fixed<3> & get_point_coordinates(unsigned int i) const;

	virtual arma::vec::fixed<3> get_point_normal_coordinates(unsigned int i) const = 0;

	/**
	Pointer to the shape model's control points
	@return vertices pointer to the control points
	*/
	const std::vector<PointType> & get_points() const;

	unsigned int get_point_index(std::shared_ptr<PointType> point) const;

	

	/**
	Returns the geometrical center of the shape
	@return geometrical center
	*/
	arma::vec::fixed<3> get_center() const;
	
	
	/**
	Augment the internal container storing vertices with a new (and not already inserted)
	one
	@param control_point pointer to the new control point to be inserted
	*/
	void add_control_point(PointType & control_point);

	virtual void clear() = 0;

	/**
	Returns number of elements
	@return number of elements
	*/
	virtual unsigned int get_NElements() const  = 0;

	/**
	Returns number of control points
	@return number of control points
	*/
	unsigned int get_NControlPoints() const ;


	/**
	Constructs the KDTree holding the facets of the shape model for closest facet detection
	@param verbose true will save the bounding boxes to a file and display
	kd tree construction details
	*/
	void construct_kd_tree_control_points();

	/**
	Computes the surface area of the shape model
	*/
	virtual void compute_surface_area() = 0;
	/**
	Computes the volume of the shape model
	*/
	virtual void compute_volume() = 0;

	/**
	Computes the center of mass of the shape model
	*/
	virtual void compute_center_of_mass() = 0;
	/**
	Computes the inertia tensor of the shape model
	*/
	virtual void compute_inertia() = 0;

	/**
	Finds the intersect between the provided ray and the shape model
	@param ray pointer to ray. If a hit is found, the ray's internal is changed to store the range to the hit point
	*/
	virtual bool ray_trace(Ray * ray,bool outside = true) = 0;



	/**
	Constructs a connectivity table associated a control point pointer to its index
	in this shape model control points vector
	*/
	void initialize_index_table();

	/**
	Returns the non-dimensional inertia tensor of the body in the body-fixed
	principal axes. (rho == 1, l = (volume)^(1/3))
	@return principal inertia tensor
	*/
	const arma::mat::fixed<3,3> & get_inertia() const;

	/**
	Updates shape geometric and mass properties
	*/
	virtual void update_mass_properties() = 0;

	/**
	Returns the surface area of the shape model
	@return surface area (U^2 where U is the unit of the shape coordinates)
	*/
	double get_surface_area() const;

	
	/**
	Returns the location of the center of mass
	@return pointer to center of mass
	*/
	const arma::vec::fixed<3> & get_center_of_mass() const;

	/**
	Builds the covariance of the provided control points
	@param P covariance to set
	@param Ci pointer to first point
	@param Cj pointer to second point
	@param Ck pointer to third point
	@param Cl pointer to fourth point
	@param Cm pointer to fifth point
	@param Cp pointer to sixth point
	*/
	static void assemble_covariance(arma::mat & P,
		const PointType & Ci,
		const PointType & Cj,
		const PointType & Ck,
		const PointType & Cl,
		const PointType & Cm,
		const PointType & Cp);

	/**
	Builds the covariance of the provided control points
	@param P covariance to set
	@param Ci pointer to first point
	@param Cj pointer to second point
	@param Ck pointer to third point
	@param Cl pointer to fourth point
	@param Cm pointer to fifth point
	@param Cp pointer to sixth point
	@param Cq pointer to seventh point
	*/
	static void assemble_covariance(arma::mat & P,
		const PointType & Ci,
		const PointType & Cj,
		const PointType & Ck,
		const PointType & Cl,
		const PointType & Cm,
		const PointType & Cp,
		const PointType & Cq);


	/**
	Builds the covariance of the provided control points
	@param P covariance to set
	@param Ci pointer to first point
	@param Cj pointer to second point
	@param Ck pointer to third point
	@param Cl pointer to fourth point
	@param Cm pointer to fifth point
	@param Cp pointer to sixth point
	@param Cq pointer to seventh point
	@param Cq pointer to eigth point
	*/
	static void assemble_covariance(arma::mat & P,
		const PointType & Ci,
		const PointType & Cj,
		const PointType & Ck,
		const PointType & Cl,
		const PointType & Cm,
		const PointType & Cp,
		const PointType & Cq,
		const PointType & Cr);


	/**
	Returns radius of circumscribing sphere, measured from the shape's center of mass
	@return radius of circumscribing sphere
	*/
	double get_circumscribing_radius() const;



	arma::vec get_inertia_param() const;

	virtual const std::vector<int> & get_element_control_points(int e) const = 0;

protected:

	std::vector<std::set<int> > edges;
	std::vector<PointType> control_points;
	std::shared_ptr<KDTree<ShapeModel,PointType> > kdt_control_points = nullptr;
	std::shared_ptr<KDTreeShape> kdt_facet = nullptr;

	std::map<std::shared_ptr<PointType> ,unsigned int> pointer_to_global_index;

	FrameGraph * frame_graph;
	std::string ref_frame_name;

	arma::mat::fixed<3,3> inertia;
	arma::vec::fixed<3> cm;
	double volume;
	double surface_area;




};

#endif