#ifndef HEADER_Bezier
#define HEADER_Bezier

#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"
#include "Footpoint.hpp"

#include <boost/math/special_functions/factorials.hpp>
#include <memory>
#include <iostream>
#include <RigidBodyKinematics.hpp>

#include <set>
#include <map>


class ControlPoint;

template <class PointType> 
class ShapeModel;

class Bezier : public Element{

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Bezier(std::vector<int> vertices,ShapeModel<ControlPoint> * owning_shape);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbords
	@return Pointer to neighboring facets, plus the calling facet
	*/
	virtual std::set< int > get_neighbors(bool all_neighbors) const;


	std::set < int > get_neighbors(double u, double v) const;


	/**
	Returns pointer to the first vertex owned by $this that is
	neither $v0 and $v1. When $v0 and $v1 are on the same edge,
	this method returns a pointer to the vertex of $this that is not
	on the edge but still owned by $this
	@param v0 Pointer to first vertex to exclude
	@param v1 Pointer to first vertex to exclude
	@return Pointer to the first vertex of $this that is neither $v0 and $v1
	*/
	int vertex_not_on_edge(
		int v0,
		int v1) const ;

	/**
	Returns patch degree
	@param degree
	*/
	unsigned int get_degree() const;


	
	/**
	Evaluates the bezier patch at the barycentric 
	coordinates (u,v). Note that 0<= u + v <= 1
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return point at the surface of the bezier patch
	*/
	arma::vec::fixed<3> evaluate(const double u, const double v) const;



	/**
	Evaluates the normal of the bezier patch at the barycentric 
	coordinates (u,v). Note that 0<= u + v <= 1
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return point at the surface of the bezier patch
	*/
	arma::vec get_normal_coordinates(const double u, const double v) const;


	/**
	Returns shape fitting residuals standard deviation
	@return standard deviation of fitting residuals
	*/
	double get_fitting_residuals() const;


	/**
	Get index of the queried point
	@param i first index
	@param j second index
	@return local index to control point
	*/
	int get_point_local_index(unsigned int i, unsigned int j) const;

	int get_point_global_index(unsigned int i, unsigned int j) const;



	/**
	Returns the control point given its i and j indices (k = n - i - j)
	@param i first index
	@param j second index
	@return global index of control point
	*/	
	int get_point(unsigned int i, unsigned int j) const;

	/**
	Returns the tuple of local indices (i,j,k) of a control point within a bezier patch
	@param local_index local index of considered point
	@return local_indices (i,j,k) 
	*/
	std::tuple<int,int,int> get_local_indices(int local_index) const;


	/**
	Returns the coordinates of a control point given its i and j indices (k = n - i - j)
	@param i first index
	@param j second index
	@return coordinats of contorl point
	*/	
	const arma::vec::fixed<3> & get_point_coordinates(unsigned int i, unsigned int j) const;


	/**
	Returns P_X
	@return P_X matrix
	*/
	arma::mat get_P_X() const;


	/**
	Evaluates the partial derivative of Sum( B^n_{i,j,k}C_{ijk}) with respect to (u,v) evaluated 
	at (u,v)
	@param u first coordinate
	@param v second coordinate
	*/
	arma::mat::fixed<3,2> partial_bezier(const double u,const double v) const;




	/**
	Returns the 3x3 covariance matrix
	tracking the uncertainty in the location of
	a surface point given uncertainty in the patch's control
	points
	@param u mean of first coordinate
	@param v mean of second coordinate
	@param dir direction of ray
	@param P_X covariance on the position 
	of the control points
	@return 3x3 covariance
	*/
	arma::mat covariance_surface_point_deprecated(
		const double u,
		const double v,
		const arma::vec & dir,
		const arma::mat & P_X);


	/**	
	Sets patch covariance to prescribed value
	@param P_X prescribed value of patch covariance
	*/

	void set_P_X(arma::mat P_X){this -> P_X = P_X;}


	/**
	Returns stored footpoints
	*/
	const std::vector<Footpoint> & get_footpoints() const {return this -> footpoints;}

	/**
	Returns stored shape fitting residuals
	*/
	const std::vector<double> & get_epsilons() const {return this -> epsilons;}

	/**
	Returns stored mapping vectors for covariance training
	*/
	const std::vector<arma::vec> & get_v_i_norm_sq() const{return this ->v_i_norm_sq;}

	/**
	Returns the 3x3 covariance matrix
	tracking the uncertainty in the location of
	a surface point given uncertainty in the patch's control
	points using an alternative formulation
	@param u mean of first coordinate
	@param v mean of second coordinate
	@param dir direction of ray
	of the control points
	@return 3x3 covariance
	*/
	arma::mat::fixed<3,3> covariance_surface_point(
		const double u,
		const double v,
		const arma::vec & dir) const;

	/**
	Returns the 3x3 covariance matrix
	tracking the uncertainty in the location of
	a surface point given uncertainty in the patch's control
	points using an alternative formulation
	@param u mean of first coordinate
	@param v mean of second coordinate
	@param dir direction of ray
	@param P_X covariance on the position 
	of the control points
	@return 3x3 covariance
	*/
	arma::mat::fixed<3,3> covariance_surface_point(
		const double u,
		const double v,
		const arma::vec & dir,
		const arma::mat & P_X) const;



	
	/**
	Computes the patch covariance P_X maximizing the likelihood function 
	associated to the stored footpoints
	*/
	void train_patch_covariance();

	/**
	Sets the covariance parametrization to the prescribed values
	@param covariance_param unique covariance parameters
	*/
	void set_patch_covariance(const std::vector<double> & covariance_param);


	/**
	Returns the triple product of points i_ = (i,j), j_ = = (k,l) and k_ = = (m,p), e.g Ci_^T(Cj_ x Ck_)
	@param i first index of first point
	@param j second index of first point
	@param k first index of second point
	@param l second index of second point
	@param m first index of third point
	@param p second index of third point
	*/
	double triple_product(const int i ,const int j ,const int k ,const int l ,const int m ,const int p ) const;



	/**
	Returns the triple product of points i_ = (i,j), j_ = = (k,l) and k_ = = (m,p), e.g Ci_^T(Cj_ x Ck_)
	@param i first index of first point
	@param j second index of first point
	@param k first index of second point
	@param l second index of second point
	@param m first index of third point
	@param p second index of third point
	*/
	double triple_product(const int i ,const int j ,const int k ,const int l ,const int m ,const int p ,
		const arma::vec & deviation) const;

	/**
	Computes the quadruple product of points i_ = (i,j), j_ = (k,l), k_ = (m,p), l_ = (q,r)  e.g (Ci_^T Cj_) * (Ck_ x Cl_)
	@param result container storing result of computation
	@param i first index of first point
	@param j second index of first point
	@param k first index of second point
	@param l second index of second point
	@param m first index of third point
	@param p second index of third point
	@param q first index of fourth point
	@param r second index of fourth point
	*/
	void quadruple_product(double * result,const int i ,const int j ,const int k ,const int l ,const int m ,const int p, const int q, const int r ) const;



	// Returns the partial derivative d^2P/(dchi dv)
	arma::mat::fixed<3,2> partial_bezier_dv(const double u,const double v) const;


	// Returns the partial derivative d^2P/(dchi du)
	arma::mat::fixed<3,2> partial_bezier_du(const double u,const double v) const;

	/**
	Evaluates the Berstein polynomial
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param j second index
	@param n polynomial order
	@return evaluated bernstein polynomial
	*/
	static double bernstein(
		const double u, 
		const double v,
		const int i,
		const int j,
		const int n) ;



	/**
	Computes the partial derivative of the unit normal vector at the queried point
	with respect to a given control point
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param j second index
	@param n polynomial order
	*/
	arma::mat::fixed<3,3> partial_n_partial_Ck(
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n) const;

	
	static double compute_log_likelihood_block_diagonal(const arma::vec &  L,
		Bezier * args,
		int verbose_level = 0);

	/**
	Add footpoint to Bezier patch for the covariance training phase
	@param footpoint structure holding Ptilde/Pbar/n/u/v
	*/
	void add_footpoint(Footpoint footpoint);

	/**
	Returns true if the patch has training points already assigned, false otherwise
	@return true if the patch has training points already assigned, false otherwise
	*/
	bool has_footpoints() const;


	/**
	Erases the training data
	*/
	void reset_footpoints();


	/**
	Returns the coefficient alpha_ijk for volume computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double alpha_ijk(const int i, const int j, const int k, const int l, const int m, const int p,const int n);

	/**
	Returns the coefficient gamma_ijkl for center of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param q first index of fourth triplet
	@param r second index of fourth triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double gamma_ijkl(const int i, const int j, const int k, const int l, const int m, const int p,const int q, const int r, const int n);


	/**
	Returns the coefficient kappa_ijkl for inertia of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param q first index of fourth triplet
	@param r second index of fourth triplet
	@param s first index of fifth triplet
	@param t second index of fifth triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double kappa_ijklm(const int i, const int j, const int k, const int l, 
		const int m, const int p,const int q, const int r, 
		const int s, const int t, const int n);


	/**
	Returns the coefficient beta_ijkl for center of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double beta_ijkl( const int i,  const int j,  const int k, const  int l,  const int n);



	/**
	Returns the stacked crossed products
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@returm computed stacked cross-product
	*/
	arma::vec get_cross_products(const int i, const int j, const int k, const int l, const int m,const int p) const;


	/**
	Returns the augmented stacked crossed products
	@paran mat reference to matrix holding the stacked crossed products
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param q first index of fourth triplet
	@param r second index of fourth triplet
	@returm computed stacked cross-product
	*/
	void get_augmented_cross_products(arma::mat::fixed<12,3> & mat,const int i, const int j, const int k, const int l, const int m,const int p,
		const int q, const int r) const;

	/**
	Generates the forward table associating a local index l to the corrsponding triplet (i,j,k) 
	@param n degree
	@return forward look up table
	*/
	static std::vector<std::tuple< int,  int,  int> > forward_table( int n);



	/**
	Generates the reverse table associating a local index triplet (i,j,k) to the corresponding
	global index l
	@param n degree
	@return reverse look up table
	*/
	static std::map< std::tuple< int,  int,  int> , int> reverse_table( int n);


	
	/**
	Returns the number of combinations of k items among n.
	Returns 0 if k < 0 or k > n
	@param k subset size
	@param n set size
	@return number of combinations
	*/
	static int combinations(int k, int n);



	/**
	Returns the vector parametrization of the element covariance
	@return parametrization of the element covariance
	*/
	const arma::vec & get_P_X_param() const{return this -> P_X_param;}

protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();



	static double Sa_b(const int a, const int b);
	static double bernstein_coef(const int i , const int j , const int n);


	void construct_index_tables();

	double initialize_covariance();


	/**
	Evaluates the quadrature function for surface area computation
	@param u first barycentric coordinate
	@param v second barycentric coordinate (w = 1 - u - v)
	@param g as in A = \int g du dv 
	*/
	double g(double u, double v) const;



	/**
	Returns the partial derivative of the Bernstein polynomial B^n_{i,j,k} evaluated 
	at (u,v)
	@param u first coordinate
	@param v second coordinate
	@param i first index
	@param j second index
	@param n polynomial degree
	@return evaluated partial derivative of the Bernstein polynomial
	*/
	static arma::rowvec::fixed<2> partial_bernstein(
		const double u,
		const double v,
		const int i,
		const int j,
		const int n) ;


	static arma::rowvec::fixed<2> partial_bernstein_dv( 
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n) ;


	static arma::rowvec::fixed<2> partial_bernstein_du( 
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n) ;


	int n;
	double fitting_residuals = 0;
	double fitting_residuals_mean = 0;

	std::vector<std::tuple< int,  int,  int> > forw_table;
	std::map< std::tuple< int,  int,  int> , int> rev_table;


	arma::mat P_X;
	std::vector<Footpoint> footpoints;
	std::vector<arma::vec> v_i_norm_sq;
	std::vector<double> epsilons;
	arma::vec P_X_param;

};
#endif