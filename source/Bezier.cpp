#include <ShapeUQLib/Bezier.hpp>
#include <ShapeUQLib/ShapeModel.hpp>





#include <cassert>
#define BEZIER_DEBUG_TRAINING 0

Bezier::Bezier( std::vector<int> vertices,ShapeModel<ControlPoint> * owning_shape) : Element(vertices,owning_shape){

	if (this -> owning_shape -> get_NControlPoints() == 0){
		throw(std::runtime_error("in Bezier::Bezier, the provided owning shape has no control points"));
	}

	double n = (-3 + std::sqrt(9 - 8 * (1 - vertices.size() )))/2;
	double intpart;
	this -> P_X = 1e10 * arma::eye<arma::mat>(3 * this -> control_points.size(),3 * this -> control_points.size());




	if (modf(n,&intpart) == 0){
		this -> n = (unsigned int)(n);
	}
	else{
		throw(std::runtime_error("The control points cardinal does not correspond to a bezier control net"));
	}

	this -> construct_index_tables();

	this -> update();

}

int Bezier::get_point_local_index(unsigned int i, unsigned int j) const{

	std::tuple<int,int,int> indices = std::make_tuple(i,j,this -> n - i - j);

	return this -> rev_table.at(indices) ;
}

int Bezier::get_point_global_index(unsigned int i, unsigned int j) const{

	std::tuple<int,int,int> indices = std::make_tuple(i,j,this -> n - i - j);

	return this -> control_points[this -> rev_table.at(indices)] ;
}



int Bezier::get_point(unsigned int i, unsigned int j) const{
	std::tuple<unsigned int, unsigned int,unsigned int> indices = std::make_tuple(i,j,this -> get_degree() - i - j);
	return this -> control_points[this -> rev_table.at(indices)];
}


const arma::vec::fixed<3> & Bezier::get_point_coordinates(unsigned int i, unsigned int j) const{
	return this -> owning_shape -> get_point_coordinates(this -> get_point(i, j)) ;
}


std::tuple<int, int,int> Bezier::get_local_indices(int i) const{
	return this -> forw_table.at(i);
}




void Bezier::construct_index_tables(){

	this -> rev_table = reverse_table(this -> n);
	this -> forw_table =  forward_table(this -> n);

}


double Bezier::Sa_b(const int a, const int b){
	if (a < 0 || b < 0){
		return 0;
	}
	double sum = 0;

	for (int k = 0; k <= b ; ++k){
		sum += Bezier::combinations(k, b ) * std::pow(-1., k ) / (a + k + 1);
	}
	return sum;

}


double Bezier::triple_product(const int i ,const int j ,const int k ,const int l ,const int m ,const int p ) const{

	std::tuple<unsigned int, unsigned int,unsigned int> i_ = std::make_tuple(i,j,this -> n - i - j);
	std::tuple<unsigned int, unsigned int,unsigned int> j_ = std::make_tuple(k,l,this -> n - k - l);
	std::tuple<unsigned int, unsigned int,unsigned int> k_ = std::make_tuple(m,p,this -> n - m - p);
	double const * Ci =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(i_)]).colptr(0);
	double const * Cj =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(j_)]).colptr(0);
	double const * Ck =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(k_)]).colptr(0);

	return Ci[0] * Cj[1] * Ck[2] + Cj[0] * Ck[1] * Ci[2] + Ck[0] * Ci[1] * Cj[2] -
	Ci[0] * Ck[1] * Cj[2] - Cj[0] * Ci[1] * Ck[2] - Ck[0] * Cj[1] * Ci[2];


}

double Bezier::triple_product(const int i ,const int j ,const int k ,const int l ,const int m ,const int p ,
	const arma::vec & deviation) const{

	std::tuple<unsigned int, unsigned int,unsigned int> i_ = std::make_tuple(i,j,this -> n - i - j);
	std::tuple<unsigned int, unsigned int,unsigned int> j_ = std::make_tuple(k,l,this -> n - k - l);
	std::tuple<unsigned int, unsigned int,unsigned int> k_ = std::make_tuple(m,p,this -> n - m - p);
	
	const ControlPoint & Ci =  this -> owning_shape -> get_point(this -> control_points[this -> rev_table.at(i_)]);
	const ControlPoint & Cj =  this -> owning_shape -> get_point(this -> control_points[this -> rev_table.at(j_)]);
	const ControlPoint & Ck =  this -> owning_shape -> get_point(this -> control_points[this -> rev_table.at(k_)]);

	int i_g = Ci.get_global_index();
	int j_g = Cj.get_global_index();
	int k_g = Ck.get_global_index();

	return arma::dot(Ci.get_point_coordinates() + deviation.rows(3 * i_g,3 * i_g + 2),
		arma::cross(Cj.get_point_coordinates() + deviation.rows(3 * j_g,3 * j_g + 2),
			Ck.get_point_coordinates() + deviation.rows(3 * k_g,3 * k_g + 2)));

}



void Bezier::quadruple_product(double * result,const int i ,const int j ,const int k ,const int l ,const int m ,const int p, const int q, const int r ) const{

	std::tuple<unsigned int, unsigned int,unsigned int> i_ = std::make_tuple(i,j,this -> n - i - j);
	std::tuple<unsigned int, unsigned int,unsigned int> j_ = std::make_tuple(k,l,this -> n - k - l);
	std::tuple<unsigned int, unsigned int,unsigned int> k_ = std::make_tuple(m,p,this -> n - m - p);
	std::tuple<unsigned int, unsigned int,unsigned int> l_ = std::make_tuple(q,r,this -> n - q - r);

	double const * Ci =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(i_)]).colptr(0);
	double const * Cj =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(j_)]).colptr(0);
	double const * Ck =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(k_)]).colptr(0);
	double const * Cl =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(l_)]).colptr(0);


	double cp[3];
	result[0] = Ci[0];
	result[1] = Ci[1];
	result[2] = Ci[2];

	double Cx = Ck[1] * Cl[2] - Ck[2] * Cl[1];
	double Cy = Ck[2] * Cl[0] - Ck[0] * Cl[2];
	double Cz = Ck[0] * Cl[1] - Ck[1] * Cl[0];
	cp[0] = Cx; cp[1] = Cy; cp[2] = Cz;

	double dot_Cj_cp = Cj[0] * cp[0] + Cj[1] * cp[1]+ Cj[2] * cp[2];

	result[0] *= dot_Cj_cp;
	result[1] *= dot_Cj_cp;
	result[2] *= dot_Cj_cp;


}



double Bezier::bernstein_coef(const int i , const int j , const int n){

	if (i < 0  || j < 0 || n < 0 || i > n || j > n || i + j > n){
		return 0;
	}


	return boost::math::factorial<double>(n) / (boost::math::factorial<double>(i) * boost::math::factorial<double>(j) * boost::math::factorial<double>(n - i - j));


}

double Bezier::alpha_ijk(const int i, const int j, const int k, const int l, const int m, const int p,const int n){


	int sum_indices = i + k + j + l + m + p;

	double alpha = double( n * n ) / 3 * Bezier::bernstein_coef(i ,j,n) * (

		Bezier::bernstein_coef(k - 1 ,l,n - 1) * Bezier::bernstein_coef(m ,p -1,n - 1) * Sa_b(l + j + p - 1,3 * n - sum_indices ) * Sa_b(k + m + i - 1,3 * n - i - k - m )
		- Bezier::bernstein_coef(k - 1 ,l,n - 1) * Bezier::bernstein_coef(m ,p,n - 1) * Sa_b(l + j + p,3 * n - sum_indices - 1 ) * Sa_b(k + m + i - 1,3 * n - i - k - m )
		- Bezier::bernstein_coef(k ,l,n - 1) * Bezier::bernstein_coef(m ,p - 1,n - 1) * Sa_b(l + j + p - 1,3 * n - sum_indices - 1 ) * Sa_b(k + m + i,3 * n - i - k - m - 1)
		+ Bezier::bernstein_coef(k ,l,n - 1) * Bezier::bernstein_coef(m ,p,n - 1) * Sa_b(l + j + p,3 * n - sum_indices - 2 ) * Sa_b(k + m + i,3 * n - i - k - m - 1)

		);

	return alpha;
}


double Bezier::gamma_ijkl(const int i, const int j, const int k, const int l, const int m, const int p,const int q, const int r, const int n){

	int sum_indices = i + j + k + l + m + p + q + r;

	double gamma = double( n * n ) / 4 * Bezier::bernstein_coef(i ,j,n) * Bezier::bernstein_coef(k ,l,n) * (

		Bezier::bernstein_coef(m - 1 ,p,n - 1) * ( Bezier::bernstein_coef(q ,r -1,n - 1) * Sa_b(l + j + p  + r- 1,4 * n - sum_indices ) 
			-  Bezier::bernstein_coef(q ,r,n - 1) * Sa_b(l + j + p  + r,4 * n - sum_indices -1)) * Sa_b(k + m + i + q - 1,4 * n - i - k - m - q)
		- Bezier::bernstein_coef(m  ,p,n - 1) *( Bezier::bernstein_coef(q ,r - 1,n - 1) * Sa_b(l + j + p  + r - 1,4 * n - sum_indices -1) 
			-  Bezier::bernstein_coef(q ,r,n - 1) * Sa_b(l + j + p  + r,4 * n - sum_indices -2)) * Sa_b(k + m + i + q,4 * n - i - k - m - q - 1)

		);

	return gamma;


}


double Bezier::kappa_ijklm(const int i, const int j, const int k, const int l, 
	const int m, const int p,const int q, const int r, 
	const int s, const int t, const int n){


	int sum_indices = i + j + k + l + m + p + q + r + s + t;

	double kappa = - double( n * n ) / 5 * Bezier::bernstein_coef(i ,j,n) * Bezier::bernstein_coef(k ,l,n) * Bezier::bernstein_coef(m ,p,n) * (

		Bezier::bernstein_coef(q - 1 ,r,n - 1) * ( Bezier::bernstein_coef(s ,t -1,n - 1) * Sa_b(l + j + r  + t + p- 1,5 * n - sum_indices ) 
			-  Bezier::bernstein_coef(s ,t,n - 1) * Sa_b(l + j + p + r  + t,5 * n - sum_indices -1)) * Sa_b(k + q + i + s + m- 1,5 * n - i - k - q - s - m)
		- Bezier::bernstein_coef(q  ,r,n - 1) *( Bezier::bernstein_coef(s ,t - 1,n - 1) * Sa_b(l + j + p + r  + t - 1,5 * n - sum_indices -1) 
			-  Bezier::bernstein_coef(s ,t,n - 1) * Sa_b(l + j + p + r  + t,5 * n - sum_indices -2)) * Sa_b(k + q + i + s + m,5 * n - i - k - q - s - m - 1)

		);

	return kappa;

}


double Bezier::beta_ijkl(const int i, const int j, const int k, const int l, const int n){


	double beta = 1./ 8 * (
		Bezier::combinations(i, n) 
		* Bezier::combinations(j, n) 
		* Bezier::combinations(k, n) * n* 
		(
			Bezier::combinations( l -1 , n - 1) * Sa_b(i + j + k + l - 1,4 * n - i - j - k - l)
			- Bezier::combinations( l , n - 1) * Sa_b(i + j + k + l, 4 * n - i - j - k - l - 1)));

	return beta;
}


arma::vec Bezier::get_cross_products(const int i, const int j, const int k, const int l, const int m,const int p) const{

	arma::vec stacked_cp(9);

	std::tuple<unsigned int, unsigned int,unsigned int> i_ = std::make_tuple(i,j,this -> n - i - j);
	std::tuple<unsigned int, unsigned int,unsigned int> j_ = std::make_tuple(k,l,this -> n - k - l);
	std::tuple<unsigned int, unsigned int,unsigned int> k_ = std::make_tuple(m,p,this -> n - m - p);

	const arma::vec::fixed<3> & Ci =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(i_)]);
	const arma::vec::fixed<3> & Cj =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(j_)]);
	const arma::vec::fixed<3> & Ck =  this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(k_)]);

	stacked_cp.subvec(0,2) = arma::cross(Cj,Ck);
	stacked_cp.subvec(3,5) = arma::cross(Ck,Ci);
	stacked_cp.subvec(6,8) = arma::cross(Ci,Cj);
	return stacked_cp;

}


void Bezier::get_augmented_cross_products(arma::mat::fixed<12,3> & mat,const int i, const int j, const int k, const int l, const int m,const int p,
	const int q, const int r) const{

	std::tuple<unsigned int, unsigned int,unsigned int> i_ = std::make_tuple(i,j,this -> n - i - j);
	std::tuple<unsigned int, unsigned int,unsigned int> j_ = std::make_tuple(k,l,this -> n - k - l);
	std::tuple<unsigned int, unsigned int,unsigned int> k_ = std::make_tuple(m,p,this -> n - m - p);
	std::tuple<unsigned int, unsigned int,unsigned int> l_ = std::make_tuple(q,r,this -> n - q - r);


	const arma::vec::fixed<3> & Ci = this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(i_)]);
	const arma::vec::fixed<3> & Cj = this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(j_)]);
	const arma::vec::fixed<3> & Ck = this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(k_)]);
	const arma::vec::fixed<3> & Cl = this -> owning_shape -> get_point_coordinates(this -> control_points[this -> rev_table.at(l_)]);

	arma::vec::fixed<3> v_kl = arma::cross(Ck,Cl);
	arma::vec::fixed<3> v_lj = arma::cross(Cl,Cj);
	arma::vec::fixed<3> v_jk = arma::cross(Cj,Ck);

	mat.submat(0,0,2,2) = arma::eye<arma::mat>(3,3) * arma::dot(Cj,v_kl);
	mat.submat(3,0,5,2) = v_kl * Ci.t();
	mat.submat(6,0,8,2) = v_lj * Ci.t();
	mat.submat(9,0,11,2) = v_jk * Ci.t();


}


unsigned int Bezier::get_degree() const{
	return this -> n;
}




std::set < int > Bezier::get_neighbors(double u, double v) const{


	const ControlPoint & V0 = this -> owning_shape -> get_point(this -> get_point(this -> get_degree(),0));
	const ControlPoint & V1 = this -> owning_shape -> get_point(this -> get_point(0,this -> get_degree()));


	if ( v < 0){
		return V0.common_elements(this -> get_point(0,0));
	}

	if (1 - u - v < 0){
		return V0.common_elements(this -> get_point(0,this -> get_degree()));
	}


	if (u < 0){
		return V1.common_elements(this -> get_point(0,0));
	}

	else{
		return this -> get_neighbors(true);
	}


}


std::set < int > Bezier::get_neighbors(bool all_neighbors) const{

	std::set<int > neighbors;
	const std::tuple<int,int,int> V0_tuple = std::make_tuple(this -> n,0,0);
	const std::tuple<int,int,int> V1_tuple = std::make_tuple(0,this -> n,0);
	const std::tuple<int,int,int> V2_tuple = std::make_tuple(0,0,this -> n);

	int V0_index = this -> control_points[this -> rev_table.at(V0_tuple)];
	int V1_index = this -> control_points[this -> rev_table.at(V1_tuple)];
	int V2_index = this -> control_points[this -> rev_table.at(V2_tuple)];


	const ControlPoint & V0 = this -> owning_shape -> get_point(V0_index);
	const ControlPoint & V1 = this -> owning_shape -> get_point(V1_index);
	const ControlPoint & V2 = this -> owning_shape -> get_point(V2_index);


	if (all_neighbors == true) {
		// Returns all facets sharing control_points with $this

		std::set<int> V0_owners = V0.get_owning_elements();
		std::set<int> V1_owners = V1.get_owning_elements();
		std::set<int> V2_owners = V2.get_owning_elements();


		for (auto facet_it = V0_owners.begin();facet_it != V0_owners.end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V1_owners.begin();facet_it != V1_owners.end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V2_owners.begin();facet_it != V2_owners.end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}


	}

	else {
		// Returns facets sharing edges with $this
		std::set<int > neighboring_facets_e0 = V0.common_elements(V1_index);
		std::set<int > neighboring_facets_e1 = V1.common_elements(V2_index);
		std::set<int > neighboring_facets_e2 = V2.common_elements(V0_index);

		for (auto it = neighboring_facets_e0.begin(); it != neighboring_facets_e0.end(); ++it) {
			neighbors.insert(*it);
		}

		for (auto it = neighboring_facets_e1.begin(); it != neighboring_facets_e1.end(); ++it) {
			neighbors.insert(*it);
		}

		for (auto it = neighboring_facets_e2.begin(); it != neighboring_facets_e2.end(); ++it) {
			neighbors.insert(*it);
		}
	}
	return neighbors;

}

arma::vec::fixed<3> Bezier::evaluate(const double u, const double v) const{

	arma::vec P = arma::zeros<arma::vec>(3);

	for (unsigned int l = 0; l < this -> control_points.size(); ++l){


		int i = std::get<0>(this -> forw_table[l]);
		int j = std::get<1>(this -> forw_table[l]);

		P += this -> bernstein(u,v,i,j,this -> n) * this -> owning_shape -> get_point_coordinates(this -> control_points[l]);

	}

	return P;
}



arma::vec Bezier::get_normal_coordinates(const double u, const double v) const{
	arma::mat partials = this -> partial_bezier(u,v);
	return arma::normalise(arma::cross(partials.col(0),partials.col(1)));
}


double Bezier::bernstein(
	const double u, 
	const double v,
	const int i,
	const int j,
	const int n) {

	if (i < 0 || i > n || j < 0 || j > n || i + j > n){
		return 0;
	}

	if (n == 0){
		return 1;
	}

	return Bezier::bernstein_coef(i,j,n) * (std::pow(u,i) *  std::pow(v,j)
		* std::pow(1 - u - v, n - i - j ));

}

arma::rowvec::fixed<2> Bezier::partial_bernstein( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::rowvec::fixed<2> partials = {
		n *( bernstein(u, v,i - 1,j,n - 1) - bernstein(u, v,i,j,n - 1)),
		n *( bernstein(u, v,i,j - 1,n - 1) - bernstein(u, v,i,j,n - 1))
	};
	return partials;
}


arma::rowvec::fixed<2> Bezier::partial_bernstein_du( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::rowvec::fixed<2> partials = n * ( partial_bernstein(u, v,i - 1,j,n - 1) - partial_bernstein(u, v,i,j,n - 1));




	return partials;
}

arma::rowvec::fixed<2> Bezier::partial_bernstein_dv( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::rowvec::fixed<2> partials = n * ( partial_bernstein(u, v,i,j-1,n - 1) - partial_bernstein(u, v,i,j,n - 1));

	return partials;
}

arma::mat::fixed<3,2> Bezier::partial_bezier_du(
	const double u,
	const double v) const {

	arma::mat::fixed<3,2> partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		
		int i = std::get<0>(this -> forw_table[l]);
		
		int j = std::get<1>(this -> forw_table[l]);

		partials += this -> owning_shape -> get_point_coordinates(this -> control_points[l]) * Bezier::partial_bernstein_du(u,v,i,j,this -> n) ;
	}

	return partials;

}

arma::mat::fixed<3,2> Bezier::partial_bezier_dv(
	const double u,
	const double v) const {

	arma::mat::fixed<3,2> partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		
		int i = std::get<0>(this -> forw_table[l]);
		
		int j = std::get<1>(this -> forw_table[l]);

		partials += this -> owning_shape -> get_point_coordinates(this -> control_points[l]) * Bezier::partial_bernstein_du(u,v,i,j,this -> n) ;
	}
	return partials;

}


arma::mat::fixed<3,2> Bezier::partial_bezier(
	const double u,
	const double v) const{

	arma::mat::fixed<3,2> partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){	
		int i = std::get<0>(this -> forw_table.at(l));
		int j = std::get<1>(this -> forw_table.at(l));

		partials += this -> owning_shape -> get_point_coordinates(this -> control_points.at(l)) * Bezier::partial_bernstein(u,v,i,j,this -> n) ;
	}
	return partials;

}


arma::mat::fixed<3,3> Bezier::partial_n_partial_Ck(const double u, const double v,const int i ,  const int j, const int n) const{



	auto P_chi = this -> partial_bezier(u,v);
	auto P_u = P_chi.col(0);
	auto P_v = P_chi.col(1);

	double norm = arma::norm(arma::cross(P_u,P_v));
	arma::mat P_u_tilde = RBK::tilde(P_u);
	arma::mat P_v_tilde = RBK::tilde(P_v);


	arma::rowvec dBernstein_chi = Bezier::partial_bernstein(u,v,i,j,n);


	return (1./norm * (arma::eye<arma::mat>(3,3) - P_u_tilde * P_v * arma::cross(P_u,P_v).t() / std::pow(norm,2))
		* (P_u_tilde * dBernstein_chi(1) - P_v_tilde * dBernstein_chi(0)));

}



arma::mat Bezier::get_P_X() const{
	return this -> P_X;
}




void Bezier::compute_normal(){

}

void Bezier::compute_area(){

	// The area is computed by quadrature
	// arma::vec weights = {-27./96.,25./96,25./96,25./96};
	// arma::vec u = {1./3.,1./5.,1./5,3./5};
	// arma::vec v = {1./3.,1./5.,3./5,1./5};

	// this -> area = 0;
	// for (int i = 0; i < weights.n_rows; ++i){
	// 	this -> area += weights(i) * g(u(i),v(i));
	// }

}

void Bezier::compute_center(){
	this -> center = this -> evaluate(1./3.,1./3.);

}


int Bezier::combinations(int k, int n){

	if (k < 0 || k > n || n < 0){
		return 0;
	}


	return boost::math::factorial<double>(n) / (boost::math::factorial<double>(k)  * boost::math::factorial<double>(n - k));

}



std::map< std::tuple< int,  int,  int> , int> Bezier::reverse_table( int n){

	std::map< std::tuple< int,  int,  int>, int> map;
	unsigned int l = 0;

	for (int i = n; i > -1 ; -- i){

		for ( int k = 0 ; k < n + 1 - i; ++k){
			
			int j = n - i - k;
			auto indices = std::make_tuple(i,j,k);
			map[indices] = l;

			l = l + 1;
		}
	}

	return map;

}

std::vector<std::tuple< int,  int,  int> > Bezier::forward_table( int n){

	std::vector<std::tuple< int,  int,  int> > table;


	for (int i = n; i > -1 ; -- i){

		for ( int k = 0 ; k < n + 1 - i; ++k){
			int j = n - i - k;
			auto indices = std::make_tuple(i,j,k);

			table.push_back(indices);
		}
	}

	return table;
}



void Bezier::set_patch_covariance(const std::vector<double> & covariance_param){


	unsigned int N_C = this -> control_points.size();
	this -> P_X_param.resize(covariance_param.size());
	for (int i =0; i < covariance_param.size(); ++i){
		this -> P_X_param(i) = covariance_param[i];
	}


	this -> P_X = arma::diagmat(arma::exp(arma::vectorise(arma::repmat(this -> P_X_param,1,3),1)));



}








