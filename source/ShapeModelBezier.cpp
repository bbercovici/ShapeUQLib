#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "KDTreeShape.hpp"

#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::vec>(6) )

#pragma omp declare reduction (+ : arma::mat::fixed<3,3> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::mat>(3,3) )


#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::mat>(6,6) )


template <class PointType>
ShapeModelBezier<PointType>::ShapeModelBezier(std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel<PointType>(ref_frame_name,frame_graph){

}

template <class PointType>
ShapeModelBezier<PointType>::ShapeModelBezier(const ShapeModelTri<PointType> & shape_model,
	std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel<PointType>(ref_frame_name,frame_graph){

	// All the facets of the original shape model are browsed
	// The shape starts as a uniform union of order-2 Bezier patches

	// The control point of this shape model are the same as that
	// of the provided shape
	this -> control_points = shape_model.get_points();

	// The ownership relationships are reset
	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		this -> control_points[i].reset_ownership();
		assert(this -> control_points[i].get_global_index() == i);
	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (unsigned int i = 0; i < shape_model.get_NElements(); ++i){
		
		Bezier patch(shape_model.get_element_control_points(i),this);
		auto control_points_indices = patch.get_points();
		patch.set_global_index(i);

		this -> elements.push_back(patch);

		this -> control_points[control_points_indices[0]].add_ownership(i);
		this -> control_points[control_points_indices[1]].add_ownership(i);
		this -> control_points[control_points_indices[2]].add_ownership(i);

	}

	this -> construct_kd_tree_control_points();
	this -> populate_mass_properties_coefs_deterministics();
	this -> populate_mass_properties_coefs_stochastics();

	this -> assemble_mapping_matrices();
	this -> update_mass_properties();


}


template <class PointType>
std::shared_ptr<arma::mat> ShapeModelBezier<PointType>::get_info_mat_ptr() const{
	return this -> info_mat_ptr;
}

template <class PointType>
std::shared_ptr<arma::vec> ShapeModelBezier<PointType>::get_dX_bar_ptr() const{
	return this -> dX_bar_ptr;
}

template <class PointType>
void ShapeModelBezier<PointType>::initialize_info_mat(){
	unsigned int N = this -> control_points.size();
	this -> info_mat_ptr = std::make_shared<arma::mat>(arma::eye<arma::mat>(3 * N,3 * N));
}

template <class PointType>
void ShapeModelBezier<PointType>::initialize_dX_bar(){
	unsigned int N = this -> control_points.size();
	this -> dX_bar_ptr = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3 * N));
}


template <class PointType>
arma::mat ShapeModelBezier<PointType>::random_sampling(unsigned int N,const arma::mat & R) const{

	arma::mat points = arma::zeros<arma::mat>(3,N);
	arma::mat S = arma::chol(R,"lower");

	// N points are randomly sampled from the surface of the shape model
	// #pragma omp parallel for
	for (unsigned int i = 0; i < N; ++i){

		unsigned int element_index = arma::randi<arma::vec>( 1, arma::distr_param(0,this -> elements.size() - 1) ) (0);

		const Bezier & patch = this -> elements[element_index];
		arma::vec random = arma::randu<arma::vec>(2);
		double u = random(0);
		double v = (1 - u) * random(1);

		points.col(i) = patch.evaluate(u,v) + S * arma::randn<arma::vec>(3) ;

	}

	return points;

}


template <class PointType>
void ShapeModelBezier<PointType>::compute_surface_area(){
	throw(std::runtime_error("Warning: should only be used for post-processing\n"));

}

template <class PointType>
void ShapeModelBezier<PointType>::update_mass_properties() {
	
	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();

	this -> compute_volume();
	this -> compute_center_of_mass();
	this -> compute_inertia();

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time in ShapeModelBezier<PointType>::update_mass_properties: " << elapsed_seconds.count() << " s"<< std::endl;


}

template <class PointType>
void ShapeModelBezier<PointType>::compute_volume(){
	
	double volume = 0;

	#pragma omp parallel for reduction(+:volume) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		const Bezier & patch = this -> elements[el_index];	
		
		for (int index = 0 ; index <  this -> volume_indices_coefs_table.size(); ++index) {

			int i =  int(this -> volume_indices_coefs_table[index][0]);
			int j =  int(this -> volume_indices_coefs_table[index][1]);
			int k =  int(this -> volume_indices_coefs_table[index][2]);
			int l =  int(this -> volume_indices_coefs_table[index][3]);
			int m =  int(this -> volume_indices_coefs_table[index][4]);
			int p =  int(this -> volume_indices_coefs_table[index][5]);
			
			volume += this -> volume_indices_coefs_table[index][6] * patch.triple_product(i,j,k,l,m,p);

		}

	}
	this -> volume = volume;
}


template <class PointType>
double ShapeModelBezier<PointType>::compute_volume(const arma::vec & deviation) const{
	
	double volume = 0;

	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		const Bezier & patch = this -> elements[el_index];	
		
		for (int index = 0 ; index <  this -> volume_indices_coefs_table.size(); ++index) {

			int i =  int(this -> volume_indices_coefs_table[index][0]);
			int j =  int(this -> volume_indices_coefs_table[index][1]);
			int k =  int(this -> volume_indices_coefs_table[index][2]);
			int l =  int(this -> volume_indices_coefs_table[index][3]);
			int m =  int(this -> volume_indices_coefs_table[index][4]);
			int p =  int(this -> volume_indices_coefs_table[index][5]);
			
			volume += this -> volume_indices_coefs_table[index][6] * patch . triple_product(i,j,k,l,m,p,deviation);

		}

	}
	return volume;
}


template <class PointType>
unsigned int ShapeModelBezier<PointType>::get_NElements() const {
	return this -> elements . size();
}




template <class PointType>
void ShapeModelBezier<PointType>::compute_center_of_mass(){

	this -> cm = arma::zeros<arma::vec>(3);

	double cx = 0;
	double cy = 0;
	double cz = 0;

	#pragma omp parallel for reduction(+:cx,cy,cz)
	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {

		const Bezier & patch = this -> elements[el_index];	

		for (auto index = 0 ; index <  this -> cm_gamma_indices_coefs_table.size(); ++index) {

			int i =  int(this -> cm_gamma_indices_coefs_table[index][0]);
			int j =  int(this -> cm_gamma_indices_coefs_table[index][1]);
			int k =  int(this -> cm_gamma_indices_coefs_table[index][2]);
			int l =  int(this -> cm_gamma_indices_coefs_table[index][3]);
			int m =  int(this -> cm_gamma_indices_coefs_table[index][4]);
			int p =  int(this -> cm_gamma_indices_coefs_table[index][5]);
			int q =  int(this -> cm_gamma_indices_coefs_table[index][6]);
			int r =  int(this -> cm_gamma_indices_coefs_table[index][7]);


			double result[3];

			patch. quadruple_product(result,i ,j ,k ,l ,m ,p, q, r );

			cx += this -> cm_gamma_indices_coefs_table[index][8] * result[0];
			cy += this -> cm_gamma_indices_coefs_table[index][8] * result[1];
			cz += this -> cm_gamma_indices_coefs_table[index][8] * result[2];


		}

	}

	this -> cm = {cx,cy,cz};
	this -> cm = this -> cm / this -> volume;
}

template <class PointType>
arma::vec::fixed<3> ShapeModelBezier<PointType>::compute_center_of_mass(const double & volume,const arma::vec & deviation) const{

	double cx = 0;
	double cy = 0;
	double cz = 0;

	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {

		const Bezier & patch = this -> elements[el_index];	

		for (auto index = 0 ; index <  this -> cm_gamma_indices_coefs_table.size(); ++index) {

			int i =  int(this -> cm_gamma_indices_coefs_table[index][0]);
			int j =  int(this -> cm_gamma_indices_coefs_table[index][1]);
			int k =  int(this -> cm_gamma_indices_coefs_table[index][2]);
			int l =  int(this -> cm_gamma_indices_coefs_table[index][3]);
			int m =  int(this -> cm_gamma_indices_coefs_table[index][4]);
			int p =  int(this -> cm_gamma_indices_coefs_table[index][5]);
			int q =  int(this -> cm_gamma_indices_coefs_table[index][6]);
			int r =  int(this -> cm_gamma_indices_coefs_table[index][7]);

			int i_g = patch.get_point(i,j);

			const ControlPoint & Ci = this -> control_points[i_g];
			
			
			arma::vec result = (Ci . get_point_coordinates() 
				+ deviation.rows(3 * i_g,3 * i_g + 2)) * patch . triple_product(k,l,m,p,q,r,deviation);

			cx += this -> cm_gamma_indices_coefs_table[index][8] * result(0);
			cy += this -> cm_gamma_indices_coefs_table[index][8] * result(1);
			cz += this -> cm_gamma_indices_coefs_table[index][8] * result(2);


		}

	}

	arma::vec cm = {cx,cy,cz};

	return ( cm / volume);
}

template <class PointType>
void ShapeModelBezier<PointType>::compute_inertia(){

	arma::mat::fixed<3,3> inertia = arma::zeros<arma::mat>(3,3);

	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		const Bezier & patch = this -> elements[el_index];	
		
		for (int index = 0 ; index <  this -> inertia_indices_coefs_table.size(); ++index) {

			int i =  int(this -> inertia_indices_coefs_table[index][0]);
			int j =  int(this -> inertia_indices_coefs_table[index][1]);
			int k =  int(this -> inertia_indices_coefs_table[index][2]);
			int l =  int(this -> inertia_indices_coefs_table[index][3]);
			int m =  int(this -> inertia_indices_coefs_table[index][4]);
			int p =  int(this -> inertia_indices_coefs_table[index][5]);
			int q =  int(this -> inertia_indices_coefs_table[index][6]);
			int r =  int(this -> inertia_indices_coefs_table[index][7]);
			int s =  int(this -> inertia_indices_coefs_table[index][8]);
			int t =  int(this -> inertia_indices_coefs_table[index][9]);
			
			inertia += (this -> inertia_indices_coefs_table[index][10]  
				* RBK::tilde(patch.get_point_coordinates(i,j)) 
				* RBK::tilde(patch.get_point_coordinates(k,l))
				* patch.triple_product(m,p,q,r,s,t));
		}

	}
	this -> inertia = inertia;

}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::compute_inertia(const arma::vec & deviation) const{

	arma::mat::fixed<3,3> inertia = arma::zeros<arma::mat>(3,3);

	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		const Bezier & patch = this -> elements[el_index];	
		
		for (int index = 0 ; index <  this -> inertia_indices_coefs_table.size(); ++index) {

			int i =  int(this -> inertia_indices_coefs_table[index][0]);
			int j =  int(this -> inertia_indices_coefs_table[index][1]);
			int k =  int(this -> inertia_indices_coefs_table[index][2]);
			int l =  int(this -> inertia_indices_coefs_table[index][3]);
			int m =  int(this -> inertia_indices_coefs_table[index][4]);
			int p =  int(this -> inertia_indices_coefs_table[index][5]);
			int q =  int(this -> inertia_indices_coefs_table[index][6]);
			int r =  int(this -> inertia_indices_coefs_table[index][7]);
			int s =  int(this -> inertia_indices_coefs_table[index][8]);
			int t =  int(this -> inertia_indices_coefs_table[index][9]);

			int i_g = patch . get_point(i,j);
			int j_g = patch . get_point(k,l);

			inertia += (this -> inertia_indices_coefs_table[index][10]  
				* RBK::tilde(this -> control_points[i_g].get_point_coordinates() + deviation.rows(3 * i_g, 3 * i_g + 2)) 
				* RBK::tilde(this -> control_points[j_g].get_point_coordinates() + deviation.rows(3 * j_g, 3 * j_g + 2))
				* patch.triple_product(m,p,q,r,s,t,deviation));

		}

	}
	return inertia;

}

template <class PointType>
arma::vec::fixed<3> ShapeModelBezier<PointType>::get_point_normal_coordinates(unsigned int i) const{

	auto owning_elements = this -> control_points[i].get_owning_elements();
	arma::vec::fixed<3> n = {0,0,0};


	for (auto e : owning_elements){

		const std::vector<int> & control_points = this -> elements[e].get_points();
		
		int local_index = -1;
		for (int k =0 ; k < control_points.size(); ++k){
			if (control_points[k] == i){
				local_index = k;
			}
		}

		assert(local_index >= 0);



		auto local_tuple = this -> elements[e].get_local_indices(local_index);


		double u = std::get<0>(local_tuple) / this -> get_degree();
		double v = std::get<1>(local_tuple) / this -> get_degree();


		n += this -> elements[e].get_normal_coordinates(u,v);


	}


	return arma::normalise(n);

}







template <class PointType>
const std::vector<int> & ShapeModelBezier<PointType>::get_element_control_points(int e) const{
	return this -> elements[e].get_points();
}

template <class PointType>
void ShapeModelBezier<PointType>::find_correlated_elements(){

	this -> correlated_elements.clear();
	this -> correlated_elements.resize(this -> get_NElements());

	for (unsigned int e = 0; e < this -> get_NElements(); ++e){

		std::vector < int > elements_correlated_with_e ;

		const Bezier & patch_e = this -> elements[e];
		const std::vector<int> & patch_e_control_points = patch_e.get_points();

		for (unsigned int f = 0; f < this -> get_NElements(); ++f){

			const Bezier & patch_f = this -> elements[f];
			const std::vector<int> & patch_f_control_points = patch_f.get_points();

			for (unsigned int i = 0; i < patch_e_control_points.size(); ++i){
				int i_g = patch_e_control_points[i];

				for (unsigned int j = 0; j < patch_f_control_points.size(); ++j){

					int j_g = patch_f_control_points[j];

					// If true, these two patches are correlated
					double max_value;
					max_value = arma::abs(this -> get_point_covariance(i_g,j_g)).max();
					if (max_value > 0){
						if (std::find(elements_correlated_with_e.begin(),elements_correlated_with_e.end(),f) == elements_correlated_with_e.end() ){
							elements_correlated_with_e.push_back(f);
						}
					}

					

				}

				


			}

		}

		this -> correlated_elements[e] = elements_correlated_with_e;

	}




}


template <class PointType>
double ShapeModelBezier<PointType>::get_volume_sd() const{
	return this -> volume_sd;
}

template <class PointType>
arma::mat ShapeModelBezier<PointType>::get_cm_cov() const{
	return this -> cm_cov;
}

template <class PointType>
void ShapeModelBezier<PointType>::compute_all_statistics(){

	double vol_sd_temp = 0;
	arma::mat::fixed<3,3> cm_cov_temp = arma::zeros<arma::mat>(3,3);
	arma::mat::fixed<6,6> P_I_temp = arma::zeros<arma::mat>(6,6);
	arma::vec::fixed<6> P_M_I_temp = arma::zeros<arma::vec>(6);


	std::vector<std::pair<int ,int > > connected_elements;
	for (unsigned int e = 0; e < this -> elements.size(); ++e) {
		for (const auto el : this -> correlated_elements[e]){
			connected_elements.push_back(std::make_pair(e,el));
		}
	}

	this -> save_connectivity(connected_elements);

	std::cout << "\n- Computing all statistics over the " << connected_elements.size() << " surface element combinations ...\n";

	boost::progress_display progress(connected_elements.size()) ;
	
	#if !__APPLE__
	#pragma omp parallel for reduction(+:vol_sd_temp), reduction(+:cm_cov_temp), reduction(+:P_I_temp), reduction(+:P_M_I_temp)
	#endif

	for (unsigned int k = 0; k < connected_elements.size(); ++k) {

		const Bezier & patch_e = this -> elements[connected_elements[k].first];
		const Bezier & patch_f = this -> elements[connected_elements[k].second];

		vol_sd_temp += this -> compute_patch_pair_vol_sd_contribution(patch_e,patch_f) ;

		cm_cov_temp += this -> compute_patch_pair_cm_cov_contribution(patch_e,patch_f) ;
		P_I_temp += this -> compute_patch_pair_PI_contribution(patch_e,patch_f);
		P_M_I_temp += this -> compute_patch_pair_P_MI_contribution(patch_e,patch_f);
		++progress;

	}


	this -> volume_sd = std::sqrt(vol_sd_temp);
	this -> cm_cov = cm_cov_temp / std::pow(this -> volume,2);
	this -> P_MI = P_M_I_temp;
	this -> P_I = P_I_temp;


	this -> compute_P_MX();
	this -> compute_P_Y();
	this -> compute_P_moments();
	this -> compute_P_dims();


	this -> compute_P_Evectors();
	this -> compute_P_eigenvectors();
	this -> compute_P_sigma();


	this -> save_connectivity(connected_elements);



}

template <class PointType>
void ShapeModelBezier<PointType>::save_connectivity(const std::vector< std::pair<int,int> > & connected_elements) const{


	arma::mat connectivity_mat(this -> elements.size(),this -> elements.size());
	connectivity_mat.fill(arma::datum::nan);

	for (int k = 0; k < connected_elements.size(); ++k){
		auto pair = connected_elements[k];

		connectivity_mat(pair.first,pair.second) = arma::norm(
			this -> elements[pair.first].get_center()
			- this -> elements[pair.second].get_center()
			);


	}


	connectivity_mat.save("connectivity_mat.txt",arma::raw_ascii);

}










template <class PointType>
double ShapeModelBezier<PointType>::compute_patch_pair_vol_sd_contribution(const Bezier & patch_e,const Bezier & patch_f) const{

	double d_vol_sd = 0;
	arma::mat::fixed<9,9> P_CC = arma::zeros<arma::mat>(9,9);

	for (int index = 0 ; index <  this -> volume_sd_indices_coefs_table.size(); ++index) {
				// i
		int i =  int(this -> volume_sd_indices_coefs_table[index][0]);
		int j =  int(this -> volume_sd_indices_coefs_table[index][1]);

				// j
		int k =  int(this -> volume_sd_indices_coefs_table[index][2]);
		int l =  int(this -> volume_sd_indices_coefs_table[index][3]);

				// k
		int m =  int(this -> volume_sd_indices_coefs_table[index][4]);
		int p =  int(this -> volume_sd_indices_coefs_table[index][5]);

				// l
		int q =  int(this -> volume_sd_indices_coefs_table[index][6]);
		int r =  int(this -> volume_sd_indices_coefs_table[index][7]);

				// m
		int s =  int(this -> volume_sd_indices_coefs_table[index][8]);
		int t =  int(this -> volume_sd_indices_coefs_table[index][9]);

				// p
		int u =  int(this -> volume_sd_indices_coefs_table[index][10]);
		int v =  int(this -> volume_sd_indices_coefs_table[index][11]);


		int i_g = patch_e.get_point_global_index(i,j);
		int j_g = patch_e.get_point_global_index(k,l);
		int k_g = patch_e.get_point_global_index(m,p);

		int l_g = patch_f.get_point_global_index(q,r);
		int m_g = patch_f.get_point_global_index(s,t);
		int p_g = patch_f.get_point_global_index(u,v);

		

		const std::array<int,6> patch_e_indices_array = {i,j,k,l,m,p};
		const std::array<int,6> patch_f_indices_array = {q,r,s,t,u,v};

		const arma::vec::fixed<9> & left_vec = this -> elements_to_volume_mapping_matrices[patch_e.get_global_index()].at(patch_e_indices_array);
		const arma::vec::fixed<9> & right_vec = this -> elements_to_volume_mapping_matrices[patch_f.get_global_index()].at(patch_f_indices_array);

		d_vol_sd += this -> volume_sd_indices_coefs_table[index][12] * this -> increment_volume_variance(P_CC,left_vec,	right_vec, i_g, j_g, k_g, l_g,  m_g, p_g);

	}
	return d_vol_sd;

}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::compute_patch_pair_cm_cov_contribution(const Bezier & patch_e,const Bezier & patch_f)  const{

	int i_g,j_g,k_g,l_g;
	int m_g,p_g,q_g,r_g;


	arma::mat::fixed<3,3> d_cm_cov = arma::zeros<arma::mat>(3,3);
	arma::mat::fixed<12,12> P_CC;

	for (int index = 0 ; index <  this -> cm_cov_1_indices_coefs_table.size(); ++index) {

		const std::vector<double> & coefs_row = this -> cm_cov_1_indices_coefs_table[index];

				// i
		int i =  int(coefs_row[0]);
		int j =  int(coefs_row[1]);

				// j
		int k =  int(coefs_row[2]);
		int l =  int(coefs_row[3]);

				// k
		int m =  int(coefs_row[4]);
		int p =  int(coefs_row[5]);

				// l
		int q =  int(coefs_row[6]);
		int r =  int(coefs_row[7]);

				// m
		int s =  int(coefs_row[8]);
		int t =  int(coefs_row[9]);

				// p
		int u =  int(coefs_row[10]);
		int v =  int(coefs_row[11]);

				// q
		int w =  int(coefs_row[12]);
		int x =  int(coefs_row[13]);

				// r
		int y =  int(coefs_row[14]);
		int z =  int(coefs_row[15]);


		i_g = patch_e. get_point_global_index(i,j);
		j_g = patch_e. get_point_global_index(k,l);
		k_g = patch_e. get_point_global_index(m,p);
		l_g = patch_e. get_point_global_index(q,r);

		m_g = patch_f. get_point_global_index(s,t);
		p_g = patch_f. get_point_global_index(u,v);
		q_g = patch_f. get_point_global_index(w,x);
		r_g = patch_f. get_point_global_index(y,z);	


		const std::array<int,8> patch_e_indices_array = {i,j,k,l,m,p,q,r};
		const std::array<int,8> patch_f_indices_array = {s,t,u,v,w,x,y,z,};

		const arma::mat::fixed<12,3> & left_mat = this -> elements_to_cm_mapping_matrices[patch_e.get_global_index()].at((patch_e_indices_array));
		const arma::mat::fixed<12,3> & right_mat = this -> elements_to_cm_mapping_matrices[patch_f.get_global_index()].at((patch_f_indices_array));

		d_cm_cov += coefs_row[16] * this -> increment_cm_cov(P_CC,left_mat,right_mat, i_g,j_g,k_g,l_g, m_g,p_g,q_g,r_g);
	}
	return d_cm_cov;

}





template <class PointType>
arma::mat::fixed<6,6> ShapeModelBezier<PointType>::compute_patch_pair_PI_contribution(const Bezier & patch_e,const Bezier & patch_f) const{

	int i_g,j_g,k_g,l_g,m_g;
	int p_g,q_g,r_g,s_g,t_g;

	
	arma::mat::fixed<6,6> d_P_I = arma::zeros<arma::mat>(6,6);
	arma::mat::fixed<15,15> P_CC = arma::zeros<arma::mat>(15,15);

	i_g=j_g=k_g=l_g=m_g=p_g=q_g=r_g=s_g=t_g= 0;

	for (unsigned int index = 0 ; index <  this -> P_I_indices_coefs_table.size(); ++index) {

		const std::vector<double> & coefs_row = this -> P_I_indices_coefs_table[index];

				// i
		int i =  int(coefs_row[0]);
		int j =  int(coefs_row[1]);

				// j
		int k =  int(coefs_row[2]);
		int l =  int(coefs_row[3]);

				// k
		int m =  int(coefs_row[4]);
		int p =  int(coefs_row[5]);

				// l
		int q =  int(coefs_row[6]);
		int r =  int(coefs_row[7]);

				// m
		int s =  int(coefs_row[8]);
		int t =  int(coefs_row[9]);

				// p
		int u =  int(coefs_row[10]);
		int v =  int(coefs_row[11]);

				// q
		int w =  int(coefs_row[12]);
		int x =  int(coefs_row[13]);

				// r
		int y =  int(coefs_row[14]);
		int z =  int(coefs_row[15]);

				// s
		int a =  int(coefs_row[16]);
		int b =  int(coefs_row[17]);

				// t
		int c =  int(coefs_row[18]);
		int d =  int(coefs_row[19]);


		i_g = patch_e. get_point_global_index(i,j);
		j_g = patch_e. get_point_global_index(k,l);
		k_g = patch_e. get_point_global_index(m,p);
		l_g = patch_e. get_point_global_index(q,r);
		m_g = patch_e. get_point_global_index(s,t);


		p_g = patch_f. get_point_global_index(u,v);
		q_g = patch_f. get_point_global_index(w,x);
		r_g = patch_f. get_point_global_index(y,z);
		s_g = patch_f. get_point_global_index(a,b);
		t_g = patch_f. get_point_global_index(c,d);	

		const std::array<int,10> patch_e_indices_array = {i,j,k,l,m,p,q,r,s,t};
		const std::array<int,10> patch_f_indices_array = {u,v,w,x,y,z,a,b,c,d};

		const arma::mat::fixed<6,15> & left_mat = this -> elements_to_inertia_mapping_matrices[patch_e.get_global_index()].at(patch_e_indices_array);
		const arma::mat::fixed<6,15> & right_mat = this -> elements_to_inertia_mapping_matrices[patch_f.get_global_index()].at(patch_f_indices_array);


		d_P_I += coefs_row[20] * this -> increment_P_I(P_CC,left_mat,right_mat, i_g,j_g,k_g,l_g,m_g, p_g,q_g,r_g,s_g,t_g);

	}

	return d_P_I;
}

template <class PointType>
arma::vec::fixed<6> ShapeModelBezier<PointType>::compute_patch_pair_P_MI_contribution(const Bezier & patch_e,const Bezier & patch_f) const{

	int i_g,j_g,k_g,l_g,m_g;
	int p_g,q_g,r_g;

	arma::vec::fixed<6> d_P_MI = arma::zeros<arma::vec>(6);
	arma::mat::fixed<15,9> P_CC;
	for (int index = 0 ; index <  this -> P_MI_indices_coefs_table.size(); ++index) {

		const std::vector<double> & coefs_row = this -> P_MI_indices_coefs_table[index];

				// i
		int i =  int(coefs_row[0]);
		int j =  int(coefs_row[1]);

				// j
		int k =  int(coefs_row[2]);
		int l =  int(coefs_row[3]);

				// k
		int m =  int(coefs_row[4]);
		int p =  int(coefs_row[5]);

				// l
		int q =  int(coefs_row[6]);
		int r =  int(coefs_row[7]);

				// m
		int s =  int(coefs_row[8]);
		int t =  int(coefs_row[9]);

				// p
		int u =  int(coefs_row[10]);
		int v =  int(coefs_row[11]);

				// q
		int w =  int(coefs_row[12]);
		int x =  int(coefs_row[13]);

				// r
		int y =  int(coefs_row[14]);
		int z =  int(coefs_row[15]);


		i_g = patch_e. get_point_global_index(i,j);
		j_g = patch_e. get_point_global_index(k,l);
		k_g = patch_e. get_point_global_index(m,p);
		l_g = patch_e. get_point_global_index(q,r);
		m_g = patch_e. get_point_global_index(s,t);	

		p_g = patch_f. get_point_global_index(u,v);
		q_g = patch_f. get_point_global_index(w,x);
		r_g = patch_f. get_point_global_index(y,z);


		const std::array<int,10> patch_e_indices_array = {i,j,k,l,m,p,q,r,s,t};
		const arma::mat::fixed<6,15> & left_mat = this -> elements_to_inertia_mapping_matrices[patch_e.get_global_index()].at(patch_e_indices_array);

		const std::array<int,6> patch_f_indices_array = {u,v,w,x,y,z};
		const arma::vec::fixed<9> & right_vec = this -> elements_to_volume_mapping_matrices[patch_f.get_global_index()].at(patch_f_indices_array);


		d_P_MI += coefs_row[16] *  this -> increment_P_MI(P_CC,left_mat,right_vec, i_g,j_g,k_g,l_g,m_g, p_g,q_g,r_g);
	}

	return d_P_MI;
}


template <class PointType>
double ShapeModelBezier<PointType>::increment_volume_variance(arma::mat::fixed<9,9> & P_CC,
	const arma::vec::fixed<9> & left_vec,
	const arma::vec::fixed<9> & right_vec, 
	int i,int j,int k, 
	int l, int m, int p) const {

	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, l);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, m);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, p);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, l);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, m);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, p);

	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, l);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, m);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, p);

	return arma::dot(left_vec, P_CC * right_vec);
}


template <class PointType>
void ShapeModelBezier<PointType>::construct_cm_mapping_mat(arma::mat::fixed<12,3> & mat,
	int i,int j,int k,int l) const {

	const arma::vec::fixed<3> & Ci = this -> control_points[i].get_point_coordinates();
	const arma::vec::fixed<3> & Cj = this -> control_points[j].get_point_coordinates();
	const arma::vec::fixed<3> & Ck = this -> control_points[k].get_point_coordinates();
	const arma::vec::fixed<3> & Cl = this -> control_points[l].get_point_coordinates();

	mat.submat(0,0,2,2) = arma::eye<arma::mat>(3,3) * arma::dot(Cj,arma::cross(Ck,Cl));
	mat.submat(3,0,5,2) = arma::cross(Ck,Cl) * Ci. t();
	mat.submat(6,0,8,2) = arma::cross(Cl,Cj) * Ci. t();
	mat.submat(9,0,11,2) = arma::cross(Cj,Ck) * Ci. t();



}


template <class PointType>
void ShapeModelBezier<PointType>::construct_inertia_mapping_mat(arma::mat::fixed<6,15> & mat,
	int i,int j,int k,int l,int m) const{

	const arma::vec::fixed<3> & Ci = this -> control_points[i].get_point_coordinates();
	const arma::vec::fixed<3> & Cj = this -> control_points[j].get_point_coordinates();
	const arma::vec::fixed<3> & Ck = this -> control_points[k].get_point_coordinates();
	const arma::vec::fixed<3> & Cl = this -> control_points[l].get_point_coordinates();
	const arma::vec::fixed<3> & Cm = this -> control_points[m].get_point_coordinates();

	mat.row(0) = ShapeModelBezier<PointType>::L_row(0,0,Ci,Cj,Ck,Cl,Cm);
	mat.row(1) = ShapeModelBezier<PointType>::L_row(1,1,Ci,Cj,Ck,Cl,Cm);
	mat.row(2) = ShapeModelBezier<PointType>::L_row(2,2,Ci,Cj,Ck,Cl,Cm);
	mat.row(3) = ShapeModelBezier<PointType>::L_row(0,1,Ci,Cj,Ck,Cl,Cm);
	mat.row(4) = ShapeModelBezier<PointType>::L_row(0,2,Ci,Cj,Ck,Cl,Cm);
	mat.row(5) = ShapeModelBezier<PointType>::L_row(1,2,Ci,Cj,Ck,Cl,Cm);

	// mat.fill(1);
}

template <class PointType>
arma::rowvec::fixed<15> ShapeModelBezier<PointType>::L_row(int q, int r, 
	const arma::vec::fixed<3> & Ci,
	const arma::vec::fixed<3> & Cj,
	const arma::vec::fixed<3> & Ck,
	const arma::vec::fixed<3> & Cl,
	const arma::vec::fixed<3> & Cm){


	arma::vec L_col(15);
	arma::vec e_q = arma::zeros<arma::vec>(3);
	e_q(q) = 1;
	arma::vec e_r = arma::zeros<arma::vec>(3);
	e_r(r) = 1;

	L_col.rows(0,2) = - arma::dot(Ck,arma::cross(Cl,Cm)) * RBK::tilde(e_r) * RBK::tilde(Cj) * e_q;
	L_col.rows(3,5) = - arma::dot(Ck,arma::cross(Cl,Cm)) * RBK::tilde(e_q) * RBK::tilde(Ci) * e_r;
	L_col.rows(6,8) = arma::dot(e_r, RBK::tilde(Ci) * RBK::tilde(Cj) * e_q ) * arma::cross(Cl,Cm);
	L_col.rows(9,11) = arma::dot(e_r, RBK::tilde(Ci) * RBK::tilde(Cj) * e_q ) * arma::cross(Cm,Ck);
	L_col.rows(12,14) = arma::dot(e_r, RBK::tilde(Ci) * RBK::tilde(Cj) * e_q ) * arma::cross(Ck,Cl);

	return L_col.t();
}




template <class PointType>
void ShapeModelBezier<PointType>::run_monte_carlo(int N,
	arma::vec & results_volume,
	arma::mat & results_cm,
	arma::mat & results_inertia,
	arma::mat & results_moments,
	arma::mat & results_mrp,
	arma::mat & results_lambda_I,
	arma::mat & results_eigenvectors,
	arma::mat & results_Evectors,
	arma::mat & results_Y,
	arma::mat & results_MI,
	arma::mat & results_dims,
	std::string output_path){

	arma::arma_rng::set_seed(0);

	results_volume = arma::zeros<arma::vec>(N);
	results_cm = arma::zeros<arma::mat>(3,N);
	results_inertia = arma::zeros<arma::mat>(6,N);
	results_moments = arma::zeros<arma::mat>(4,N);
	results_mrp = arma::zeros<arma::mat>(3,N);
	results_lambda_I = arma::zeros<arma::mat>(7,N);
	results_eigenvectors = arma::zeros<arma::mat>(9,N);
	results_Evectors = arma::zeros<arma::mat>(9,N);
	results_Y = arma::zeros<arma::mat>(4,N);
	results_MI = arma::zeros<arma::mat>(7,N);
	results_dims = arma::zeros<arma::mat>(3,N);


	this -> take_and_save_slice(2,output_path + "/slice_z_baseline.txt",0);
	this -> take_and_save_slice(1,output_path + "/slice_y_baseline.txt",0);
	this -> take_and_save_slice(0,output_path + "/slice_x_baseline.txt",0);


	this -> save_to_obj(output_path + "/iter_baseline.obj");

	arma::mat all_deviations(3 * this -> get_NControlPoints(),N);

	std::cout << "Drawing random deviations...\n";
	boost::progress_display progress_display_deviations(N) ;

	for (int iter = 0; iter < N; ++iter){
		arma::vec deviation = this -> shape_covariance_sqrt * arma::randn<arma::vec>(3 * this -> get_NControlPoints());	
		all_deviations.col(iter) = deviation;
		++progress_display_deviations;
	}

	boost::progress_display progress(N) ;
	#pragma omp parallel for
	for (int iter = 0; iter < N; ++iter){

		const arma::vec & deviation = all_deviations.col(iter);

		// Should be able to provide deviation in control points 
		// here
		const double volume = this -> compute_volume(deviation);
		const arma::vec::fixed<3> center_of_mass = this -> compute_center_of_mass(volume,deviation);
		const arma::mat::fixed<3,3> inertia  = this -> compute_inertia(deviation);

		arma::mat I_C = inertia - volume * RBK::tilde(center_of_mass) * RBK::tilde(center_of_mass).t() ;
		arma::vec I = {I_C(0,0),I_C(1,1),I_C(2,2),I_C(0,1),I_C(0,2),I_C(1,2)};

		arma::vec moments_col(4);
		arma::vec eig_val = arma::eig_sym(I_C);
		arma::mat eig_vec =  ShapeModelBezier<PointType>::get_principal_axes_stable(I_C);
		moments_col.rows(0,2) = eig_val;


		moments_col(3) = volume;
		results_volume(iter) = volume;
		results_cm.col(iter) = center_of_mass;

		results_inertia.col(iter) = I;
		results_moments.col(iter) = moments_col;
		results_mrp.col(iter) = RBK::dcm_to_mrp(eig_vec);

		results_lambda_I.col(iter).rows(0,5) = I;
		results_lambda_I.col(iter)(6) = eig_val(2);

		results_eigenvectors.col(iter).rows(0,2) = eig_vec.col(0);
		results_eigenvectors.col(iter).rows(3,5) = eig_vec.col(1);
		results_eigenvectors.col(iter).rows(6,8) = eig_vec.col(2);
		results_Evectors.col(iter) = ShapeModelBezier<PointType>::get_E_vectors(I_C);
		results_Y.col(iter) = ShapeModelBezier<PointType>::get_Y(volume,I_C);
		results_MI.col(iter).rows(0,5) = I;
		results_MI.col(iter)(6) = volume;

		results_dims.col(iter) = ShapeModelBezier<PointType>::get_dims(volume,I_C);

		// saving shape model

		if (iter < 20){
			this -> take_and_save_slice(2,output_path + "/slice_z_" + std::to_string(iter) + ".txt",0,deviation);
			this -> take_and_save_slice(1,output_path + "/slice_y_" + std::to_string(iter) + ".txt",0,deviation);
			this -> take_and_save_slice(0,output_path + "/slice_x_" + std::to_string(iter) + ".txt",0,deviation);
			this -> save_to_obj(output_path + "/iter_" + std::to_string(iter) + ".obj");
		}
		++progress;

	}
	
}


template <class PointType>
arma::vec::fixed<3> ShapeModelBezier<PointType>::get_dims(const double & volume,
	const arma::mat::fixed<3,3> & I_C){


	arma::vec moments = arma::eig_sym(I_C);

	double A = moments(0);
	double B = moments(1);
	double C = moments(2);

	arma::vec::fixed<3> dims = {
		std::sqrt(B + C - A),
		std::sqrt(A + C - B),
		std::sqrt(A + B - C)
	};

	return std::sqrt(5./(2 * volume)) * dims;



}


template <class PointType>
bool ShapeModelBezier<PointType>::ray_trace(Ray * ray,bool outside){

	return this -> kdt_facet -> hit(this -> get_KDTreeShape(),ray,false,this);
	
}

template <class PointType>
void ShapeModelBezier<PointType>::assemble_mapping_matrices(){

	this -> elements_to_volume_mapping_matrices.clear();
	this -> elements_to_cm_mapping_matrices.clear();
	this -> elements_to_inertia_mapping_matrices.clear();

	this -> elements_to_volume_mapping_matrices.resize(this -> elements.size());
	this -> elements_to_cm_mapping_matrices.resize(this -> elements.size());
	this -> elements_to_inertia_mapping_matrices.resize(this -> elements.size());

	std::cout << "Assembling mapping matrices\n";
	for (int e = 0; e < this -> elements.size(); ++e){

		const Bezier & patch_e = this -> elements[e];
		
		// Volume
		for (int c = 0; c < static_cast<int>(this -> volume_indices_coefs_table.size()); ++c){

			int i = static_cast<int>(this -> volume_indices_coefs_table[c][0]);
			int j = static_cast<int>(this -> volume_indices_coefs_table[c][1]);
			int k = static_cast<int>(this -> volume_indices_coefs_table[c][2]);
			int l = static_cast<int>(this -> volume_indices_coefs_table[c][3]);
			int m = static_cast<int>(this -> volume_indices_coefs_table[c][4]);
			int p = static_cast<int>(this -> volume_indices_coefs_table[c][5]);


			std::array<int, 6> array = {i,j,k,l,m,p};


			this -> elements_to_volume_mapping_matrices[e][array] = patch_e.get_cross_products(i,j,k,l,m,p);
		}

		// Center of mass
		for (int c = 0; c < static_cast<int>(this -> cm_gamma_indices_coefs_table.size()); ++c){

			int i = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][0]);
			int j = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][1]);
			int k = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][2]);
			int l = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][3]);
			int m = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][4]);
			int p = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][5]);
			int q = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][6]);
			int r = static_cast<int>(this -> cm_gamma_indices_coefs_table[c][7]);

			std::array<int, 8> array = {i,j,k,l,m,p,q,r};


			int i_g = patch_e. get_point_global_index(i,j);
			int j_g = patch_e. get_point_global_index(k,l);
			int k_g = patch_e. get_point_global_index(m,p);
			int l_g = patch_e. get_point_global_index(q,r);
			arma::mat::fixed<12,3> left_mat;

			this -> construct_cm_mapping_mat(left_mat,i_g,j_g,k_g,l_g);

			this -> elements_to_cm_mapping_matrices[e][array] = left_mat;
		}

		// Inertia
		for (int c = 0; c < static_cast<int>(this -> inertia_indices_coefs_table.size()); ++c){

			int i = static_cast<int>(this -> inertia_indices_coefs_table[c][0]);
			int j = static_cast<int>(this -> inertia_indices_coefs_table[c][1]);
			int k = static_cast<int>(this -> inertia_indices_coefs_table[c][2]);
			int l = static_cast<int>(this -> inertia_indices_coefs_table[c][3]);
			int m = static_cast<int>(this -> inertia_indices_coefs_table[c][4]);
			int p = static_cast<int>(this -> inertia_indices_coefs_table[c][5]);
			int q = static_cast<int>(this -> inertia_indices_coefs_table[c][6]);
			int r = static_cast<int>(this -> inertia_indices_coefs_table[c][7]);
			int s = static_cast<int>(this -> inertia_indices_coefs_table[c][8]);
			int t = static_cast<int>(this -> inertia_indices_coefs_table[c][9]);
			

			std::array<int, 10> array = {i,j,k,l,m,p,q,r,s,t};

			int i_g = patch_e. get_point_global_index(i,j);
			int j_g = patch_e. get_point_global_index(k,l);
			int k_g = patch_e. get_point_global_index(m,p);
			int l_g = patch_e. get_point_global_index(q,r);
			int m_g = patch_e. get_point_global_index(s,t);

			arma::mat::fixed<6,15> left_mat;

			this -> construct_inertia_mapping_mat(left_mat,i_g,j_g,k_g,l_g,m_g);

			this -> elements_to_inertia_mapping_matrices[e][array] = left_mat;
		}




	}


}



template <class PointType>
void ShapeModelBezier<PointType>::populate_mass_properties_coefs_stochastics(){

	
	this -> volume_sd_indices_coefs_table.clear();
	this ->	cm_cov_1_indices_coefs_table.clear();
	this -> cm_cov_2_indices_coefs_table.clear();
	this -> P_I_indices_coefs_table.clear();
	this -> P_MI_indices_coefs_table.clear();

	int n = this -> get_degree();
	std::cout << "- Shape degree: " << n << std::endl;


	std::vector<std::vector<int> >  base_vector;
	ShapeModelBezier<PointType>::build_bezier_base_index_vector(n,base_vector);
	std::vector<std::vector<std::vector<int> > > index_vectors;

	
	// Volume sd
	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(6,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];

		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];

		double alpha_1 = Bezier::alpha_ijk(i, j, k, l, m, p, n);
		double alpha_2 = Bezier::alpha_ijk(q, r, s, t, u, v, n);
		double aa = alpha_1 * alpha_2;
		if (std::abs(aa) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				aa
			};
			this -> volume_sd_indices_coefs_table.push_back(index_vector);

		}

	}


	std::cout << "- Volume SD coefficients: " << this -> volume_sd_indices_coefs_table.size() << std::endl;

	


	// CM covar, 1


	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(8,
		base_vector,
		index_vectors);



	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];


		double gamma_1 = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
		double gamma_2 = Bezier::gamma_ijkl(s, t, u, v, w, x,y, z, n);

		double gamma = gamma_1 * gamma_2;

		if (std::abs(gamma) > 0){

			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				double(y),double(z),
				gamma
			};
			this -> cm_cov_1_indices_coefs_table.push_back(index_vector);
		}

	}






		// CM covar, 2

	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(7,
		base_vector,
		index_vectors);



	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];


		double gamma = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
		double alpha = Bezier::alpha_ijk(s, t, u, v, w, x, n);

		double coef = gamma * alpha;

		if (std::abs(coef) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				coef
			};
			this -> cm_cov_2_indices_coefs_table.push_back(index_vector);
		}

	}
	std::cout << "- CM cov coefficients : " << this -> cm_cov_1_indices_coefs_table.size() + this -> cm_cov_2_indices_coefs_table.size() << std::endl;



	// Inertia statistics 1

	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(10,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];
		int a = vector[8][0];
		int b = vector[8][1];
		int c = vector[9][0];
		int d = vector[9][1];

		double kappa_kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n) * Bezier::kappa_ijklm(u, v, w, x, y, z,a, b,c,d, n);
		if (std::abs(kappa_kappa) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				double(y),double(z),
				double(a),double(b),
				double(c),double(d),
				kappa_kappa
			};
			this -> P_I_indices_coefs_table.push_back(index_vector);
		}
	}

	arma::vec cm_cov_coefs_arma(this -> cm_cov_1_indices_coefs_table.size());
	for (int i = 0; i < this -> cm_cov_1_indices_coefs_table.size(); ++i){
		cm_cov_coefs_arma(i) =  this -> cm_cov_1_indices_coefs_table[i].back();
	}
	cm_cov_coefs_arma.save("cm_cov_coefs_arma.txt",arma::raw_ascii);

	arma::vec inertia_cov_coefs_arma(this -> P_I_indices_coefs_table.size());
	
	for (int i = 0; i < this -> P_I_indices_coefs_table.size(); ++i){
		inertia_cov_coefs_arma(i) =  this -> P_I_indices_coefs_table[i].back();
	}
	inertia_cov_coefs_arma.save("inertia_cov_coefs_arma.txt",arma::raw_ascii);


	// Inertia statistics 2

	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(8,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];


		double alpha_kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n) * Bezier::alpha_ijk(u, v, w, x, y, z, n);

		if (std::abs(alpha_kappa) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				double(y),double(z),
				alpha_kappa
			};
			this -> P_MI_indices_coefs_table.push_back(index_vector);
		}
	}


	std::cout << "- P_I stats coefficients: " << this -> P_I_indices_coefs_table.size() << std::endl;
	std::cout << "- P_MI stats coefficients: " <<  this -> P_MI_indices_coefs_table.size() << std::endl;

}


template <class PointType>
void ShapeModelBezier<PointType>::populate_mass_properties_coefs_deterministics(){

	this -> cm_gamma_indices_coefs_table.clear();
	this -> volume_indices_coefs_table.clear();
	this -> inertia_indices_coefs_table.clear();


	int n = this -> get_degree();
	std::cout << "- Shape degree: " << n << std::endl;


	std::vector<std::vector<int> >  base_vector;
	ShapeModelBezier<PointType>::build_bezier_base_index_vector(n,base_vector);
	std::vector<std::vector<std::vector<int> > > index_vectors;

	// Volume
	ShapeModelBezier<PointType>::build_bezier_index_vectors(3,
		base_vector,
		index_vectors);


	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];

		double alpha = Bezier::alpha_ijk(i, j, k, l, m, p, n);
		if(std::abs(alpha) > 0){
			std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),alpha};
			this -> volume_indices_coefs_table.push_back(index_vector);
		}
	}

	std::cout << "- Volume coefficients: " << this -> volume_indices_coefs_table.size() << std::endl;

	// CM
	// i


	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(4,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		double gamma = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
		if (std::abs(gamma) > 0){
			std::vector<double> index_vector = {
				double(i),
				double(j),
				double(k),
				double(l),
				double(m),
				double(p),
				double(q),
				double(r),
				gamma
			} ;
			this -> cm_gamma_indices_coefs_table.push_back(index_vector);
		}

	}


	std::cout << "- CM coefficients: " << this -> cm_gamma_indices_coefs_table.size() << std::endl;

	// Inertia
	index_vectors.clear();
	ShapeModelBezier<PointType>::build_bezier_index_vectors(5,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		double kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n);

		if (std::abs(kappa) > 0){

			std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
				double(s),double(t),kappa
			};

			this -> inertia_indices_coefs_table.push_back(index_vector);
		}
	}



	std::cout << "- Inertia coefficients: " << this -> inertia_indices_coefs_table.size() << std::endl;

}


template <class PointType>
void ShapeModelBezier<PointType>::build_bezier_index_vectors(const int & n_indices,
	const std::vector<std::vector<int> > & base_vector,
	std::vector<std::vector<std::vector<int> > > & index_vectors,
	std::vector < std::vector<int> > temp_vector,
	const int depth){

	if (temp_vector.size() == 0){
		for ( int i = 0; i < n_indices; ++i ){
			temp_vector.push_back(std::vector<int>());
		}
	}

	for (unsigned int i = 0; i < base_vector.size(); ++i ){

		temp_vector[depth] = base_vector[i];

		if (depth == n_indices - 1 || n_indices == 1){
			index_vectors.push_back(temp_vector);
		}
		else{
			build_bezier_index_vectors(n_indices,base_vector,index_vectors,temp_vector,
				depth + 1);
		}

	}

}




template <class PointType>
void ShapeModelBezier<PointType>::compute_point_covariances(double sigma_sq,double correl_distance) {

	double epsilon = 1e-2;

	this -> point_covariances.clear();
	this -> point_covariances.resize(this -> get_NControlPoints());
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> point_covariances[i].resize(this -> get_NControlPoints());

	}

	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){


		arma::vec::fixed<3> ni = this -> get_point_normal_coordinates(i);

		arma::vec::fixed<3> u_2 = arma::randn<arma::vec>(3);
		u_2 = arma::normalise(arma::cross(ni,u_2));

		arma::vec u_1 = arma::cross(u_2,ni);
		arma::mat::fixed<3,3> P = sigma_sq * (ni * ni.t() + epsilon * (u_1 * u_1.t() + u_2 * u_2.t()));

		std::vector<int> Pi_cor_indices = {int(i)};
		std::vector<arma::mat::fixed<3,3>> Pi_cor = {P};

		this -> point_covariances[i][i] = P;

		for (unsigned int j = i + 1; j < this -> get_NControlPoints(); ++j){


			arma::vec::fixed<3> nj = this -> get_point_normal_coordinates(j);

			double distance = arma::norm(this ->get_point_coordinates(i)
				- this ->get_point_coordinates(j));

			if ( distance < 3 * correl_distance){
				double decay = std::exp(- std::pow(distance / correl_distance,2)) ;

				arma::mat::fixed<3,3> P_correlated = sigma_sq * decay * ni * nj.t();

				this -> point_covariances[i][j] = P_correlated;
				this -> point_covariances[j][i] = P_correlated.t();

			}
			else{

				this -> point_covariances[i][j] = arma::zeros<arma::mat>(3,3);
				this -> point_covariances[j][i] = arma::zeros<arma::mat>(3,3);

			}


		}


	}

	std::cout << "Finding correlated elements in shape\n";
	this -> find_correlated_elements();
	std::cout << "Done finding correlated elements in shape\n";


}

template <class PointType>
const arma::mat::fixed<3,3> & ShapeModelBezier<PointType>::get_point_covariance(int i, int j) const {


	return this -> point_covariances.at(i).at(j);

}

template <class PointType>
void ShapeModelBezier<PointType>::compute_shape_covariance_sqrt(){


	this -> shape_covariance_sqrt = arma::zeros<arma::mat>(3 * this -> get_NControlPoints(),
		3 * this -> get_NControlPoints());

	arma::mat shape_covariance_arma(3 * this -> get_NControlPoints(),
		3 * this -> get_NControlPoints());


	for (int i = 0; i < this -> get_NControlPoints(); ++i){
		for (int j = 0; j < this -> get_NControlPoints(); ++j){
			shape_covariance_arma.submat(3 * i,3 * j,3 * i + 2, 3 * j + 2)  = this -> get_point_covariance(i,j);
		}
	}

	arma::vec eig_val;
	arma::mat eig_vec;

	arma::eig_sym(eig_val,eig_vec,shape_covariance_arma);


	// Regularizing the eigenvalue decomposition
	double min_val = arma::abs(eig_val).min();
	for (int i = 0; i < eig_val.n_rows; ++i){
		if (eig_val(i) < 0){
			eig_val(i) = min_val;
		}
	}

	this -> shape_covariance_sqrt = eig_vec * arma::diagmat(arma::sqrt(eig_val)) * eig_vec.t();

	// std::ofstream file("cov_mat.txt");
	// std::ofstream file_sqrt("sqrt.txt");

	// file.precision(15);
	// file_sqrt.precision(15);

	// shape_covariance_arma.raw_print(file);
	// this -> shape_covariance_sqrt.raw_print(file_sqrt);

}

template <class PointType>
void ShapeModelBezier<PointType>::set_elements(std::vector<Bezier> elements){
	this -> elements = elements;
}

template <class PointType>
void ShapeModelBezier<PointType>::clear(){
	this -> control_points.clear();
	this -> elements.clear();
}

template <class PointType>
void ShapeModelBezier<PointType>::build_bezier_base_index_vector(const int n,std::vector<std::vector<int> > & base_vector){

	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

			std::vector<int> pair = {i,j};
			base_vector.push_back(pair);
		}
	}

}


template <class PointType>
void ShapeModelBezier<PointType>::save_both(std::string partial_path){

	this -> save(partial_path + ".b");

	ShapeModelBezier self = (*this);

	self.elevate_degree();
	self.elevate_degree();
	self.elevate_degree();
	self.elevate_degree();
	self.elevate_degree();
	self.elevate_degree();
	self.elevate_degree();

	self.save_to_obj(partial_path + ".obj");


}

template <class PointType>
void ShapeModelBezier<PointType>::save(std::string path) {
	// The coordinates are written to a file
	std::ofstream shape_file;
	shape_file.open(path);
	shape_file << this -> get_degree() << "\n";

	for (int i = 0; i < this -> control_points.size(); ++i){
		auto p = this -> control_points[i].get_point_coordinates();
		shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << "\n";
	}

	for (int e = 0; e < this -> elements.size(); ++e){
		
		shape_file << "f ";

		auto points =  this -> elements[e].get_points();
		
		for (unsigned int index = 0; index < points.size(); ++index){

			if (index + 1 < points.size()){
				shape_file << points[index] + 1 << " ";
			}
			else{
				if (e + 1 < this -> elements.size()){
					shape_file << points[index] + 1 << "\n";
				}
				else{
					shape_file << points[index] + 1;
				}
			}

		}

	}

}


template <class PointType>
void ShapeModelBezier<PointType>::construct_kd_tree_shape(){


	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();


	// The KD tree is constructed by building 
	// an "enclosing" (not strictly-speaking) KD tree from the bezier shape

	// An inverse map going from vertex pointer to global indices is created
	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support


	std::vector<std::vector<int> > vertices_in_facets;
	std::vector<int> facets_super_element;

	for (unsigned int e = 0; e < this -> get_NElements(); ++e){

		const Bezier & patch = this -> get_element(e);

	// The facets are created

		for (unsigned int l = 0; l < patch . get_degree(); ++l){

			for (unsigned int t = 0; t < l + 1; ++t){

				if (t <= l){

					int v0 = patch . get_point_global_index(patch . get_degree() - l,l - t);
					int v1 = patch . get_point_global_index(patch . get_degree() - l - 1,l - t + 1);
					int v2 = patch . get_point_global_index(patch . get_degree() - l - 1,l-t);

					std::vector<int> vertices_in_facet;
					vertices_in_facet.push_back(v0);
					vertices_in_facet.push_back(v1);
					vertices_in_facet.push_back(v2);
					
					facets_super_element.push_back(e);
					vertices_in_facets.push_back(vertices_in_facet);

				}

				if (t > 0 ){

					int v0 = patch . get_point_global_index(patch . get_degree() - l,l-t);
					int v1 = patch . get_point_global_index(patch . get_degree() - l,l - t + 1 );
					int v2 = patch . get_point_global_index(patch . get_degree() - l -1,l - t + 1);

					std::vector<int> vertices_in_facet;
					vertices_in_facet.push_back(v0);
					vertices_in_facet.push_back(v1);
					vertices_in_facet.push_back(v2);
					
					facets_super_element.push_back(e);
					vertices_in_facets.push_back(vertices_in_facet);


				}

			}

		}
	}

	// We know have everything we need to create the enclosing shape
	this -> enclosing_polyhedron = std::make_shared<ShapeModelTri<ControlPoint>>(ShapeModelTri<ControlPoint>(vertices_in_facets,
		facets_super_element,this -> control_points));


	for(int e = 0; e < this -> enclosing_polyhedron -> get_NElements(); ++e){
		Facet & facet =  this -> enclosing_polyhedron -> get_element(e);
		facet.set_owning_shape(this -> enclosing_polyhedron.get());
	}



	std::vector<int> facets;
	for (int e = 0; e < this -> enclosing_polyhedron -> get_NElements(); ++e){
		facets.push_back(e);
	}


	this -> kdt_facet = std::make_shared<KDTreeShape>(KDTreeShape(this -> enclosing_polyhedron.get()));
	this -> kdt_facet -> build(facets, 0);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "\n Elapsed time during Bezier KDTree construction : " << elapsed_seconds.count() << "s\n\n";

}


template <class PointType>
unsigned int ShapeModelBezier<PointType>::get_degree() const{
	return this -> elements[0].get_degree();
}

template <class PointType>
void ShapeModelBezier<PointType>::save_to_obj(std::string path) const{

	// An inverse map going from vertex pointer to global indices is created

	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

	std::map<int , unsigned int> pointer_to_global_indices;
	std::vector<arma::vec::fixed<3> > vertices;
	std::vector<std::tuple<int,int,int > > facets;


	// The global indices of the control points are found. 
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		const Bezier & patch = this -> elements[i];

		const std::vector<int> & control_points = patch.get_points();


		for (unsigned int index = 0; index < control_points. size(); ++index){

			if (pointer_to_global_indices.find(control_points[index])== pointer_to_global_indices.end()){

				unsigned int size =  pointer_to_global_indices.size();

				pointer_to_global_indices[control_points[index]] = size;
				auto local_indices = patch.get_local_indices(index);
				double u =  double(std::get<0>(local_indices)) / patch.get_degree();
				double v =  double(std::get<1>(local_indices)) / patch.get_degree();

				arma::vec::fixed<3>  surface_point = patch.evaluate(u,v);

				vertices.push_back(surface_point);
			}

		}


	// The facets are created

		for (unsigned int l = 0; l < patch.get_degree(); ++l){

			for (unsigned int t = 0; t < l + 1; ++t){

				if (t <= l){



					int v0 = patch.get_point(patch.get_degree() - l,l - t);
					int v1 = patch.get_point(patch.get_degree() - l - 1,l - t + 1);
					int v2 = patch.get_point(patch.get_degree() - l - 1,l-t);

					facets.push_back(std::make_tuple(v0,v1,v2));
				}

				if (t > 0 ){

					int v0 = patch.get_point(patch.get_degree() - l,l-t);
					int v1 = patch.get_point(patch.get_degree() - l,l - t + 1 );
					int v2 = patch.get_point(patch.get_degree() - l -1,l - t + 1);

					facets.push_back(std::make_tuple(v0,v1,v2));
				}

			}

		}
	}

	// The coordinates are written to a file

	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int i = 0; i < vertices.size(); ++i){
		shape_file << "v " << vertices[i](0) << " " << vertices[i](1) << " " << vertices[i](2) << "\n";
	}

	for (unsigned int i = 0; i < facets.size(); ++i){
		unsigned int indices[3];
		indices[0] = pointer_to_global_indices[std::get<0>(facets[i])] + 1;
		indices[1] = pointer_to_global_indices[std::get<1>(facets[i])] + 1;
		indices[2] = pointer_to_global_indices[std::get<2>(facets[i])] + 1;


		shape_file << "f " << indices[0] << " " << indices[1] << " " << indices[2] << "\n";

	}

}



template <class PointType>
arma::vec::fixed<9> ShapeModelBezier<PointType>::get_E_vectors(const arma::mat::fixed<3,3> & inertia){

	arma::vec::fixed<9> E_vectors;
	arma::vec moments = arma::eig_sym(inertia);
	for (int i = 0; i < 3; ++i){

		arma::mat L = inertia - moments(i) * arma::eye<arma::mat>(3,3);


		arma::vec norm_vec = {
			arma::norm(arma::cross(L.row(0).t(),L.row(1).t())),
			arma::norm(arma::cross(L.row(0).t(),L.row(2).t())),
			arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))
		};

		int j = norm_vec.index_max();

		if (j == 0){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(0).t(),L.row(1).t());
		}
		else if (j == 1){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(0).t(),L.row(2).t());
		}
		else if (j == 2){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(1).t(),L.row(2).t());
		}
	}

	return E_vectors;

}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::get_principal_axes_stable(const arma::mat::fixed<3,3> & inertia){


	auto E_vectors = ShapeModelBezier<PointType>::get_E_vectors(inertia);

	arma::mat::fixed<3,3> pa;
	pa.col(0) = E_vectors.rows(0,2) / arma::norm( E_vectors.rows(0,2));
	pa.col(1) = E_vectors.rows(3,5) / arma::norm( E_vectors.rows(3,5));
	pa.col(2) = E_vectors.rows(6,8) / arma::norm( E_vectors.rows(6,8));

	return pa;

}


template <class PointType>
arma::mat::fixed<6,6> ShapeModelBezier<PointType>::increment_P_I(arma::mat::fixed<15,15> & P_CC,
	const arma::mat::fixed<6,15> & left_mat,
	const arma::mat::fixed<6,15>  & right_mat, 
	int i,int j,int k,int l,int m,
	int p, int q, int r, int s, int t) const{

	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, p);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, q);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, r);
	P_CC.submat(0,9,2,11) = this -> get_point_covariance(i, s);
	P_CC.submat(0,12,2,14) = this -> get_point_covariance(i, t);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, p);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, q);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, r);
	P_CC.submat(3,9,5,11) = this -> get_point_covariance(j, s);
	P_CC.submat(3,12,5,14) = this -> get_point_covariance(j, t);

	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, p);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, q);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, r);
	P_CC.submat(6,9,8,11) = this -> get_point_covariance(k, s);
	P_CC.submat(6,12,8,14) = this -> get_point_covariance(k, t);


	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, p);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, q);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, r);
	P_CC.submat(9,9,11,11) = this -> get_point_covariance(l, s);
	P_CC.submat(9,12,11,14) = this -> get_point_covariance(l, t);


	P_CC.submat(12,0,14,2) = this -> get_point_covariance(m, p);
	P_CC.submat(12,3,14,5) = this -> get_point_covariance(m, q);
	P_CC.submat(12,6,14,8) = this -> get_point_covariance(m, r);
	P_CC.submat(12,9,14,11) = this -> get_point_covariance(m, s);
	P_CC.submat(12,12,14,14) = this -> get_point_covariance(m,t);


	return left_mat * P_CC * right_mat.t();




}

template <class PointType>
arma::vec::fixed<6> ShapeModelBezier<PointType>::increment_P_MI(arma::mat::fixed<15,9> & P_CC,const arma::mat::fixed<6,15> & left_mat,
	const arma::vec::fixed<9>  & right_vec, 
	int i,int j,int k,int l,int m,
	int p, int q, int r) const{

	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, p);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, q);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, r);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, p);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, q);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, r);

	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, p);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, q);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, r);


	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, p);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, q);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, r);

	P_CC.submat(12,0,14,2) = this -> get_point_covariance(m, p);
	P_CC.submat(12,3,14,5) = this -> get_point_covariance(m, q);
	P_CC.submat(12,6,14,8) = this -> get_point_covariance(m, r);

	return left_mat * P_CC * right_vec;

}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::increment_cm_cov(arma::mat::fixed<12,12> & P_CC,
	const arma::mat::fixed<12,3> & left_mat,
	const arma::mat::fixed<12,3>  & right_mat, 
	int i,int j,int k,int l, 
	int m, int p, int q, int r) const{


	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, m);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, p);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, q);
	P_CC.submat(0,9,2,11) = this -> get_point_covariance(i, r);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, m);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, p);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, q);
	P_CC.submat(3,9,5,11) = this -> get_point_covariance(j, r);


	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, m);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, p);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, q);
	P_CC.submat(6,9,8,11) = this -> get_point_covariance(k, r);


	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, m);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, p);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, q);
	P_CC.submat(9,9,11,11) = this -> get_point_covariance(l, r);



	return left_mat.t() * P_CC * right_mat;
}

template <class PointType>
void ShapeModelBezier<PointType>::compute_P_Y(){
	arma::mat::fixed<4,4> mat = arma::zeros<arma::mat>(4,4);

	mat.submat(0,0,2,2) = ShapeModelBezier<PointType>::P_XX();
	mat.submat(0,3,2,3) = this -> P_MX;
	mat.submat(3,0,3,2) = this -> P_MX.t();
	mat(3,3) = std::pow(this -> volume_sd,2);


	this -> P_Y = mat;
}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::P_XX() const { 


	return ShapeModelBezier<PointType>::partial_X_partial_I() * this -> P_I * ShapeModelBezier<PointType>::partial_X_partial_I().t();

}

template <class PointType>
void ShapeModelBezier<PointType>::compute_P_MX() {


	this -> P_MX = ShapeModelBezier<PointType>::partial_X_partial_I() * this -> P_MI;

}

template <class PointType>
arma::mat::fixed<3,6> ShapeModelBezier<PointType>::partial_X_partial_I() const{

	arma::mat::fixed<3,6> mat = arma::zeros<arma::mat>(3,6);

	mat.row(0) = ShapeModelBezier<PointType>::partial_T_partial_I();
	mat.row(1) = ShapeModelBezier<PointType>::partial_U_partial_I();
	mat.row(2) = ShapeModelBezier<PointType>::partial_theta_partial_I();
	return mat;
}

template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::partial_T_partial_I() {
	arma::rowvec dTdI = {1,1,1,0,0,0};
	return dTdI;
}


template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::partial_theta_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};



	double Pi = arma::dot(I, Q * I);

	double U = std::sqrt(T * T - 3 * Pi)/3;

	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));


	arma::rowvec dthetadI = (ShapeModelBezier<PointType>::partial_theta_partial_Theta(Theta) 
		* ShapeModelBezier<PointType>::partial_Theta_partial_W(T,Pi,U,d) 
		* ShapeModelBezier<PointType>::partial_W_partial_I());

	return dthetadI;
}


template <class PointType>
arma::vec::fixed<4> ShapeModelBezier<PointType>::get_Y(const double & volume, 
	const arma::mat::fixed<3,3> & I_C
	) {


	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};



	double Pi = arma::dot(I, Q * I);

	double U = std::sqrt(T * T - 3 * Pi)/3;

	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);


	arma::vec::fixed<4> vec;
	vec(0) = T;
	vec(1) = U;
	vec(2) = theta;
	vec(3) = volume;

	return vec;
}

template <class PointType>
arma::rowvec::fixed<4> ShapeModelBezier<PointType>::partial_Theta_partial_W(const double & T,const double & Pi,const double & U,const double & d){

	arma::rowvec::fixed<4> dThetadW =  {
		(-6 * T * T + 9 * Pi) * U,
		9 * T * U,
		-27 * U,
		-3 * (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)
	};
	return 1./(54 * std::pow(U,4)) * dThetadW;


}

template <class PointType>
double ShapeModelBezier<PointType>::partial_theta_partial_Theta(const double & Theta){
	return - 1. / std::sqrt(1 - Theta * Theta);
}

template <class PointType>
arma::mat::fixed<4,6> ShapeModelBezier<PointType>::partial_W_partial_I() const{

	arma::mat::fixed<4,6> dWdI;
	dWdI.row(0) = ShapeModelBezier<PointType>::partial_T_partial_I();
	dWdI.row(1) = ShapeModelBezier<PointType>::partial_Pi_partial_I();
	dWdI.row(2) = ShapeModelBezier<PointType>::partial_d_partial_I();
	dWdI.row(3) = ShapeModelBezier<PointType>::partial_U_partial_I();

	return dWdI;

}

template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::partial_d_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::rowvec::fixed<6> dddI = {
		I_yy * I_zz - std::pow(I_yz,2),
		I_xx * I_zz - std::pow(I_xz,2),
		I_xx * I_yy - std::pow(I_xy,2),
		2 * (I_xz * I_yz - I_zz * I_xy),
		2 * (I_xy * I_yz - I_yy * I_xz),
		2 * (I_xy * I_xz - I_xx * I_yz)
	};

	return dddI;

}

template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::partial_U_partial_I() const{

	arma::rowvec::fixed<6> dUdI = ShapeModelBezier<PointType>::partial_U_partial_Z() * ShapeModelBezier<PointType>::partial_Z_partial_I() ;


	return dUdI;

}

template <class PointType>
arma::rowvec::fixed<2> ShapeModelBezier<PointType>::partial_U_partial_Z() const{

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};

	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;


	arma::rowvec::fixed<2> dUdZ = {2 * T,-3}	;
	return 1./(18 * U) * dUdZ;

}
template <class PointType>
arma::mat::fixed<2,6> ShapeModelBezier<PointType>::partial_Z_partial_I() const {

	arma::mat::fixed<2,6> dZdI;
	dZdI.row(0) = ShapeModelBezier<PointType>::partial_T_partial_I();
	dZdI.row(1) = ShapeModelBezier<PointType>::partial_Pi_partial_I();

	return dZdI;

}

template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::partial_Pi_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};

	return 2 * I.t() * Q;

}

template <class PointType>
arma::rowvec::fixed<3> ShapeModelBezier<PointType>::partial_A_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dAdY = {1./3,- 2 * std::cos(theta / 3),2./3 * U * std::sin(theta/3)};

	return dAdY;

}


template <class PointType>
arma::rowvec::fixed<3> ShapeModelBezier<PointType>::partial_B_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dBdY = {1./3,- 2 * std::cos(theta / 3 - 2 * arma::datum::pi /3),2./3 * U * std::sin(theta/3 - 2 * arma::datum::pi /3)};

	return dBdY;

}

template <class PointType>
arma::rowvec::fixed<3> ShapeModelBezier<PointType>::partial_C_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dCdY = {1./3,- 2 * std::cos(theta / 3 + 2 * arma::datum::pi /3),2./3 * U * std::sin(theta/3 + 2 * arma::datum::pi /3)};

	return dCdY;

}

template <class PointType>
void ShapeModelBezier<PointType>::compute_P_moments(){

	auto dMdY = ShapeModelBezier<PointType>::partial_M_partial_Y();

	this -> P_moments = dMdY * this -> P_Y * dMdY.t();

}

template <class PointType>
arma::mat::fixed<4,4>  ShapeModelBezier<PointType>::partial_M_partial_Y() const{


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);


	arma::mat::fixed<4,4> mat = arma::zeros<arma::mat>(4,4);

	mat.submat(0,0,0,2) = ShapeModelBezier<PointType>::partial_A_partial_Y(theta,U);
	mat.submat(1,0,1,2) = ShapeModelBezier<PointType>::partial_B_partial_Y(theta,U);
	mat.submat(2,0,2,2) = ShapeModelBezier<PointType>::partial_C_partial_Y(theta,U);
	mat(3,3) = 1;
	return mat;


}

template <class PointType>
void ShapeModelBezier<PointType>::take_and_save_slice(int axis,std::string path, const double & c,const arma::vec & deviation) const {

	std::vector<std::vector<arma::vec> > lines;
	this -> take_slice(axis,lines,c,deviation);
	this -> save_slice(axis,path,lines);

}


template <class PointType>
void ShapeModelBezier<PointType>::save_slice(int axis, 
	std::string path, const std::vector<std::vector<arma::vec> > & lines) const{


	int a_1,a_2;

	if (axis == 0){
		a_1 = 1;
		a_2 = 2;
	}
	else if (axis == 1){
		a_1 = 0;
		a_2 = 2;
	}
	else if (axis == 2 ){
		a_1 = 0;
		a_2 = 1;	
	}
	else{
		a_1 = a_2 = 0;
		throw(std::runtime_error("Specified incorrect axis: " + std::to_string(axis)));
	}



	arma::mat lines_arma;

	if (lines.size() > 0){
		lines_arma = arma::mat(lines.size(),4);

		for (int i = 0; i < lines.size() ; ++i){

			arma::rowvec rowvec = {lines[i][0](a_1),lines[i][0](a_2),lines[i][1](a_1),lines[i][1](a_2)};
			lines_arma.row(i) = rowvec;
		}

		lines_arma.save(path, arma::raw_ascii);


	}



}





template <class PointType>
void ShapeModelBezier<PointType>::take_slice(int axis,
	std::vector<std::vector<arma::vec> > & lines,
	const double & c,
	const arma::vec & deviation) const{


	arma::vec n_plane = {0,0,0};
	n_plane(axis) = 1;


	if (this -> get_degree() != 1){
		throw(std::runtime_error("Only works with bezier shapes of degree one"));
	}

	arma::mat::fixed<3,2> T = {
		{1,0},
		{0,1},
		{-1,-1}
	};

	arma::vec::fixed<3> e3 = {0,0,1};
	arma::mat::fixed<3,3> C;

	// Each surface element is "sliced"
	for (int el = 0; el < this -> elements.size(); ++el){

		const ControlPoint & P0 = this -> control_points[this -> elements[el].get_point(1,0)];
		const ControlPoint & P1 = this -> control_points[this -> elements[el].get_point(0,1)];
		const ControlPoint & P2 = this -> control_points[this -> elements[el].get_point(0,0)];

		C.col(0) = P0.get_point_coordinates();
		C.col(1) = P1.get_point_coordinates();
		C.col(2) = P2.get_point_coordinates();

		if (deviation.size() > 0){

			int global_index = P0.get_global_index();
			C.col(0) += deviation.rows(3 * global_index, 3 * global_index + 2);

			global_index = P1.get_global_index();
			C.col(1) += deviation.rows(3 * global_index, 3 * global_index + 2);

			global_index = P2.get_global_index();
			C.col(2) += deviation.rows(3 * global_index, 3 * global_index + 2);

		}

		arma::rowvec M = n_plane.t() * C * T;
		double e = c - arma::dot(n_plane,C * e3);
		arma::vec intersect;

		std::vector <arma::vec> intersects;

		// Looking for an intersect along the u = 0 edge
		if (std::abs(M(1)) > 1e-6){
			double v_intersect = e / M(1);
			if (v_intersect >= 0 && v_intersect <= 1 ){
				arma::vec Y = {0,v_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect);
			}
		}

		// Looking for an intersect along the v = 0 edge
		if (std::abs(M(0)) > 1e-6){
			double u_intersect = e / M(0);
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,0};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect);
			}
		}

		// Looking for an intersect along the w = 0 edge
		// using u as the parameter

		if (std::abs(M(0) - M(1)) > 1e-6){
			double u_intersect = (e - M(1)) / (M(0) - M(1));
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,1 - u_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect);
			}
		}
		if (intersects.size() == 2){
			lines.push_back(intersects);
		}

	}

}







template <class PointType>
void ShapeModelBezier<PointType>::compute_P_sigma(){

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	arma::mat eigvec = ShapeModelBezier<PointType>::get_principal_axes_stable(I_C);

	arma::mat mapping_mat = arma::zeros<arma::mat>(3,9);
	mapping_mat.submat(0,3,0,5) = eigvec.col(2).t();
	mapping_mat.submat(1,6,1,8) = eigvec.col(0).t();
	mapping_mat.submat(2,0,2,2) = eigvec.col(1).t();

	this -> P_sigma = 1./16 * mapping_mat * this -> P_eigenvectors * mapping_mat.t();

}

template <class PointType>
arma::mat::fixed<3,4> ShapeModelBezier<PointType>::partial_dim_partial_M() const{


	arma::vec moments = arma::eig_sym(this -> inertia);
	double A = moments(0);
	double B = moments(1);
	double C = moments(2);


	arma::mat::fixed<3,4> mat=  {
		{-1,1,1, -( B + C - A) / this -> volume}, 
		{1,-1,1, -( A + C - B) / this -> volume}, 
		{1,1,-1, -( A + B - C) / this -> volume}
	} ;

	return 5./(4 * this -> volume) * arma::diagmat(1./this -> get_dims(this -> volume,
		this -> inertia)) * mat;


}

template <class PointType>
void ShapeModelBezier<PointType>::compute_P_dims() {


	arma::mat::fixed<3,4> partial = this -> partial_dim_partial_M();

	this -> P_dims = partial * this -> P_moments * partial.t();

}



template <class PointType>
void ShapeModelBezier<PointType>::compute_P_eigenvectors() {

	arma::mat::fixed<9,9> partial_mat = arma::zeros<arma::mat>(9,9);

	arma::vec moments = arma::eig_sym(this -> inertia);


	partial_mat.submat(0,0,2,2) = ShapeModelBezier<PointType>::partial_elambda_Elambda(moments(0));
	partial_mat.submat(3,3,5,5) = ShapeModelBezier<PointType>::partial_elambda_Elambda(moments(1));
	partial_mat.submat(6,6,8,8) = ShapeModelBezier<PointType>::partial_elambda_Elambda(moments(2));



	this -> P_eigenvectors = partial_mat * this -> P_Evectors * partial_mat.t();

}

template <class PointType>
void ShapeModelBezier<PointType>::compute_P_Evectors() {

	arma::mat::fixed<9,9> P_Evectors;


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;


	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);

	for (int a = 0; a < 3; ++a ){
		for (int b = 0; b < 3; ++b){

			P_Evectors.submat(3 * a, 3 * b, 3 * a + 2, 3 * b + 2) = (
				ShapeModelBezier<PointType>::partial_E_partial_R(moments(a)) 
				* ShapeModelBezier<PointType>::P_R_lambda_R_mu(moments(a), moments(b),a,b,theta,U) 
				* ShapeModelBezier<PointType>::partial_E_partial_R(moments(b)).t()
				);
		}
	}


	this -> P_Evectors = P_Evectors;

}


template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::
P_ril_rjm(
	const double lambda, 
	const double mu,
	const int i,
	const int j,
	const int lambda_index,
	const int mu_index,
	const double theta,
	const double U) const{

	arma::mat::fixed<7,7> mat = arma::zeros<arma::mat>(7,7);

	mat.submat(0,0,5,5) = this -> P_I;
	mat.submat(6,0,6,5) = ShapeModelBezier<PointType>::P_lambda_I(lambda_index,theta,U);
	mat.submat(0,6,5,6) = ShapeModelBezier<PointType>::P_lambda_I(mu_index,theta,U).t();
	mat(6,6) = this -> P_moments(lambda_index,mu_index);



	arma::mat::fixed<3,3> P = ShapeModelBezier<PointType>::partial_r_i_partial_I_lambda(i) * mat * ShapeModelBezier<PointType>::partial_r_i_partial_I_lambda(j).t();


	return P;


}	


template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::get_P_lambda_I(const int lambda_index) const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);





	return this -> P_lambda_I(lambda_index,theta,U);




}


template <class PointType>
arma::rowvec::fixed<6> ShapeModelBezier<PointType>::P_lambda_I(const int lambda_index,
	const double theta, const double U) const{

	arma::rowvec::fixed<3> d_lambda_d_Y;
	if (lambda_index == 0){
		d_lambda_d_Y = ShapeModelBezier<PointType>::partial_A_partial_Y(theta,U);
	}
	else if (lambda_index == 1){
		d_lambda_d_Y = ShapeModelBezier<PointType>::partial_B_partial_Y(theta,U);
	}
	else if (lambda_index == 2){
		d_lambda_d_Y = ShapeModelBezier<PointType>::partial_C_partial_Y(theta,U);

	}
	else{
		throw (std::runtime_error("unsupported case"));
	}

	return d_lambda_d_Y * ShapeModelBezier<PointType>::partial_Y_partial_I() * this -> P_I;

}

template <class PointType>
arma::mat::fixed<3,6> ShapeModelBezier<PointType>::partial_Y_partial_I() const{ 

	arma::mat::fixed<3,6> mat;

	mat.row(0) = ShapeModelBezier<PointType>::partial_T_partial_I();
	mat.row(1) = ShapeModelBezier<PointType>::partial_U_partial_I();
	mat.row(2) = ShapeModelBezier<PointType>::partial_theta_partial_I();

	return mat;

}


template <class PointType>
arma::mat::fixed<3,7> ShapeModelBezier<PointType>::
partial_r_i_partial_I_lambda(const int i){

	arma::mat::fixed<3,7> mat;

	if (i == 0){

		mat = {
			{1,0,0,0,0,0,-1},
			{0,0,0,1,0,0,0},
			{0,0,0,0,1,0,0}
		};

	}
	else if (i == 1){

		mat = {
			{0,0,0,1,0,0,0},
			{0,1,0,0,0,0,-1},
			{0,0,0,0,0,1,0}
		};


	}
	else if (i == 2){
		mat = {
			{0,0,0,0,1,0,0},
			{0,0,0,0,0,1,0},
			{0,0,1,0,0,0,-1}
		};
	}
	else{
		throw (std::runtime_error("unsupported case"));
	}
	return mat;

}

template <class PointType>
arma::mat::fixed<3,3> ShapeModelBezier<PointType>::partial_elambda_Elambda(const double & lambda) const{


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	arma::mat L = I_C - lambda * arma::eye<arma::mat>(3,3);

	arma::vec Elambda;

	arma::vec norm_vec = {
		arma::norm(arma::cross(L.row(0).t(),L.row(1).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(1))),
		arma::norm(arma::cross(L.row(0).t(),L.row(2).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(2))),
		arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))/(arma::norm(L.row(1)) * arma::norm(L.row(2)))
	};

	int i = norm_vec.index_max();

	if (i == 0){
		Elambda = arma::cross(L.row(0).t(),L.row(1).t());
	}
	else if (i == 1){
		Elambda = arma::cross(L.row(0).t(),L.row(2).t());
	}
	else if (i == 2){
		Elambda = arma::cross(L.row(1).t(),L.row(2).t());
	}
	else{
		throw (std::runtime_error("case not supported"));
	}

	arma::mat::fixed<3,3> partial = 1./arma::norm(Elambda) * (arma::eye<arma::mat>(3,3) - Elambda * Elambda.t() / arma::dot(Elambda,Elambda));



	return partial;
}

template <class PointType>
arma::mat::fixed<9,9> ShapeModelBezier<PointType>::P_R_lambda_R_mu(const double lambda, 
	const double mu,
	const int lambda_index,
	const int mu_index,
	const double theta,
	const double U) const {

	arma::mat::fixed<9,9> mat = arma::zeros<arma::mat>(9,9); 

	for (int i = 0; i < 3 ; ++ i){

		for (int j = 0; j < 3 ; ++ j){

			mat.submat(3 * i, 3 * j, 3 * i + 2, 3 * j + 2) = ShapeModelBezier<PointType>::P_ril_rjm(lambda, mu,
				i,j,lambda_index,mu_index,theta,U);

		}

	}

	return mat;


}

template <class PointType>
arma::mat::fixed<9,9> ShapeModelBezier<PointType>::P_E_lambda_E_mu() const {

	throw(std::runtime_error("Implementation is incomplete\n"));
	// arma::mat::fixed<9,9> mat;

	// arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	// double T = arma::trace(I_C);
	// double d = arma::det (I_C);

	// double I_xx = I_C(0,0);
	// double I_yy = I_C(1,1);
	// double I_zz = I_C(2,2);
	// double I_xy = I_C(0,1);
	// double I_xz = I_C(0,2);
	// double I_yz = I_C(1,2);

	// arma::vec moments = arma::eig_sym(I_C);


	// arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	// arma::mat::fixed<6,6> Q = {
	// 	{0,1./2.,1./2,0,0,0},
	// 	{1./2.,0,1./2,0,0,0},
	// 	{1./2,1./2.,0,0,0,0},
	// 	{0,0,0,-1,0,0},
	// 	{0,0,0,0,-1,0},
	// 	{0,0,0,0,0,-1}
	// };

	// double Pi = arma::dot(I, Q * I);
	// double U = std::sqrt(T * T - 3 * Pi)/3;
	// double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	// double theta = std::acos(Theta);


	// for (int a = 0; a < 3 ; ++ a){

	// 	double lambda = moments(a);

	// 	for (int b = 0; b < 3 ; ++ b){

	// 		double mu = moments(b);

	// 		mat.submat(3 * a, 3 * b, 3 * a + 2, 3 * b + 2) = (
	// 			ShapeModelBezier<PointType>::partial_E_partial_R(lambda)
	// 			* P_R_lambda_R_mu(lambda, mu,a,b,theta,U) 
	// 			* ShapeModelBezier<PointType>::partial_E_partial_R(mu).t());

	// 	}
	// }


}

template <class PointType>
arma::mat::fixed<3,9> ShapeModelBezier<PointType>::partial_E_partial_R(const double lambda) const{

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;


	arma::vec moments = arma::eig_sym(I_C);

	arma::mat L = I_C - lambda * arma::eye<arma::mat>(3,3);

	arma::mat::fixed<3,9> dEdR = arma::zeros<arma::mat>(3,9);

	arma::vec norm_vec = {
		arma::norm(arma::cross(L.row(0).t(),L.row(1).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(1))),
		arma::norm(arma::cross(L.row(0).t(),L.row(2).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(2))),
		arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))/(arma::norm(L.row(1)) * arma::norm(L.row(2)))
	};

	int i = norm_vec.index_max();



	if (i == 0){
		dEdR.submat(0,0,2,2) = - RBK::tilde(L.row(1).t());
		dEdR.submat(0,3,2,5) = RBK::tilde(L.row(0).t());

	}
	else if (i == 1){
		dEdR.submat(0,0,2,2) = - RBK::tilde(L.row(2).t());
		dEdR.submat(0,6,2,8) = RBK::tilde(L.row(0).t());

	}
	else{
		dEdR.submat(0,3,2,5) = - RBK::tilde(L.row(2).t());
		dEdR.submat(0,6,2,8) = RBK::tilde(L.row(1).t());

	}

	return dEdR;

}


template <class PointType>
void ShapeModelBezier<PointType>::elevate_degree(){

	// Containers for the new elements/control points
	std::vector< std::vector<int> >  elements_to_control_points(this -> elements.size());

	std::vector<ControlPoint> new_control_points;
	int n = this -> get_degree();

	// The forward table is constructed
	auto forw_before_elevation = Bezier::forward_table(n);
	auto forw_after_elevation = Bezier::forward_table(n + 1);

	// The reverse table is constructed
	auto rev_before_elevation = Bezier::reverse_table(n);
	auto rev_after_elevation = Bezier::reverse_table(n + 1);

	for (int e = 0; e < this -> elements.size(); ++e){
		for (int i = 0; i < forw_after_elevation.size(); ++i){
			elements_to_control_points[e].push_back(-1);
		}
	}

	for (int e = 0; e < this -> elements.size(); ++e ){
		
		// These are the indices to the old control points owned by this element
		const std::vector<int> & element_control_points = this -> elements[e].get_points();
		
		for (int i = 0; i < forw_after_elevation.size(); ++i){

			

			if (elements_to_control_points[e][i] == -1){
				auto mu = forw_after_elevation[i];

			// The coordinates of the new control points are computed
				arma::vec::fixed<3> new_C = arma::zeros<arma::vec>(3);



				for (unsigned int l = 0; l < forw_before_elevation.size(); ++l){


					auto lambda = forw_before_elevation[l];

					double coef = 
					double(Bezier::combinations(std::get<0>(lambda),std::get<0>(mu))) 
					* double(Bezier::combinations(std::get<1>(lambda),std::get<1>(mu))) 
					* double(Bezier::combinations(std::get<2>(lambda),std::get<2>(mu)))
					/ double(Bezier::combinations(n,n + 1));

					new_C += coef * this -> control_points[element_control_points[l]].get_point_coordinates();
				}

				ControlPoint new_control_point(this);
				new_control_point.set_point_coordinates(new_C);
				std::set<int> ownership;
				std::tuple<int,int,int> inserted_point_tuple_original_element = forw_after_elevation[i];



				if (std::get<0>(forw_after_elevation[i]) == 0 || std::get<1>(forw_after_elevation[i]) == 0 || std::get<2>(forw_after_elevation[i]) == 0 ){
					// edge point, things get complicated

					// This is a corner point
					if (std::get<0>(forw_after_elevation[i]) == n + 1 || std::get<1>(forw_after_elevation[i]) == n + 1 || std::get<2>(forw_after_elevation[i]) == n + 1){
						

						std::tuple<int,int,int> old_indices,new_indices;
						
						
						if(	std::get<0>(forw_after_elevation[i]) == n + 1){
					// corner vertex
							old_indices = std::make_tuple (n,0,0);
							new_indices = std::make_tuple (n + 1 ,0,0);

							

						}
						else if (std::get<1>(forw_after_elevation[i]) == n + 1){
					// corner vertex
							old_indices = std::make_tuple (0,n,0);
							new_indices = std::make_tuple (0,n + 1,0);

							

						}

						else if (std::get<2>(forw_after_elevation[i]) == n + 1){
					// corner vertex
							old_indices = std::make_tuple (0,0,n);
							new_indices = std::make_tuple (0,0,n + 1);


						}

						int global_index_of_old_control_point = element_control_points[rev_before_elevation[old_indices]];

						ownership = this -> control_points[global_index_of_old_control_point].get_owning_elements();

						// For all the elements that own this control point
						for (auto element : ownership){
							// For all the control points in this element
							const auto & owning_element_control_points = this -> elements[element].get_points();
							for (int j = 0; j < owning_element_control_points.size(); ++j){
								
								// if the global index of that control point corresponds to the global index
								// of the invariant
								// we have found the new global index of that control point
								if (owning_element_control_points[j] == global_index_of_old_control_point){

									std::tuple<int,int,int> old_indices_in_element = forw_before_elevation[j];
									if (std::get<0>(old_indices_in_element) == n){
										std::get<0>(old_indices_in_element) += 1;
									}
									else if (std::get<1>(old_indices_in_element) == n){
										std::get<1>(old_indices_in_element) += 1;
									}
									else if (std::get<2>(old_indices_in_element) == n){
										std::get<2>(old_indices_in_element) += 1;
									}
									else{
										throw(std::runtime_error("impossible, we should be able to find a corner vertex"));
									}

									elements_to_control_points[element][rev_after_elevation[old_indices_in_element]] = new_control_points.size();
									break;
								}
							}
						}

					}


					// This is an edge point.
					// The idea is:
					// 1. Find the corner point that immediately precedes the point
					// 2. Find where this corner point shows in the other element that owns it
					// 3. In this other element, the corner point now succedes the point
					// 4. Finding the right index shift should be trivial
					else{

						std::tuple<int,int,int> corner_point_before_indices;
						std::tuple<int,int,int> corner_point_after_indices;

						int distance_from_corner_point_before;

						if(std::get<0>(forw_after_elevation[i]) == 0){
							// point is on v-w edge
							// the corner point that precedes it is thus
							// (0,n,0)
							corner_point_before_indices = std::make_tuple(0,n,0);
							corner_point_after_indices = std::make_tuple(0,0,n);

							distance_from_corner_point_before = std::get<2>(forw_after_elevation[i]);
						}

						else if(std::get<1>(forw_after_elevation[i]) == 0){
							// point is on w-u edge
							// the corner point that precedes it is thus
							// (0,0,n)
							corner_point_before_indices = std::make_tuple(0,0,n);
							corner_point_after_indices = std::make_tuple(n,0,0);

							distance_from_corner_point_before = std::get<0>(forw_after_elevation[i]);

						}

						else if(std::get<2>(forw_after_elevation[i]) == 0){
							// point is on u-v edge
							// the corner point that precedes it is thus
							// (n,0,0)
							corner_point_before_indices = std::make_tuple(n,0,0);
							corner_point_after_indices = std::make_tuple(0,n,0);

							distance_from_corner_point_before = std::get<1>(forw_after_elevation[i]);


						}
						else {
							throw(std::runtime_error("impossible, this point should belong on an edge"));
						}

						int corner_point_before_global_index = element_control_points[rev_before_elevation[corner_point_before_indices]];
						int corner_point_after_global_index = element_control_points[rev_before_elevation[corner_point_after_indices]];

						// The other element where things are happening is found by intersecting the owning elements of 
						// the two corner points forming this edge
						std::set<int> owners_corner_point_before = this -> control_points[corner_point_before_global_index].get_owning_elements();
						std::set<int> owners_corner_point_after = this -> control_points[corner_point_after_global_index].get_owning_elements();

						std::set<int> edge;
						set_intersection(owners_corner_point_before.begin(),owners_corner_point_before.end(),
							owners_corner_point_after.begin(),owners_corner_point_after.end(),
							std::inserter(edge,edge.begin()));

						// This edge should always have two elements
						assert(edge.size() == 2);
						assert(edge.find(e) != edge.end());
						edge.erase(e);

						// We now have the index of the element on the other side of the edge. 
						// the goal is to determine where the new element lives on the other edge
						int other_element_index = (*edge.begin());


						// We should now find the local indices of the two endpoints in the other element
						const auto & other_element_control_points = this -> elements[other_element_index].get_points();

						// To be fair, we only need one of the two corner points: the one after the inserted point
						// (which happens to be before the inserted point in the other element)

						int local_index_corner_point_before_other_element = std::distance(other_element_control_points.begin(),
							std::find(other_element_control_points.begin(),
								other_element_control_points.end(),
								corner_point_after_global_index));

						// local_index_corner_point_before_other_element now holds the local index of the control point immediately
						// before the inserted point in the other element
						// We need to find its index tuple

						auto index_tuple_before_elevation = forw_before_elevation[local_index_corner_point_before_other_element];
						std::tuple<int,int,int> inserted_point_tuple_other_element;
						
						// We now know where the inserted point is
						if (std::get<0>(index_tuple_before_elevation)>0){
							// starting from (n+1,0,0) on u--v
							inserted_point_tuple_other_element = std::make_tuple(distance_from_corner_point_before,n + 1 - distance_from_corner_point_before,0);
						}
						else if (std::get<1>(index_tuple_before_elevation)>0){
							// starting from (0,n+1,0) on v--w

							inserted_point_tuple_other_element = std::make_tuple(0,distance_from_corner_point_before,n + 1 - distance_from_corner_point_before);

						}
						else if (std::get<2>(index_tuple_before_elevation)>0){
							// starting from (0,0,n+1) on w--u

							inserted_point_tuple_other_element = std::make_tuple(n + 1 - distance_from_corner_point_before,0,distance_from_corner_point_before);

						}
						else{
							throw(std::runtime_error("impossible"));
						}

						//
						elements_to_control_points[e][rev_after_elevation[inserted_point_tuple_original_element]] = new_control_points.size();
						elements_to_control_points[other_element_index][rev_after_elevation[inserted_point_tuple_other_element]] = new_control_points.size();
						ownership.insert(e);
						ownership.insert(other_element_index);

					}

				}
				// This is a center point, no problem
				else{
					elements_to_control_points[e][rev_after_elevation[inserted_point_tuple_original_element]] = new_control_points.size();
					ownership.insert(e);

				}


				new_control_point.set_owning_elements(ownership);
				new_control_points.push_back(new_control_point);

			}

		}

	}

	this -> control_points = new_control_points;
	for (int e = 0; e < this -> elements.size(); ++e){
		this -> elements[e] = Bezier(elements_to_control_points[e],this);
		this -> elements[e].set_global_index(e);
	}

	

}


template <class PointType>
void ShapeModelBezier<PointType>::set_patch_covariances(const std::vector<std::vector<double >> & covariance_params){


	assert(covariance_params.size() == this -> elements.size());

	for (int e = 0; e < this -> elements.size(); ++e){
		this -> elements[e].set_patch_covariance(covariance_params[e]);
		
	}



}






template class ShapeModelBezier<ControlPoint>;

