#include <chrono>
#include <assert.h>
#include <ControlPoint.hpp>
#include <KDTree.hpp>
#include <ShapeModel.hpp>

template <class PointType>
ShapeModel<PointType>::ShapeModel() {

}


template <class PointType>
ShapeModel<PointType>::ShapeModel(std::string ref_frame_name,
	FrameGraph * frame_graph) {
	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
}
template <class PointType>
std::string ShapeModel<PointType>::get_ref_frame_name() const {
	return this -> ref_frame_name;
}
template <class PointType>
const arma::mat::fixed<3,3> & ShapeModel<PointType>::get_inertia() const {
	return this -> inertia;

}
template <class PointType>
double ShapeModel<PointType>::get_volume() const{
	return this -> volume;
}

template <class PointType>
arma::vec ShapeModel<PointType>::get_inertia_param() const{

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	return I;
}

template <class PointType>
double ShapeModel<PointType>::get_surface_area() const {
	return this -> surface_area;
}

template <class PointType>
const arma::vec::fixed<3> & ShapeModel<PointType>::get_center_of_mass() const{
	return this -> cm;
}

template <class PointType>
void ShapeModel<PointType>::shift_to_barycenter() {

	arma::vec x = - this -> get_center_of_mass();
	this -> translate(x);
	this -> compute_center_of_mass();
	assert(arma::norm(this -> cm) < 1e-10);

}
template <class PointType>
void ShapeModel<PointType>:: align_with_principal_axes() {


	this -> compute_inertia();

	std::cout << "Non-dimensional inertia: " << std::endl;
	std::cout << this -> inertia << std::endl;

	arma::vec::fixed<3> moments;
	arma::mat::fixed<3,3> axes;

	this -> get_principal_inertias(axes,moments);

	this -> rotate(axes.t());

	this -> inertia = arma::diagmat(moments);

}



template <class PointType>
void ShapeModel<PointType>::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}

template <class PointType>
const std::vector<PointType> & ShapeModel<PointType>::get_points() const{
	return this -> control_points;

}

template <class PointType>
PointType & ShapeModel<PointType>::get_point(unsigned int i) {
	return this -> control_points[i];
}

template <class PointType>
const arma::vec::fixed<3> & ShapeModel<PointType>::get_point_coordinates(unsigned int i) const{
	return this -> control_points[i].get_point_coordinates();
}


template <class PointType>
unsigned int ShapeModel<PointType>::get_NControlPoints() const {
	return this -> control_points . size();
}

template <class PointType>
std::shared_ptr<KDTreeShape> ShapeModel<PointType>::get_KDTreeShape() const {
	return this -> kdt_facet;
}

template <class PointType>
void ShapeModel<PointType>::construct_kd_tree_control_points(){

	std::vector<int> indices;
	for (int i =0; i < this -> get_NControlPoints(); ++i){
			indices.push_back(i);
	}

	this -> kdt_control_points = std::make_shared< KDTree<ShapeModel,PointType> >(KDTree< ShapeModel,PointType> (this));
	this -> kdt_control_points -> build(indices,0);

}


template <class PointType>
arma::vec::fixed<3> ShapeModel<PointType>::get_center() const{
	arma::vec center = {0,0,0};
	unsigned int N = this -> get_NControlPoints();

	for (int i  = 0; i < N; ++i){
		center += this -> get_point_coordinates(i) / N;
	}
	return center;
}

template <class PointType>
void ShapeModel<PointType>::translate( const arma::vec::fixed<3> & x){
	unsigned int N = this -> get_NControlPoints();
	for (int i  = 0; i < N; ++i){
		arma::vec::fixed<3> coords = this -> get_point_coordinates(i);
		coords += x;
		this -> control_points[i].set_point_coordinates(coords) ;
	}
}

template <class PointType>
void ShapeModel<PointType>::rotate(const arma::mat::fixed<3,3> & M){

	unsigned int N = this -> get_NControlPoints();

	for (int i  = 0; i < N; ++i){
		arma::vec::fixed<3> coords = M*this -> get_point_coordinates(i);
		this -> control_points[i].set_point_coordinates(coords) ;
	}
}

template <class PointType>
void ShapeModel<PointType>::get_bounding_box(double * bounding_box,arma::mat M) const {

	arma::vec bbox_min = M * this -> control_points[0]. get_point_coordinates();
	arma::vec bbox_max = M * this -> control_points[0]. get_point_coordinates();

	for ( unsigned int vertex_index = 0; vertex_index < this -> get_NControlPoints(); ++ vertex_index) {
		bbox_min = arma::min(bbox_min,M * this -> control_points[vertex_index].get_point_coordinates());
		bbox_max = arma::max(bbox_max,M * this -> control_points[vertex_index].get_point_coordinates());

	}

	bounding_box[0] = bbox_min(0);
	bounding_box[1] = bbox_min(1);
	bounding_box[2] = bbox_min(2);
	bounding_box[3] = bbox_max(0);
	bounding_box[4] = bbox_max(1);
	bounding_box[5] = bbox_max(2);

	std::cout << "- Bounding box : " << std::endl;
	std::cout << "-- xmin : " << bbox_min(0) << std::endl;
	std::cout << "-- xmax : " << bbox_max(0) << std::endl;

	std::cout << "-- ymin : " << bbox_min(1) << std::endl;
	std::cout << "-- ymax : " << bbox_max(1) << std::endl;


	std::cout << "-- zmin : " << bbox_min(2) << std::endl;
	std::cout << "-- zmax : " << bbox_max(2) << std::endl;


}

template <class PointType>
void ShapeModel<PointType>::get_principal_inertias(arma::mat & axes,arma::vec & moments) const{


	arma::eig_sym(moments,axes,this -> inertia);
	

	// Check if two axes are flipped here

	// The longest distance measured from the center of mass along the first principal
	// axis to the surface should be measured along +x

	double bbox[6];
	
	arma::vec e0 = axes.col(0);
	arma::vec e1 = axes.col(1);
	arma::vec e2 = axes.col(2);


	if (arma::det(axes) < 0){
		e0 = -e0;
	}

	axes = arma::join_rows(e0,arma::join_rows(axes.col(1),axes.col(2)));


	this -> get_bounding_box(bbox,axes.t());
	arma::vec x_max = {bbox[3],bbox[4],bbox[5]}; 
	arma::vec x_min = {bbox[0],bbox[1],bbox[2]}; 

	arma::mat M0 = arma::eye<arma::mat>(3,3);
	arma::mat M1 = {{1,0,0},{0,-1,0},{0,0,-1}};
	arma::mat M2 = {{-1,0,0},{0,1,0},{0,0,-1}};
	arma::mat M3 = {{-1,0,0},{0,-1,0},{0,0,1}};

	if (std::abs(arma::dot(x_max,e0)) > std::abs(arma::dot(x_min,e0))){

		if(std::abs(arma::dot(x_max,e1)) > std::abs(arma::dot(x_min,e1))){
			axes = axes * M0;
		}

		else{
			axes = axes * M1;
		}


	}
	else{
		if(std::abs(arma::dot(x_max,e1)) > std::abs(arma::dot(x_min,e1))){
			axes = axes * M2;
		}

		else{

			axes = axes * M3;
		}
	}
	

	std::cout << "Principal axes: " << std::endl;
	std::cout << axes << std::endl;



	std::cout << "Non-dimensional principal moments: " << std::endl;
	std::cout << moments << std::endl;


}


template <class PointType>
void ShapeModel<PointType>::add_control_point(PointType & vertex) {
	this -> control_points.push_back(vertex);
}


template <class PointType>
void ShapeModel<PointType>::assemble_covariance(arma::mat & P,
	const PointType & Ci,
	const PointType & Cj,
	const PointType & Ck,
	const PointType & Cl,
	const PointType & Cm,
	const PointType & Cp){


	P = arma::zeros<arma::mat>(9,9);

	// First row
	if (Ci.get_global_index() == Cl.get_global_index()){
		P.submat(0,0,2,2) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cm.get_global_index()){
		P.submat(0,3,2,5) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cp.get_global_index()){
		P.submat(0,6,2,8) = Ci .get_covariance();
	}

	// Second row
	if (Cj.get_global_index() == Cl.get_global_index()){
		P.submat(3,0,5,2) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cm.get_global_index()){
		P.submat(3,3,5,5) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cp.get_global_index()){
		P.submat(3,6,5,8) = Cj .get_covariance();
	}


	// Third row
	if (Ck.get_global_index() == Cl.get_global_index()){
		P.submat(6,0,8,2) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cm.get_global_index()){
		P.submat(6,3,8,5) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cp.get_global_index()){
		P.submat(6,6,8,8) = Ck .get_covariance();
	}

}


template <class PointType>
void ShapeModel<PointType>::assemble_covariance(arma::mat & P,
	const PointType & Ci,
	const PointType & Cj,
	const PointType & Ck,
	const PointType & Cl,
	const PointType & Cm,
	const PointType & Cp,
	const PointType & Cq){


	P = arma::zeros<arma::mat>(12,9);

	// First row
	if (Ci.get_global_index() == Cm.get_global_index()){
		P.submat(0,0,2,2) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cp.get_global_index()){
		P.submat(0,3,2,5) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cq.get_global_index()){
		P.submat(0,6,2,8) = Ci .get_covariance();
	}

	// Second row
	if (Cj.get_global_index() == Cm.get_global_index()){
		P.submat(3,0,5,2) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cp.get_global_index()){
		P.submat(3,3,5,5) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cq.get_global_index()){
		P.submat(3,6,5,8) = Cj .get_covariance();
	}


	// Third row
	if (Ck.get_global_index() == Cm.get_global_index()){
		P.submat(6,0,8,2) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cp.get_global_index()){
		P.submat(6,3,8,5) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cq.get_global_index()){
		P.submat(6,6,8,8) = Ck .get_covariance();
	}

	// Fourth row
	if (Cl.get_global_index() == Cm.get_global_index()){
		P.submat(9,0,11,2) = Cl .get_covariance();
	}

	if (Cl.get_global_index() == Cp.get_global_index()){
		P.submat(9,3,11,5) = Cl .get_covariance();
	}

	if (Cl.get_global_index() == Cq.get_global_index()){
		P.submat(9,6,11,8) = Cl .get_covariance();
	}

}

template <class PointType>
void ShapeModel<PointType>::assemble_covariance(arma::mat & P,
	const PointType & Ci,
	const PointType & Cj,
	const PointType & Ck,
	const PointType & Cl,
	const PointType & Cm,
	const PointType & Cp,
	const PointType & Cq,
	const PointType & Cr){


	P = arma::zeros<arma::mat>(12,12);

	// First row
	if (Ci.get_global_index() == Cm.get_global_index()){
		P.submat(0,0,2,2) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cp.get_global_index()){
		P.submat(0,3,2,5) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cq.get_global_index()){
		P.submat(0,6,2,8) = Ci .get_covariance();
	}

	if (Ci.get_global_index() == Cr.get_global_index()){
		P.submat(0,9,2,11) = Ci .get_covariance();
	}

	// Second row
	if (Cj.get_global_index() == Cm.get_global_index()){
		P.submat(3,0,5,2) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cp.get_global_index()){
		P.submat(3,3,5,5) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cq.get_global_index()){
		P.submat(3,6,5,8) = Cj .get_covariance();
	}

	if (Cj.get_global_index() == Cr.get_global_index()){
		P.submat(3,9,5,11) = Cj .get_covariance();
	}


	// Third row
	if (Ck.get_global_index() == Cm.get_global_index()){
		P.submat(6,0,8,2) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cp.get_global_index()){
		P.submat(6,3,8,5) = Ck .get_covariance();
	}

	if (Ck.get_global_index() == Cq.get_global_index()){
		P.submat(6,6,8,8) = Ck .get_covariance();
	}


	if (Ck.get_global_index() == Cr.get_global_index()){
		P.submat(6,9,8,11) = Ck .get_covariance();
	}

	// Fourth row
	if (Cl.get_global_index() == Cm.get_global_index()){
		P.submat(9,0,11,2) = Cl .get_covariance();
	}

	if (Cl.get_global_index() == Cp.get_global_index()){
		P.submat(9,3,11,5) = Cl .get_covariance();
	}

	if (Cl.get_global_index() == Cq.get_global_index()){
		P.submat(9,6,11,8) = Cl .get_covariance();
	}


	if (Cl.get_global_index() == Cr.get_global_index()){
		P.submat(9,9,11,11) = Cl .get_covariance();
	}


}

template <class PointType>
double ShapeModel<PointType>::get_circumscribing_radius() const{

	double radius  = arma::norm(this -> control_points[0].get_point_coordinates() - this -> cm );

	for ( unsigned int i = 0; i < this -> get_NControlPoints(); ++ i) {
		radius = std::max(arma::norm(this -> control_points[i].get_point_coordinates() - this -> cm),radius);
	}

	return radius;

}

template class ShapeModel<ControlPoint>;


