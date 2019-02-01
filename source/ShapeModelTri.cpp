#include "ShapeModelTri.hpp"
#include "Facet.hpp"

#pragma omp declare reduction (+ : arma::vec::fixed<3> : omp_out += omp_in) \
initializer( omp_priv = arma::zeros<arma::vec>(3) )


template <class PointType>
void ShapeModelTri<PointType>::update_mass_properties() {



	////std::cout << "Computing mass properties...\n";
	////std::cout << "\t Computing surface_area...\n";

	this -> compute_surface_area();

	////std::cout << "\t Computing volume...\n";

	this -> compute_volume();
	
	////std::cout << "\t Computing center of mass...\n";
	
	this -> compute_center_of_mass();
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	
	////std::cout << "\t Computing inertia...\n";

	this -> compute_inertia();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	////std::cout << "elapsed time in ShapeModelTri<PointType>::update_mass_properties: " << elapsed_seconds.count() << " s"<< std::endl;
	

}




template <class PointType>
ShapeModelTri<PointType>::ShapeModelTri(const std::vector<std::vector<int>> & vertices_in_facets,
	const std::vector<int> & super_elements,
	const std::vector<PointType> & control_points) {

	// Vertices are added to the shape model

	for (unsigned int vertex_index = 0; vertex_index < control_points.size(); ++vertex_index) {

		ControlPoint vertex(this);
		vertex.set_point_coordinates(control_points[vertex_index].get_point_coordinates());
		vertex.set_global_index(vertex_index);

		this -> add_control_point(vertex);

	}
	////std::cout << control_points.size() << " control points.\n";

	std::vector<Facet> elements;
	
	// Facets are added to the shape model
	for (unsigned int facet_index = 0; facet_index < vertices_in_facets.size(); ++facet_index) {

		// The vertices stored in this facet are pulled.
		int v0 = vertices_in_facets[facet_index][0];
		int v1 = vertices_in_facets[facet_index][1];
		int v2 = vertices_in_facets[facet_index][2];

		std::vector<int> vertices;
		vertices.push_back(v0);
		vertices.push_back(v1);
		vertices.push_back(v2);

		this -> get_point(v0).add_ownership(facet_index);
		this -> get_point(v1).add_ownership(facet_index);
		this -> get_point(v2).add_ownership(facet_index);

		Facet facet(vertices,this);
		facet.set_global_index(facet_index);
		facet.set_super_element(super_elements[facet_index]);
		elements. push_back(facet);
	}

	////std::cout << elements.size() << " facets.\n";

	this -> set_elements(elements);

	this -> update_facets();
	this -> check_normals_consistency();

}



















template <class PointType>
void ShapeModelTri<PointType>::update_facets() {

	for (auto & facet : this -> elements) {
		facet.update();
	}

}

template <class PointType>
void ShapeModelTri<PointType>::update_facets(std::set<Facet *> & elements) {

	for (auto & facet : elements) {
		facet -> update();
	}

}

template <class PointType>
void ShapeModelTri<PointType>::set_elements(std::vector<Facet> elements){
	this -> elements = elements;
}


template <class PointType>
void ShapeModelTri<PointType>::clear(){
	this -> control_points.clear();
	this -> elements.clear();

}




template <class PointType>
const std::vector<int> & ShapeModelTri<PointType>::get_element_control_points(int e) const{
	return this -> elements[e].get_points();
}



template <class PointType>
unsigned int ShapeModelTri<PointType>::get_NElements() const {
	return this -> elements . size();
}



template <class PointType>
arma::vec::fixed<3> ShapeModelTri<PointType>::get_point_normal_coordinates(unsigned int i) const{

	auto owning_elements = this -> control_points[i].get_owning_elements();
	arma::vec::fixed<3> n = {0,0,0};
	for (auto e : owning_elements){

		n += this -> elements[e].get_normal_coordinates();

	}
	return arma::normalise(n);

}

template <class PointType>
Facet & ShapeModelTri<PointType>::get_element(int e){
	return this -> elements[e];
}





template <class PointType>
void ShapeModelTri<PointType>::save(std::string path,const arma::vec & X,const arma::mat & M) const {
	std::ofstream shape_file;
	shape_file.open(path);

	
	for (unsigned int vertex_index = 0;vertex_index < this -> get_NControlPoints();++vertex_index) {

		arma::vec::fixed<3> coords = this -> control_points[vertex_index].get_point_coordinates();

		coords = M * coords + X;

		shape_file << "v " << coords(0)  << " " << coords(1) << " " << coords(2) << std::endl;
	}

	for (unsigned int facet_index = 0;facet_index < this -> get_NElements();++facet_index) {

		unsigned int v0 =  this -> elements[facet_index].get_points().at(0) + 1;
		unsigned int v1 =  this -> elements[facet_index].get_points().at(1) + 1;
		unsigned int v2 =  this -> elements[facet_index].get_points().at(2) + 1;

		shape_file << "f " << v0 << " " << v1 << " " << v2 << std::endl;

	}


	shape_file.close();




}









template <class PointType>
void ShapeModelTri<PointType>::check_normals_consistency(double tol) const {
	double facet_area_average = 0;

	arma::vec surface_sum= arma::zeros<arma::vec>(3);

	for (unsigned int facet_index = 0; facet_index < this -> elements.size(); ++facet_index) {

		const Facet & facet = this -> elements[facet_index];

		surface_sum += facet.get_area() * facet.get_normal_coordinates();
		facet_area_average += facet.get_area();

	}


	facet_area_average = facet_area_average / this -> elements.size();
	if (arma::norm(surface_sum) / facet_area_average > tol) {
		////std::cout <<  "Warning : normals were incorrectly oriented. norm(sum(n * s))/sum(s)= " + std::to_string(arma::norm(surface_sum) / facet_area_average) << std::endl;
	}

}


template <class PointType>
void ShapeModelTri<PointType>::compute_volume() {
	double volume = 0;

	#pragma omp parallel for reduction(+:volume) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;facet_index < this -> elements.size();++facet_index) {

		const std::vector<int> & vertices = this -> elements[facet_index].get_points();

		const arma::vec & r0 =  this -> control_points[vertices[0]]. get_point_coordinates();
		const arma::vec & r1 =  this -> control_points[vertices[1]]. get_point_coordinates();
		const arma::vec & r2 =  this -> control_points[vertices[2]]. get_point_coordinates();
		double dv = arma::dot(r0, arma::cross(r1 - r0, r2 - r0)) / 6.;
		volume += dv;

	}

	this -> volume = volume;

}



template <class PointType>
void ShapeModelTri<PointType>::compute_center_of_mass() {
	
	double volume = this -> get_volume();
	double cx = 0;
	double cy = 0;
	double cz = 0;

	#pragma omp parallel for reduction (+:cx,cy,cz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;facet_index < this -> elements.size();++facet_index) {

		const std::vector<int> & vertices = this -> elements[facet_index].get_points();
		

		const arma::vec::fixed<3> & r0 =  this -> control_points[vertices[0]].get_point_coordinates();
		const arma::vec::fixed<3> & r1 =  this -> control_points[vertices[1]].get_point_coordinates();
		const arma::vec::fixed<3> & r2 =  this -> control_points[vertices[2]].get_point_coordinates();

		
		double dv = 1. / 6. * arma::dot(r1, arma::cross(r1 - r0, r2 - r0));

		// C += (r0 + r1 + r2) / 4 * dv / volume;

		double coef =  dv / (4 * volume);
		cx += (r0(0) + r1(0) + r2(0)) * coef;
		cy += (r0(1) + r1(1) + r2(1)) * coef;
		cz += (r0(2) + r1(2) + r2(2)) * coef;

	}


	this -> cm =  {cx,cy,cz};


}

template <class PointType>
void ShapeModelTri<PointType>::compute_inertia() {


	double P_xx = 0;
	double P_yy = 0;
	double P_zz = 0;
	double P_xy = 0;
	double P_xz = 0;
	double P_yz = 0;



	# pragma omp parallel for reduction(+:P_xx,P_yy,P_zz,P_xy,P_xz,P_yz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;facet_index < this -> elements.size();++facet_index) {

		const std::vector<int> & vertices = this -> elements[facet_index].get_points();

		// Normalized coordinates
		const arma::vec::fixed<3> & r0 =  this -> control_points[vertices[0]].get_point_coordinates();
		const arma::vec::fixed<3> & r1 =  this -> control_points[vertices[1]].get_point_coordinates();
		const arma::vec::fixed<3> & r2 =  this -> control_points[vertices[2]].get_point_coordinates();

		double const * r0d =  r0. colptr(0);
		double const * r1d =  r1. colptr(0);
		double const * r2d =  r2. colptr(0);

		double dv = 1. / 6. * arma::dot(r1, arma::cross(r1 - r0, r2 - r0));



		P_xx += dv / 20 * (2 * r0d[0] * r0d[0]
			+ 2 * r1d[0] * r1d[0]
			+ 2 * r2d[0] * r2d[0]
			+ r0d[0] * r1d[0]
			+ r0d[0] * r1d[0]
			+ r0d[0] * r2d[0]
			+ r0d[0] * r2d[0]
			+ r1d[0] * r2d[0]
			+ r1d[0] * r2d[0]);


		P_yy += dv / 20 * (2 * r0d[1] * r0d[1]
			+ 2 * r1d[1] * r1d[1]
			+ 2 * r2d[1] * r2d[1]
			+ r0d[1] * r1d[1]
			+ r0d[1] * r1d[1]
			+ r0d[1] * r2d[1]
			+ r0d[1] * r2d[1]
			+ r1d[1] * r2d[1]
			+ r1d[1] * r2d[1]);

		P_zz += dv / 20 * (2 * r0d[2] * r0d[2]
			+ 2 * r1d[2] * r1d[2]
			+ 2 * r2d[2] * r2d[2]
			+ r0d[2] * r1d[2]
			+ r0d[2] * r1d[2]
			+ r0d[2] * r2d[2]
			+ r0d[2] * r2d[2]
			+ r1d[2] * r2d[2]
			+ r1d[2] * r2d[2]);

		P_xy += dv / 20 * (2 * r0d[0] * r0d[1]
			+ 2 * r1d[0] * r1d[1]
			+ 2 * r2d[0] * r2d[1]
			+ r0d[0] * r1d[1]
			+ r0d[1] * r1d[0]
			+ r0d[0] * r2d[1]
			+ r0d[1] * r2d[0]
			+ r1d[0] * r2d[1]
			+ r1d[1] * r2d[0]);

		P_xz += dv / 20 * (2 * r0d[0] * r0d[2]
			+ 2 * r1d[0] * r1d[2]
			+ 2 * r2d[0] * r2d[2]
			+ r0d[0] * r1d[2]
			+ r0d[2] * r1d[0]
			+ r0d[0] * r2d[2]
			+ r0d[2] * r2d[0]
			+ r1d[0] * r2d[2]
			+ r1d[2] * r2d[0]);

		P_yz += dv / 20 * (2 * r0d[1] * r0d[2]
			+ 2 * r1d[1] * r1d[2]
			+ 2 * r2d[1] * r2d[2]
			+ r0d[1] * r1d[2]
			+ r0d[2] * r1d[1]
			+ r0d[1] * r2d[2]
			+ r0d[2] * r2d[1]
			+ r1d[1] * r2d[2]
			+ r1d[2] * r2d[1]);

	}

	// The inertia tensor is finally assembled

	arma::mat I = {
		{P_yy + P_zz, -P_xy, -P_xz},
		{ -P_xy, P_xx + P_zz, -P_yz},
		{ -P_xz, -P_yz, P_xx + P_yy}
	};

	this -> inertia = I;
}



template <class PointType>
void ShapeModelTri<PointType>::compute_surface_area() {
	double surface_area = 0;

	#pragma omp parallel for reduction(+:surface_area) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0; facet_index < this -> elements.size(); ++facet_index) {

		surface_area += this -> elements[facet_index].get_area();
	}

	this -> surface_area = surface_area;

}








template class ShapeModelTri<ControlPoint>;












