#include <memory>
#include <ShapeModel.hpp>
#include <Facet.hpp>

Facet::Facet(std::vector<int> & vertices,ShapeModel<ControlPoint> * owning_shape) : Element(vertices,owning_shape){

	if (control_points.size()!= 3){
		throw(std::runtime_error("This facet was provided with " + std::to_string(vertices.size()) + " vertices"));
	}

	this -> update();
}


std::set < int  > Facet::get_neighbors(bool all_neighbors) const {

	std::set<int > neighbors;

	const ControlPoint & V0 = this -> owning_shape -> get_point(this -> control_points[0]);
	const ControlPoint & V1 = this -> owning_shape -> get_point(this -> control_points[1]);
	const ControlPoint & V2 = this -> owning_shape -> get_point(this -> control_points[2]);

	int v0_index = this -> control_points[0];
	int v1_index = this -> control_points[1];
	int v2_index = this -> control_points[2];


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
		std::set<int > neighboring_facets_e0 = V0.common_elements(v1_index);
		std::set<int > neighboring_facets_e1 = V1.common_elements(v2_index);
		std::set<int > neighboring_facets_e2 = V2.common_elements(v0_index);

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


void Facet::compute_normal() {

	const arma::vec::fixed<3> & P0 = this -> owning_shape -> get_point_coordinates(this -> control_points[0]);
	const arma::vec::fixed<3> & P1 = this -> owning_shape -> get_point_coordinates(this -> control_points[1]);
	const arma::vec::fixed<3> & P2 = this -> owning_shape -> get_point_coordinates(this -> control_points[2]);
	this -> normal = arma::normalise(arma::cross(P1 - P0, P2 - P0));
}


int Facet::vertex_not_on_edge(int v0,int v1) const {

	for (unsigned int i = 0; i < 3; ++i) {

		int global_index = this -> control_points[i];

		if (global_index != v0 && global_index != v1 ) {
			return global_index;
		}
	}
	return -1;

}

const arma::vec::fixed<3> & Facet::get_normal_coordinates() const{
	return this -> normal;
}

void Facet::compute_center() {

	arma::vec::fixed<3> center = arma::zeros(3);
	for (unsigned int vertex_index = 0; vertex_index < this -> control_points . size(); ++vertex_index) {
		center += this -> owning_shape -> get_point_coordinates(this -> control_points[vertex_index]);
	}
	this -> center = center / this -> control_points . size();

}

void Facet::compute_area() {
	const arma::vec::fixed<3> & P0 = this -> owning_shape -> get_point_coordinates(this -> control_points[0]) ;
	const arma::vec::fixed<3> & P1 = this -> owning_shape -> get_point_coordinates(this -> control_points[1]) ;
	const arma::vec::fixed<3> & P2 = this -> owning_shape -> get_point_coordinates(this -> control_points[2]) ;
	this -> area = arma::norm( arma::cross(P1 - P0, P2 - P0)) / 2;
}
