#include "Element.hpp"
#include "ControlPoint.hpp"
#include "ShapeModel.hpp"


Element::Element(std::vector<int> & control_points,ShapeModel<ControlPoint> * owning_shape) {
	this -> control_points = control_points;
	this -> owning_shape = owning_shape;
}


const std::vector<int> & Element::get_points() const {
	return this -> control_points;
}


double Element::get_area() const {
	return this -> area;
}

const arma::vec::fixed<3> & Element::get_point_coordinates(int point_index) const{

	return this -> owning_shape -> get_point_coordinates(point_index);

}

arma::vec::fixed<3> Element::get_center()  const{
	return this -> center;

}

void Element::set_owning_shape(ShapeModel<ControlPoint> * owning_shape){
	this -> owning_shape = owning_shape;
}


arma::vec::fixed<3>  Element::get_normal_coordinates() const  {
	return  this -> normal;
}

void Element::update() {

	this -> compute_normal();
	this -> compute_area();
	this -> compute_center();

}

void Element::set_control_points(std::vector<int > & control_points){
	this -> control_points = control_points;
}


int Element::get_super_element() const{
	return this -> super_element;
}

void Element::set_super_element(int super_element){
	this -> super_element = super_element;
}

void Element::set_global_index(int i ){
	this -> global_index = i;
}

int Element::get_global_index() const{
	return this -> global_index;
}

int Element::get_NControlPoints_in_owning_shape() const{
	return this -> owning_shape -> get_NControlPoints();
}
