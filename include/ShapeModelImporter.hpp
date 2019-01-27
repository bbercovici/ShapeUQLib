
#ifndef HEADER_SHAPEMODELIMPORTER
#define HEADER_SHAPEMODELIMPORTER


#include <armadillo>
#include "omp.h"
#include <boost/progress.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>
#include <ControlPoint.hpp>



class ShapeModelImporter {

public:

	static void load_obj_shape_model(std::string filename, 
		double scaling_factor, bool as_is,
		ShapeModelTri<ControlPoint> & shape_model);

	/**
	Reads-in an .b file storing the bezier shape model info and sets the field of
	$shape_model to the corresponding values
	@param shape_model Pointer to the shape model to receive the read data
	*/
	static void load_bezier_shape_model(std::string filename, 
		double scaling_factor, bool as_is,
		ShapeModelBezier<ControlPoint> & shape_model);



protected:
	

};

#endif