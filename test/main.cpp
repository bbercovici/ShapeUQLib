#include <ShapeUQLib/ShapeUQLib.hpp>

int main(){
	ShapeModelTri<ControlPoint> shape_tri;


	ShapeModelBezier<ControlPoint> shape(shape_tri,"",nullptr);



	
	return 0;
}