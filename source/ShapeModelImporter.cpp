#include <ShapeModelImporter.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>
#include <Facet.hpp>
#include <Bezier.hpp>



void ShapeModelImporter::load_bezier_shape_model(std::string filename, 
	double scaling_factor, bool as_is,
	ShapeModelBezier<ControlPoint> & shape_model) {

	std::ifstream ifs( filename);

	if (!ifs.is_open()) {
		throw(std::runtime_error("There was a problem opening " + filename) );
	}

	std::string line;
	std::vector<arma::vec> control_point_coords;
	std::vector<std::vector<int> > shape_patch_indices;

	std::cout << " Reading " <<  filename << std::endl;
	int degree = -1;
	int N_c;
	while (std::getline(ifs, line)) {

		std::stringstream linestream(line);

		if (degree < 0){
			linestream >> degree;
			continue;
		}

		N_c = (degree + 1)* (degree + 2) / 2;


		char type;
		linestream >> type;

		if (type == '#' || type == 's'  || type == 'o' || type == 'm' || type == 'u' || line.size() == 0) {
			continue;
		}

		else if (type == 'v') {
			double vx, vy, vz;
			linestream >> vx >> vy >> vz;
			arma::vec vertex = {vx, vy, vz};
			control_point_coords.push_back( scaling_factor * vertex);

		}

		else if (type == 'f') {
			std::vector<int> patch_indices;

			for (int i = 0; i < N_c; ++ i){
				int v;
				linestream >> v;
				patch_indices.push_back(v - 1);
			}

			shape_patch_indices.push_back(patch_indices);

		}

		else {
			throw(std::runtime_error(" unrecognized type: "  + std::to_string(type)));
		}

	}

	std::cout << " Shape degree: " << degree << std::endl;
	std::cout << " Number of points per patch: " << N_c << std::endl;

	std::cout << " Number of control points: " << control_point_coords.size() << std::endl;
	std::cout << " Number of patches: " << shape_patch_indices.size() << std::endl;


	// Vertices are added to the shape model
	std::cout << std::endl << " Constructing control points " << std::endl  ;
	boost::progress_display progress_vertices(control_point_coords.size()) ;

	for (unsigned int vertex_index = 0; vertex_index < control_point_coords.size(); ++vertex_index) {

		ControlPoint vertex = ControlPoint(&shape_model);
		vertex.set_point_coordinates(control_point_coords[vertex_index]);
		vertex.set_global_index(vertex_index);
		shape_model. add_control_point(vertex);
		++progress_vertices;

	}

	std::cout << std::endl << " Constructing patches " << std::endl ;


	std::vector<Bezier> elements;
	
	boost::progress_display progress_facets(shape_patch_indices.size()) ;

	// Patches are added to the shape model
	for (unsigned int patch_index = 0; patch_index < shape_patch_indices.size(); ++patch_index) {

		std::vector<int> vertices;
		
		// The vertices stored in this patch are pulled.
		for ( int i = 0; i < N_c; ++ i){
			vertices.push_back(shape_patch_indices[patch_index][i]);
		}


		for ( int i = 0; i < N_c; ++ i){
			shape_model.get_point(vertices[i]).add_ownership(patch_index);
		}

		Bezier patch = Bezier(vertices,&shape_model);
		patch.set_global_index(patch_index);
		
		elements.push_back(patch);

		++progress_facets;
	}

	shape_model.set_elements(elements);

	std::cout << "done loading patches\n";
	
	shape_model.populate_mass_properties_coefs_deterministics();
	shape_model.update_mass_properties();

	if (! as_is) {

		// The shape model is shifted so as to have its coordinates
		// expressed in its barycentric frame
		shape_model. shift_to_barycenter();

		shape_model. update_mass_properties();

		// The shape model is then rotated so as to be oriented
		// with respect to its principal axes
		shape_model. align_with_principal_axes();

	}
	
}



void ShapeModelImporter::load_obj_shape_model(std::string filename, double scaling_factor, bool as_is,
	ShapeModelTri<ControlPoint> & shape_model) {

	std::ifstream ifs( filename);

	if (!ifs.is_open()) {
		throw(std::runtime_error("There was a problem opening " + filename) );
	}

	std::string line;
	std::vector<arma::vec> vertices;
	std::vector<arma::uvec> facet_vertices;

	std::cout << " Reading " <<  filename << std::endl;

	while (std::getline(ifs, line)) {

		std::stringstream linestream(line);

		std::string word1, word2, word3;

		char type;
		linestream >> type;

		if (type == '#' || type == 's'  || type == 'o' || type == 'm' || type == 'u' || line.size() == 0) {
			continue;
		}

		else if (type == 'v') {
			double vx, vy, vz;
			linestream >> vx >> vy >> vz;
			arma::vec vertex = {vx, vy, vz};
			vertices.push_back( scaling_factor * vertex);

		}

		else if (type == 'f') {
			unsigned int v0, v1, v2;
			linestream >> v0 >> v1 >> v2;

			arma::uvec vertices_in_facet = {v0 - 1, v1 - 1, v2 - 1};
			facet_vertices.push_back(vertices_in_facet);

		}

		else {
			std::cout << " unrecognized type: " << type << std::endl;
			throw;
		}

	}

	std::cout << " Number of vertices: " << vertices.size() << std::endl;
	std::cout << " Number of facets: " << facet_vertices.size() << std::endl;


	// Vertices are added to the shape model

	std::cout << std::endl << " Constructing Vertices " << std::endl  ;
	boost::progress_display progress_vertices(vertices.size()) ;

	for (unsigned int vertex_index = 0; vertex_index < vertices.size(); ++vertex_index) {


		ControlPoint vertex(&shape_model);
		vertex.set_point_coordinates(vertices[vertex_index]);
		vertex.set_global_index(vertex_index);

		shape_model. add_control_point(vertex);
		++progress_vertices;

	}

	std::cout << std::endl << " Constructing Facets " << std::endl ;

	boost::progress_display progress_facets(facet_vertices.size()) ;
	std::vector<Facet> elements;
	
	// Facets are added to the shape model
	for (unsigned int facet_index = 0; facet_index < facet_vertices.size(); ++facet_index) {

		// The vertices stored in this facet are pulled.
		int v0 = facet_vertices[facet_index][0];
		int v1 = facet_vertices[facet_index][1];
		int v2 = facet_vertices[facet_index][2];

		std::vector<int> vertices;
		vertices.push_back(v0);
		vertices.push_back(v1);
		vertices.push_back(v2);


		shape_model.get_point(v0).add_ownership(facet_index);
		shape_model.get_point(v1).add_ownership(facet_index);
		shape_model.get_point(v2).add_ownership(facet_index);

		Facet facet(vertices,&shape_model);
		facet.set_global_index(facet_index);
		elements. push_back(facet);
		++progress_facets;
	}

	shape_model.set_elements(elements);

	// The surface area, volume, center of mass of the shape model
	// are computed

	shape_model. update_mass_properties();

	if ( as_is == false) {

		// The shape model is shifted so as to have its coordinates
		// expressed in its barycentric frame
		shape_model. shift_to_barycenter();

		// The shape model is then rotated so as to be oriented
		// with respect to its principal axes
		// The inertia tensor is computed on this occasion
		shape_model. align_with_principal_axes();

	}

	// Facets are updated (their normals and centers
	// are computed) to reflect the new position/orientation
	shape_model. update_facets();

	// The consistency of the surface normals is checked
	shape_model. check_normals_consistency();

}
