#include "polyscope/polyscope.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <unsupported/Eigen/MatrixFunctions>
#include<Eigen/SparseCholesky>

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

float t=0.01;
int iter = 0;

// Options for algorithms
int iVertexSource = 7;

void addLaplacianExplicit() {
    using namespace Eigen;

    SparseMatrix<double> L;
    igl::cotmatrix(meshV,meshF,L);

    VectorXd K;
    igl::gaussian_curvature(meshV, meshF, K);

    for (int i=0;i<iter;i++){
    K = (K+t*L * K).eval();
    }

    polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("laplacian explicit", K);
    //scalarQ->setMapRange({-1.,1.});
}

void addLaplacianImplicit(){
    using namespace Eigen;

    SparseMatrix<double> L;
    igl::cotmatrix(meshV,meshF,L);

    VectorXd K;
    igl::gaussian_curvature(meshV, meshF, K);

    int size = meshV.rows();
    SparseMatrix<double> Y(size,size);
    for(int i=0;i<size;i++){
       Y.coeffRef(i,i)=1;
    }

    Y = (Y - t*L).eval();

    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(Y);
    for (int i=0;i<iter;i++){
    K= solver.solve(K);
    }

    polyscope::getSurfaceMesh("input mesh")
        ->addVertexScalarQuantity("laplacian implicit", K);

    //auto pmesh = polyscope::getSurfaceMesh("input mesh");
    //auto scalarQ = pmesh->addVertexScalarQuantity("laplacian explicit", K);
    //scalarQ->setMapRange({-1.,1.});
}

void addCurvatureScalar() {
  using namespace Eigen;
  using namespace std;

  VectorXd K;
  igl::gaussian_curvature(meshV, meshF, K);
  SparseMatrix<double> M, Minv;
  igl::massmatrix(meshV, meshF, igl::MASSMATRIX_TYPE_DEFAULT, M);
  igl::invert_diag(M, Minv);
  K = (Minv * K).eval();

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexScalarQuantity("gaussian curvature", K,
                                polyscope::DataType::SYMMETRIC);
}

void computeDistanceFrom() {
  Eigen::VectorXi VS, FS, VT, FT;
  // The selected vertex is the source
  VS.resize(1);
  VS << iVertexSource;
  // All vertices are the targets
  VT.setLinSpaced(meshV.rows(), 0, meshV.rows() - 1);
  Eigen::VectorXd d;
  igl::exact_geodesic(meshV, meshF, VS, FS, VT, FT, d);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexDistanceQuantity(
          "distance from vertex " + std::to_string(iVertexSource), d);
}

void computeParameterization() {
  using namespace Eigen;
  using namespace std;

  // Fix two points on the boundary
  VectorXi bnd, b(2, 1);
  igl::boundary_loop(meshF, bnd);

  if (bnd.size() == 0) {
    polyscope::warning("mesh has no boundary, cannot parameterize");
    return;
  }

  b(0) = bnd(0);
  b(1) = bnd(round(bnd.size() / 2));
  MatrixXd bc(2, 2);
  bc << 0, 0, 1, 0;

  // LSCM parametrization
  Eigen::MatrixXd V_uv;
  igl::lscm(meshV, meshF, b, bc, V_uv);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexParameterizationQuantity("LSCM parameterization", V_uv);
}

void computeNormals() {
  Eigen::MatrixXd N_vertices;
  igl::per_vertex_normals(meshV, meshF, N_vertices);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexVectorQuantity("libIGL vertex normals", N_vertices);
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  ImGui::PushItemWidth(100);

  ImGui::InputFloat("t", &t);
  ImGui::SameLine();
  ImGui::SliderInt("Offset", &iter, 0, 10000);
  addLaplacianExplicit();
  addLaplacianImplicit();

  // Curvature
  if (ImGui::Button("add curvature")) {
    addCurvatureScalar();
  }
  
  // Normals 
  if (ImGui::Button("add normals")) {
    computeNormals();
  }

  // Param
  if (ImGui::Button("add parameterization")) {
    computeParameterization();
  }

  // Geodesics
  if (ImGui::Button("compute distance")) {
    computeDistanceFrom();
  }
  ImGui::SameLine();
  ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

int main(int argc, char **argv) {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string filename = "../spot.obj";
  std::cout << "loading: " << filename << std::endl;

  // Read the mesh
  igl::readOBJ(filename, meshV, meshF);
  //t = 0.01;

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
