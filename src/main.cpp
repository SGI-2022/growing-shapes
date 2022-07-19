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

float timeStep1 = 0.05;
float max1 = 200;
float numSteps1 = 0;

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

// Options for algorithms
int iVertexSource = 7;

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

void computeExplicitHeat(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd k;
    igl::gaussian_curvature(meshV, meshF, k);

    for (int i = 0; i < numSteps; i++) {
        k = k + timeStep * L * k;
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("explicit heat", k);
    mesh->setMapRange({ -0.1,0.1 });
}

void computeImplicitHeat(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd k;
    igl::gaussian_curvature(meshV, meshF, k);

    Eigen::SparseMatrix<double> I(L.rows(), L.rows());
    I.setIdentity();

    // first method
    Eigen::SparseMatrix<double> temp = (I - timeStep * L);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(temp);
    //Eigen::MatrixXd temp_inv = temp.inverse();
    for (int i = 0; i < numSteps; i++) {
        //k = temp_inv * k;
        k = solver.solve(k);
    }

    //// second method
    //Eigen::EigenSolver<Eigen::MatrixXd> es(temp);
    //auto eigenVecs = es.eigenvectors();
    //auto eigenVals = es.eigenvalues();
    //k = 1 / ((eigenVals.col(0)[1]) ^ int(numSteps)) * k;

    auto temp2 = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp2->addVertexScalarQuantity("implicit heat", k);
    mesh->setMapRange({ -0.1,0.1 });
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  ImGui::PushItemWidth(100);

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
 
  // Explicit Heat Equation
  ImGui::Button("Explicit Heat Equation");

  //ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step###explicit", &timeStep1);
  //ImGui::PopItemWidth();

  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  ImGui::InputFloat("Max###explicit", &max1);
  //ImGui::PopItemWidth();
  
  //ImGui::SameLine();
  //ImGui::PushItemWidth(150);
  //if (ImGui::SliderFloat("Num Steps###explicit", &numSteps1, 0, max1)) {
  //    computeExplicitHeat(numSteps1, timeStep1);
  //}
  //ImGui::PopItemWidth();

  //// Semi-implicit Heat Equation
  //ImGui::Button("Semi-implicit Heat Equation");

  //static float timeStep2 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step###implicit", &timeStep2);
  //ImGui::PopItemWidth();

  //static float max2 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max###implicit", &max2);
  //ImGui::PopItemWidth();

  //static float numSteps2 = 0;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(150);
  //if (ImGui::SliderFloat("Num Steps###implicit", &numSteps2, 0, max2)) {
  //    computeImplicitHeat(numSteps2, timeStep2);
  //}
  //ImGui::PopItemWidth();
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

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
