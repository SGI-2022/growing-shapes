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
using namespace std;


// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

Eigen::VectorXd noise_a;
Eigen::VectorXd noise_b;

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
        k = (k + timeStep * L * k).eval();
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
    Eigen::SparseMatrix<double> temp = (I - timeStep * L).eval();
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

void computeExplicitReactionDiffusionTuring(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L); 

    //Eigen::VectorXd a;
    //igl::gaussian_curvature(meshV, meshF, a);
    //Eigen::VectorXd b;
    //igl::gaussian_curvature(meshV, meshF, b);

    Eigen::VectorXd a = Eigen::VectorXd::Constant(L.rows(), 1, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Constant(L.rows(), 1, 4);

    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(L.rows(), 1, 12);  // decay rate of a
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(L.rows(), 1, 16); // growing rate of b
    float da = (float)1/16; // diffusion rate
    float db = (float)1/4; // diffusion rate
    float s = (float)1/128; // reaction rate

    alpha = alpha + noise_a;
    beta = beta + noise_b;
 
    // Turing
    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd ab = a.array() * b.array();
        a = (a + s * timeStep * (ab - a - alpha) + da * timeStep * L * a).eval();
        ab = a.array() * b.array();
        b = (b + s * timeStep * (beta - ab) + db * timeStep * L * b).eval();
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Turing Explicit", a);
    //mesh->setMapRange({ -0.1,0.1 });
}

void computeExplicitReactionDiffusionScott(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    //Eigen::VectorXd a;
    //igl::gaussian_curvature(meshV, meshF, a);
    //Eigen::VectorXd b;
    //igl::gaussian_curvature(meshV, meshF, b);

    Eigen::VectorXd a = Eigen::VectorXd::Constant(L.rows(), 1, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Constant(L.rows(), 1, 4);

    Eigen::VectorXd F = Eigen::VectorXd::Constant(L.rows(), 1, 12);  // feed rate
    Eigen::VectorXd k = Eigen::VectorXd::Constant(L.rows(), 1, 16); // degrading rate

    float da = (float)1 / 16; // diffusion rate
    float db = (float)1 / 4; // diffusion rate
    float s = (float)1 / 128; // reaction rate

    F = F + noise_a;
    k = k + noise_b;

    // Turing
    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd aab = a.array() * a.array() * b.array();
        a = (a + s * timeStep * (aab - a * (F + k)) + da * timeStep * L * a).eval();
        aab = a.array() * a.array() * b.array();
        b = (b + s * timeStep * (-1 * aab + F - F * b) + db * timeStep * L * b).eval();
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Gray & Scott Explicit", a);
    //mesh->setMapRange({ -0.1,0.1 });
}

void computeImplicitReactionDiffusion(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd a = Eigen::VectorXd::Constant(L.rows(), 1, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Constant(L.rows(), 1, 4);

    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(L.rows(), 1, 12);  // decay rate of a
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(L.rows(), 1, 16); // growing rate of b
    float da = (float)1 / 16; // diffusion rate
    float db = (float)1 / 4; // diffusion rate
    float s = (float)1 / 128; // reaction rate

    alpha = alpha + noise_a;
    beta = beta + noise_b;

    // Turing
    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd ab = a.array() * b.array(); 
        Eigen::VectorXd I(a.rows(), 1);
        I.setIdentity();
        a = (a - timeStep * s * alpha) / (I - timeStep * da * L + timeStep * s * I - timeStep * s * b);
        ab = a.array() * b.array();
        b = (b - timeStep * s * beta) / (I - timeStep * db * L + timeStep * s * I - timeStep * s * a);
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Turing Explicit", a);
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  //ImGui::PushItemWidth(100);

  //// Curvature
  //if (ImGui::Button("add curvature")) {
  //  addCurvatureScalar();
  //}
  //
  //// Normals 
  //if (ImGui::Button("add normals")) {
  //  computeNormals();
  //}

  //// Param
  //if (ImGui::Button("add parameterization")) {
  //  computeParameterization();
  //}

  //// Geodesics
  //if (ImGui::Button("compute distance")) {
  //  computeDistanceFrom();
  //}
  //ImGui::SameLine();
  //ImGui::InputInt("source vertex", &iVertexSource);

  //ImGui::PopItemWidth();
 
  // Explicit Heat Equation
  ImGui::Button("Explicit Heat Equation");

  static float timeStep1 = 0.05;
  ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step##explicit", &timeStep1);
  ImGui::PopItemWidth();

  static float max1 = 200;
  ImGui::SameLine();
  ImGui::PushItemWidth(75);
  ImGui::InputFloat("Max##explicit", &max1);
  ImGui::PopItemWidth();
  
  static float numSteps1 = 0;
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  if (ImGui::SliderFloat("Num Steps##explicit", &numSteps1, 0, max1)) {
      computeExplicitHeat(numSteps1, timeStep1);
  }
  ImGui::PopItemWidth();

  // Semi-implicit Heat Equation
  ImGui::Button("Semi-implicit Heat Equation");

  static float timeStep2 = 0.05;
  ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step##implicit", &timeStep2);
  ImGui::PopItemWidth();

  static float max2 = 200;
  ImGui::SameLine();
  ImGui::PushItemWidth(75);
  ImGui::InputFloat("Max##implicit", &max2);
  ImGui::PopItemWidth();

  static float numSteps2 = 0;
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  if (ImGui::SliderFloat("Num Steps##implicit", &numSteps2, 0, max2)) {
      computeImplicitHeat(numSteps2, timeStep2);
  }
  ImGui::PopItemWidth();

  // Reaction Diffusion Turing
  ImGui::Button("Turing Reaction Diffusion");

  static float timeStep3 = 0.05;
  ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step##Turing", &timeStep3);
  ImGui::PopItemWidth();

  static float max3 = 200;
  ImGui::SameLine();
  ImGui::PushItemWidth(75);
  ImGui::InputFloat("Max##Turing", &max3);
  ImGui::PopItemWidth();

  static float numSteps3 = 0;
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  if (ImGui::SliderFloat("Num Steps##Turing", &numSteps3, 0, max3)) {
      computeExplicitReactionDiffusionTuring(numSteps3, timeStep3);
  }
  ImGui::PopItemWidth();

  // Reaction Diffusion Gray and Scott
  ImGui::Button("Gray & Scott Reaction Diffusion");

  static float timeStep4 = 0.05;
  ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step##Scott", &timeStep4);
  ImGui::PopItemWidth();

  static float max4 = 200;
  ImGui::SameLine();
  ImGui::PushItemWidth(75);
  ImGui::InputFloat("Max##Scott", &max4);
  ImGui::PopItemWidth();

  static float numSteps4 = 0;
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  if (ImGui::SliderFloat("Num Steps##Scott", &numSteps4, 0, max4)) {
      computeExplicitReactionDiffusionScott(numSteps4, timeStep4);
  }
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

  Eigen::VectorXd noise(meshV.rows());
  noise.setRandom();
  noise_a = noise;

  Eigen::VectorXd noise2(meshV.rows());
  noise2.setRandom();
  noise_b = noise2;

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
