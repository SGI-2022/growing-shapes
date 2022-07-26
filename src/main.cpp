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
#include <igl/loop.h>
#include <igl/writeOBJ.h>

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

    int randomNum = rand() % 2;
    Eigen::VectorXd U = Eigen::VectorXd::Constant(L.rows(), 1, 1);
    Eigen::VectorXd V = Eigen::VectorXd::Constant(L.rows(), 1, 0);

    Eigen::VectorXd F = Eigen::VectorXd::Constant(L.rows(), 1, 0.0545);  // feed rate
    Eigen::VectorXd k = Eigen::VectorXd::Constant(L.rows(), 1, 0.062); // degrading rate

    float du = 1; // diffusion rate
    float dv = (float)0.5; // diffusion rate
    for (int i = 0; i < meshV.rows(); i++) {
        if (((meshV(i, 0) - meshV(100, 0)) * (meshV(i, 0) - meshV(100, 0)) + (meshV(i, 1) - meshV(100, 1)) * (meshV(i, 1) - meshV(100, 1)) + (meshV(i, 2) - meshV(100, 2)) * (meshV(i, 2) - meshV(100, 2))) < 0.01) {
            V(i) = 1.0;
        }
    }

    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd UVV = U.array() * V.array() * V.array();
        U = (U + timeStep * (-1 * UVV + F - F * U) + du * timeStep * L * U).eval();
        UVV = U.array() * V.array() * V.array();
        V = (V + timeStep * (UVV - V * (F + k)) + dv * timeStep * L * V).eval();
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Gray & Scott Explicit", U);
    //mesh->setMapRange({ -0.1,0.1 });
}

void computeImplicitReactionDiffusionTuring(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd a = Eigen::VectorXd::Constant(L.rows(), 1, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Constant(L.rows(), 1, 4);

    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(L.rows(), 1, 12) + noise_a;  // decay rate of a
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(L.rows(), 1, 16) + noise_b; // growing rate of b
    float da = (float)1 / 16; // diffusion rate
    float db = (float)1 / 4; // diffusion rate
    float s = (float)1 / 128; // reaction rate

    // Turing
    Eigen::SparseMatrix<double> I(L.rows(), L.rows());
    I.setIdentity();

    Eigen::SparseMatrix<double> a_diag(L.rows(), L.rows());
    Eigen::SparseMatrix<double> b_diag(L.rows(), L.rows());

    for (int i = 0; i < numSteps; i++) {
        for (int i = 0; i < L.rows(); i++) {
            a_diag.coeffRef(i, i) = a(i, 1);
            b_diag.coeffRef(i, i) = b(i, 1);
        }
        Eigen::SparseMatrix<double> temp1 = (I - timeStep * da * L + timeStep * s * I - timeStep * s * b_diag).eval();
        Eigen::SparseMatrix<double> temp2 = (I - timeStep * db * L + timeStep * s * a_diag).eval();

        // do I need two solvers??
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver1;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver2;
        solver1.compute(temp1);
        solver2.compute(temp2);

        a = solver1.solve(a - timeStep * s * alpha);
        b = solver2.solve(b + timeStep * s * beta);
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Turing Implicit", a);
}

void computeImplicitReactionDiffusionScott(float numSteps, float timeStep, double F_val, double k_val) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd U = Eigen::VectorXd::Constant(L.rows(), 1, 1);
    Eigen::VectorXd V = Eigen::VectorXd::Constant(L.rows(), 1, 0);

    Eigen::VectorXd F = Eigen::VectorXd::Constant(L.rows(), 1, F_val);  // feed rate
    Eigen::VectorXd k = Eigen::VectorXd::Constant(L.rows(), 1, k_val); // degrading rate
    float du = 1; // diffusion rate
    float dv = (float)0.5; // diffusion rate

    for (int i = 0; i < meshV.rows(); i++) {
        if (((meshV(i, 0) - meshV(100, 0)) * (meshV(i, 0) - meshV(100, 0)) + (meshV(i, 1) - meshV(100, 1)) * (meshV(i, 1) - meshV(100, 1)) + (meshV(i, 2) - meshV(100, 2)) * (meshV(i, 2) - meshV(100, 2))) < 0.01) {
            V(i) = 1.0;
        }
    }

    Eigen::SparseMatrix<double> I(L.rows(), L.rows());
    I.setIdentity();

    Eigen::SparseMatrix<double> temp1 = (I - timeStep * du * L).eval();
    Eigen::SparseMatrix<double> temp2 = (I - timeStep * dv * L).eval();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver1;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver2;

    solver1.compute(temp1);
    solver2.compute(temp2);

    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd UVV = U.array() * V.array() * V.array();
        Eigen::VectorXd FU = F.array() * U.array();
        U = solver1.solve((U + timeStep * (F - FU - UVV)).eval());
        Eigen::VectorXd kV = k.array() * V.array();
        Eigen::VectorXd FV = F.array() * V.array();
        V = solver2.solve((V + timeStep * (UVV - kV - FV)).eval());
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Scott Implicit", U);
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  //// Explicit Heat Equation
  //ImGui::Button("Explicit Heat Equation");

  //static float timeStep1 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step##explicit", &timeStep1);
  //ImGui::PopItemWidth();

  //static float max1 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max##explicit", &max1);
  //ImGui::PopItemWidth();
  //
  //static float numSteps1 = 0;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(150);
  //if (ImGui::SliderFloat("Num Steps##explicit", &numSteps1, 0, max1)) {
  //    computeExplicitHeat(numSteps1, timeStep1);
  //}
  //ImGui::PopItemWidth();

  //// Semi-implicit Heat Equation
  //ImGui::Button("Semi-implicit Heat Equation");

  //static float timeStep2 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step##implicit", &timeStep2);
  //ImGui::PopItemWidth();

  //static float max2 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max##implicit", &max2);
  //ImGui::PopItemWidth();

  //static float numSteps2 = 0;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(150);
  //if (ImGui::SliderFloat("Num Steps##implicit", &numSteps2, 0, max2)) {
  //    computeImplicitHeat(numSteps2, timeStep2);
  //}
  //ImGui::PopItemWidth();

  //// Reaction Diffusion Turing
  //ImGui::Button("Turing Reaction Diffusion Explicit");

  //static float timeStep3 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step##Turing", &timeStep3);
  //ImGui::PopItemWidth();

  //static float max3 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max##Turing", &max3);
  //ImGui::PopItemWidth();

  //static float numSteps3 = 0;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(150);
  //if (ImGui::SliderFloat("Num Steps##Turing", &numSteps3, 0, max3)) {
  //    computeExplicitReactionDiffusionTuring(numSteps3, timeStep3);
  //}
  //ImGui::PopItemWidth();

  //// Reaction Diffusion Gray and Scott
  //static float timeStep4 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step##Scott", &timeStep4);
  //ImGui::PopItemWidth();

  //static float numSteps4 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max##Scott", &numSteps4);
  //ImGui::PopItemWidth();

  //if (ImGui::Button("Gray & Scott Reaction Diffusion Explicit"))
  //    computeExplicitReactionDiffusionScott(numSteps4, timeStep4);

  // Implicit Reaction Diffusion Turing
  //static float timeStep5 = 0.05;
  //ImGui::PushItemWidth(50);
  //ImGui::InputFloat("Time Step##Turing_implicit", &timeStep5);
  //ImGui::PopItemWidth();

  //static float numSteps5 = 200;
  //ImGui::SameLine();
  //ImGui::PushItemWidth(75);
  //ImGui::InputFloat("Max##Turing_implicit", &numSteps5);
  //ImGui::PopItemWidth();

  //if (ImGui::Button("Turing Reaction Diffusion Implicit"))
  //    computeImplicitReactionDiffusionTuring(numSteps5, timeStep5);

  // Implicit Reaction Diffusion Scott
  static float timeStep6 = 1;
  ImGui::PushItemWidth(50);
  ImGui::InputFloat("Time Step##Scott_implicit", &timeStep6);
  ImGui::PopItemWidth();

  static double F = 0.025;
  ImGui::PushItemWidth(75);
  ImGui::InputDouble("F##Scott_implicit", &F);
  ImGui::PopItemWidth();

  static double k = 0.06;
  ImGui::PushItemWidth(75);
  ImGui::InputDouble("k##Scott_implicit", &k);
  ImGui::PopItemWidth();

  static float numSteps6 = 200;
  ImGui::PushItemWidth(75);
  ImGui::InputFloat("Number of Steps##Scott_implicit", &numSteps6);
  ImGui::PopItemWidth();

  if (ImGui::Button("Gray-Scott Reaction Diffusion Implicit"))
      computeImplicitReactionDiffusionScott(numSteps6, timeStep6, F, k);
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
  Eigen::MatrixXd origV, meshV1;
  Eigen::MatrixXi origF, meshF1;
  // Read the mesh
  igl::readOBJ(filename, origV, origF);
  Eigen::SparseMatrix<double> S, S1;
  igl::loop(origV.rows(), origF, S1, meshF1);
  meshV1 = S1 * origV;
  igl::loop(meshV1.rows(), meshF1, S, meshF);
  meshV = S * meshV1;
 
  //// Read the mesh
  //Eigen::MatrixXd origV;
  //Eigen::MatrixXi origF;

  //igl::readOBJ(filename, origV, origF);

  //Eigen::SparseMatrix<double> S;
  //igl::loop(origV.rows(), origF, S, meshF);
  //meshV = S * origV;

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
