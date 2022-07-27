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
#include <igl/slice_into.h>
#include <igl/procrustes.h>
#include <igl/adjacency_list.h>

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

Eigen::MatrixXd V_target;
Eigen::MatrixXi F_target;

Eigen::VectorXd noise_a;
Eigen::VectorXd noise_b;

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

    Eigen::SparseMatrix<double> temp = (I - timeStep * L).eval();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(temp);

    for (int i = 0; i < numSteps; i++) {
        k = solver.solve(k);
    }

    auto temp2 = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp2->addVertexScalarQuantity("implicit heat", k);
    mesh->setMapRange({ -0.1,0.1 });
}

void computeExplicitReactionDiffusionTuring(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L); 

    Eigen::VectorXd a = Eigen::VectorXd::Constant(L.rows(), 1, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Constant(L.rows(), 1, 4);

    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(L.rows(), 1, 12) + noise_a;;  // decay rate of a
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(L.rows(), 1, 16) + noise_b; // growing rate of b
    float da = (float)1/16; // diffusion rate
    float db = (float)1/4; // diffusion rate
    float s = (float)1/128; // reaction rate
 
    // Turing
    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd ab = a.array() * b.array();
        a = (a + s * timeStep * (ab - a - alpha) + da * timeStep * L * a).eval();
        ab = a.array() * b.array();
        b = (b + s * timeStep * (beta - ab) + db * timeStep * L * b).eval();
    }

    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Turing Explicit", a);
}

void computeExplicitReactionDiffusionScott(float numSteps, float timeStep) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

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

Eigen::VectorXd computeImplicitReactionDiffusionScott(float numSteps, float timeStep, double F_val, double k_val) {
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
    return U;
}

void create2DGridManual(int n, int m, Eigen::MatrixXd& V, Eigen::MatrixXi& F, float scale_1, float scale_2, bool displace, bool rotate) {
    using namespace Eigen;
    V.resize((n + 1) * (m + 1), 3);
    F.resize(n * m * 2, 3);

    for (int i = 0; i < (n + 1); i++) { // row
        for (int j = 0; j < (m + 1); j++) { // col
            float z;
            if (displace) {
                z = j * 0.5 + (float(rand()) / float((RAND_MAX)) * float(5.0));
            }
            else {
                z = 0;
            }
            if (rotate) {
                V.row((m + 1) * i + j) = Vector3d(-j * scale_1, z, -i * scale_2);
            }
            else {
                V.row((m + 1) * i + j) = Vector3d(-j * scale_1, -i * scale_2, z);
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < (n); i++) {
        for (int j = 0; j < (m); j++) {
            F.row(idx) = Vector3i((m + 1) * i + j, (m + 1) * (i + 1) + j, (m + 1) * i + j + 1);
            idx += 1;
            F.row(idx) = Vector3i((m + 1) * (i + 1) + j, (m + 1) * (i + 1) + j + 1, (m + 1) * i + j + 1);
            idx += 1;
        }
    }
}

void create2DGridVariationsAndSave() {
    Eigen::MatrixXd V1, V2, V3, V4;
    Eigen::MatrixXi F1, F2, F3, F4;
    create2DGridManual(10, 20, V1, F1, 1, 1, false, false);
    igl::writeOBJ("../original_grid.obj", V1, F1);

    create2DGridManual(10, 20, V2, F2, 2, 3, false, false);
    igl::writeOBJ("../scaled_grid.obj", F2, F2);

    create2DGridManual(10, 20, V3, F3, 1, 1, true, false);
    igl::writeOBJ("../z_displaced_grid.obj", V3, F3);

    create2DGridManual(10, 20, V4, F4, 1, 1, false, true);
    igl::writeOBJ("../y_rotated_grid.obj", V4, F4);
}

void procruste(Eigen::MatrixXd &V_procruste) {
    int n_faces = meshF.rows();
    for (int i = 0; i < n_faces; i++) { // iterating triangle by triangle 
        Eigen::VectorXi v_indices = meshF.row(i); //vertex indices in this triangle

        Eigen::MatrixXd meshV_triangle, V_target_triangle;
        igl::slice(meshV, v_indices, 1, meshV_triangle); //slice: Y = X(I,:)
        igl::slice(V_target, v_indices, 1, V_target_triangle); //slice: Y = X(I,:)

        double scale;
        Eigen::MatrixXd R;
        Eigen::VectorXd t;
        igl::procrustes(meshV_triangle, V_target_triangle, true, false, scale, R, t);
        R *= scale;
        Eigen::MatrixXd V_procruste_triangle = (meshV_triangle * R).rowwise() + t.transpose();
        igl::slice_into(V_procruste_triangle, v_indices, 1, V_procruste); //slice into: Y(I,:) = X
    }
}

void procrusteIteration(float numSteps) {
    for (int i = 0; i < numSteps; i++) {
        procruste(meshV); //update meshV mesh
    }

    //auto temp = polyscope::getSurfaceMesh("rest mesh");
    //temp->addVertexScalarQuantity("rest", meshV);
    ////temp->addFaceScalarQuantity("rest_", meshF);

    //auto temp2 = polyscope::getSurfaceMesh("target mesh");
    //temp2->addVertexScalarQuantity("target", V_target);
    ////temp2->SurfaceMesh::addFaceScalarQuantity("target_", F_target);
}

void loopOverDiamonds(float beta) {
    // find adjacent faces
    std::vector<std::vector<int>> adjFaces;
    vector<vector<int>> notUsing;
    igl::vertex_triangle_adjacency(meshV.rows(), meshF, adjFaces, notUsing);

    unordered_set<int> faces;
    int firstTriangle = INT_MAX;
    int secondTriangle = INT_MAX;

    set<Eigen::VectorXd> verts;

    // Go through each face
    for (int k = 0; k < adjFaces.size(); k++) {

        // Go through each adjacent face
        for (int l = 0; l < adjFaces[k].size(); l++) {
            verts.clear();

            // if we haven't used this triangle yet to make a rhombus
            if (faces.find(adjFaces[k][l]) == faces.end()) {
                if (firstTriangle == INT_MAX)
                    firstTriangle = adjFaces[k][l];
                else if (secondTriangle == INT_MAX)
                    secondTriangle = adjFaces[k][l];
                else {

                    // go through each vertex of first triangle
                    for (int kk = 0; kk < 3; kk++) {
                        int vertexNum = meshF.row(firstTriangle)(kk);
                        int newVertexNum = F_target.row(firstTriangle)(kk);
                        // add vertex to done set
                        verts.insert(meshV.row(vertexNum));

                        // go through each x,y,z coordinate of the vertex
                        for (int m = 0; m < 3; m++) {
                            // V = V + beta * (V~ - V) for first triangle
                            meshV.row(vertexNum)(m) += beta * (V_target.row(newVertexNum)(m) - meshV.row(vertexNum)(m));
                        }
                    }

                    // go through each vertex of second triangle (minus the one shared vertex)
                    for (int kk = 0; kk < 3; kk++) {
                        int vertexNum = meshF.row(secondTriangle)(kk);
                        int newVertexNum = F_target.row(secondTriangle)(kk);

                        // if the vertex is not the one shared vertex
                        if (verts.find(meshV.row(vertexNum)) == verts.end())
                            // add vertex to done set
                            verts.insert(meshV.row(vertexNum));

                        // go through each x,y,z coordinate of the vertex
                        for (int m = 0; m < 3; m++) {
                            // V = V + beta * (V~ - V) for second triangle
                            meshV.row(vertexNum)(m) += beta * (V_target.row(newVertexNum)(m) - meshV.row(vertexNum)(m));
                        }
                    }
                    break;
                }

                // add completed face
                faces.insert(adjFaces[k][l]);
            }
        }
    }
}

Eigen::VectorXd growingShapes(float numSteps, float timeStep, double F_val, double k_val, float alpha, float beta) {
    // simulate reaction-diffusion
    Eigen::VectorXd U = computeImplicitReactionDiffusionScott(numSteps, timeStep, F_val, k_val);

    for (int i = 0; i < numSteps; i++) {
        // update T based on morphegens
        procrusteIteration(numSteps);

        // loop over triangles
        for (int j = 0; j < meshF.rows(); j++) {
            for (int jj = 0; i < 3; i++) {
                int vertexNum = meshF.row(j)(jj);
                int newVertexNum = F_target.row(j)(jj);

                // V = V + alpha * (V~ - V)
                meshV.row(vertexNum) += alpha * (V_target.row(newVertexNum) - meshV.row(vertexNum));
            }
        }

        loopOverDiamonds(beta);
    }

    /*auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("growing", meshV);*/
    return U;
}

void callbackHeat() {
    // Explicit Heat Equation
    ImGui::Button("Explicit Heat Equation");
    static float timeStep = 0.05;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##explicit", &timeStep);
    ImGui::PopItemWidth();

    static float max = 200;
    ImGui::SameLine();
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Max##explicit", &max);
    ImGui::PopItemWidth();

    static float numSteps = 0;
    ImGui::SameLine();
    ImGui::PushItemWidth(150);
    if (ImGui::SliderFloat("Num Steps##explicit", &numSteps, 0, max))
        computeExplicitHeat(numSteps, timeStep);
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
    if (ImGui::SliderFloat("Num Steps##implicit", &numSteps2, 0, max2))
        computeImplicitHeat(numSteps2, timeStep2);
    ImGui::PopItemWidth();
}

void callbackExplicitRd() {
    Eigen::VectorXd noise(meshV.rows());
    noise.setRandom();
    noise_a = noise;

    Eigen::VectorXd noise2(meshV.rows());
    noise2.setRandom();
    noise_b = noise2;

    // Reaction Diffusion Turing
    ImGui::Button("Turing Reaction Diffusion Explicit");

    static float timeStep = 0.05;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##Turing", &timeStep);
    ImGui::PopItemWidth();

    static float max = 200;
    ImGui::SameLine();
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Max##Turing", &max);
    ImGui::PopItemWidth();

    static float numSteps = 0;
    ImGui::SameLine();
    ImGui::PushItemWidth(150);
    if (ImGui::SliderFloat("Num Steps##Turing", &numSteps, 0, max))
        computeExplicitReactionDiffusionTuring(numSteps, timeStep);
    ImGui::PopItemWidth();

    // Reaction Diffusion Gray and Scott
    static float timeStep2 = 0.05;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##Scott", &timeStep2);
    ImGui::PopItemWidth();

    static float numSteps2 = 200;
    ImGui::SameLine();
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Max##Scott", &numSteps2);
    ImGui::PopItemWidth();

    if (ImGui::Button("Gray & Scott Reaction Diffusion Explicit"))
        computeExplicitReactionDiffusionScott(numSteps2, timeStep2);
}

void callbackImplicitRd() {
    Eigen::VectorXd noise(meshV.rows());
    noise.setRandom();
    noise_a = noise;

    Eigen::VectorXd noise2(meshV.rows());
    noise2.setRandom();
    noise_b = noise2;

    // Implicit Reaction Diffusion Turing
    static float timeStep = 0.05;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##Turing_implicit", &timeStep);
    ImGui::PopItemWidth();

    static float numSteps = 2000;
    ImGui::SameLine();
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Max##Turing_implicit", &numSteps);
    ImGui::PopItemWidth();

    if (ImGui::Button("Turing Reaction Diffusion Implicit"))
        computeImplicitReactionDiffusionTuring(numSteps, timeStep);

    // Implicit Reaction Diffusion Scott
    static float timeStep2 = 1;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##Scott_implicit", &timeStep2);
    ImGui::PopItemWidth();

    static double F = 0.025;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("F##Scott_implicit", &F);
    ImGui::PopItemWidth();

    static double k = 0.06;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("k##Scott_implicit", &k);
    ImGui::PopItemWidth();

    static float numSteps2 = 2000;
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Number of Steps##Scott_implicit", &numSteps2);
    ImGui::PopItemWidth();

    if (ImGui::Button("Gray-Scott Reaction Diffusion Implicit")) {
        Eigen::VectorXd U = computeImplicitReactionDiffusionScott(numSteps2, timeStep2, F, k);
        auto temp = polyscope::getSurfaceMesh("input mesh");
        auto mesh = temp->addVertexScalarQuantity("Scott Implicit", U);
    }
}

void callbackGrowingShapes() {
    static float timeStep = 1;
    ImGui::PushItemWidth(50);
    ImGui::InputFloat("Time Step##growing_shapes", &timeStep);
    ImGui::PopItemWidth();

    static double F = 0.025;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("F##growing_shapes", &F);
    ImGui::PopItemWidth();

    static double k = 0.06;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("k##growing_shapes", &k);
    ImGui::PopItemWidth();

    static double alpha = 0.5;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("k##growing_shapes", &alpha);
    ImGui::PopItemWidth();

    static double beta = 0.5;
    ImGui::PushItemWidth(75);
    ImGui::InputDouble("F##growing_shapes", &beta);
    ImGui::PopItemWidth();

    static float max = 50;
    ImGui::PushItemWidth(75);
    ImGui::InputFloat("Max##growing_shapes", &max);
    ImGui::PopItemWidth();

    static float numSteps = 0;
    ImGui::SameLine();
    ImGui::PushItemWidth(150);
    if (ImGui::SliderFloat("Run Procruste", &numSteps, 0, max)) {
        Eigen::VectorXd U = growingShapes(numSteps, timeStep, F, k, alpha, beta);
        auto temp = polyscope::getSurfaceMesh("input mesh");
        auto mesh = temp->addVertexScalarQuantity("Scott Implicit", U);
    }
    ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
    std::string option;
    cout << "Which would you like to display? Enter \"heat\" for heat equations, \"explicit\" for explicit reaction diffusion equations, \"implicit\" for implicit reaction diffusion equations, and \"growing\" to perform growing shapes." << endl;
    cin >> option;

    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    if (option == "growing") {
        // load original mesh
        //igl::readOBJ("../original_grid.obj", meshV, meshF);
        igl::readOBJ("../spot.obj", meshV, meshF);
        // load target mesh
        // igl::readOBJ("../scaled grid.obj", V_target, F_target);
        //igl::readOBJ("../z_displaced_grid.obj", V_target, F_target);
        // igl::readOBJ("../y rotated grid.obj", V_target, F_target);
        polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
        //polyscope::registerSurfaceMesh("target mesh", V_target, F_target);
    }
    else {
        // Read the mesh
        Eigen::MatrixXd origV, meshV1;
        Eigen::MatrixXi origF, meshF1;
        // Read the mesh
        igl::readOBJ("../spot.obj", origV, origF);
        Eigen::SparseMatrix<double> S, S1;
        igl::loop(origV.rows(), origF, S1, meshF1);
        meshV1 = S1 * origV;
        igl::loop(meshV1.rows(), meshF1, S, meshF);
        meshV = S * meshV1;

        polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
    }

    if (option == "heat")
        polyscope::state::userCallback = callbackHeat;
    if (option == "explicit")
        polyscope::state::userCallback = callbackExplicitRd;
    if (option == "implicit")
        polyscope::state::userCallback = callbackImplicitRd;
    if (option == "growing")
        polyscope::state::userCallback = callbackGrowingShapes;

    // Show the gui
    polyscope::show();

    return 0;  
}
