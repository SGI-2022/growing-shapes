#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/loop.h>

#include <sstream>

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

struct RDParams
{
    double Du, Dv;
    double F;
    double k;

    double dt;
    
    Eigen::SparseMatrix<double> M;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Usolver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Vsolver;
};

void initializeParams(double Du, double Dv, double F, double k, double dt, RDParams &params)
{
    params.Du = Du;
    params.Dv = Dv;
    params.F = F;
    params.k = k;

    params.dt = dt;
    
    igl::massmatrix(meshV, meshF, igl::MassMatrixType::MASSMATRIX_TYPE_BARYCENTRIC, params.M);
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::SparseMatrix<double> Amat = params.M - params.dt * params.Du * L;
    Eigen::SparseMatrix<double> Bmat = params.M - params.dt * params.Dv * L;

    params.Usolver.compute(Amat);
    params.Vsolver.compute(Bmat);
}

void simulateOneStep(const Eigen::VectorXd& U, const Eigen::VectorXd& V, Eigen::VectorXd& newU, Eigen::VectorXd& newV, RDParams& params)
{
    int nverts = meshV.rows();

    Eigen::VectorXd F(nverts);
    Eigen::VectorXd G(nverts);
    for (int i = 0; i < nverts; i++)
    {
        F[i] = -U[i] * V[i] * V[i] + params.F * (1 - U[i]);
        G[i] = U[i] * V[i] * V[i] - (params.F + params.k) * V[i];
    }

    Eigen::VectorXd Urhs = params.M * (U + params.dt * F);
    Eigen::VectorXd Vrhs = params.M * (V + params.dt * G);

    newU = params.Usolver.solve(Urhs);
    newV = params.Vsolver.solve(Vrhs);

}

void setDistanceBased(Eigen::Vector3d point, double dist, Eigen::VectorXd& f, double val)
{
    int nverts = meshV.rows();
    for (int i = 0; i < nverts; i++)
    {
        double d = (point - meshV.row(i).transpose()).norm();
        if (d <= dist)
            f[i] = val;
    }
}

void initializeUV(Eigen::VectorXd& U, Eigen::VectorXd& V)
{
    int nverts = meshV.rows();
    U.resize(nverts);
    V.resize(nverts);
    U.setConstant(1.0);
    V.setConstant(0.0);
    Eigen::Vector3d pt = meshV.row(0).transpose();
    setDistanceBased(pt, 0.1, U, 0.5);
    setDistanceBased(pt, 0.1, V, 0.25);
}

int main(int argc, char** argv) {
    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    std::string filename = "../spot.obj";
    std::cout << "loading: " << filename << std::endl;

    Eigen::MatrixXd origV;
    Eigen::MatrixXi origF;

    // Read the mesh
    igl::readOBJ(filename, origV, origF);

    igl::loop(origV, origF, meshV, meshF, 2);
    
    double Du = 2e-5;
    double Dv = 1e-5;

    double F = 0.04;
    double k = 0.06;

    double dt = 1.0;

    bool animating = true;

    // Register the mesh with Polyscope
    auto* mesh = polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

    RDParams params;
    int nverts = meshV.rows();
    initializeParams(Du, Dv, F, k, dt, params);

    Eigen::VectorXd U(nverts);
    Eigen::VectorXd V(nverts);
    initializeUV(U, V);

    auto *Ucolor = mesh->addVertexScalarQuantity("U", U);
    Ucolor->setEnabled(true);
    mesh->addVertexScalarQuantity("V", V);

    int stepsPerFrame = 1;

    // Add the callback
    polyscope::state::userCallback = [&]()
    {
        bool reset = false;
        if (ImGui::Button("Reset Animation"))
            reset = true;

        if (animating)
        {
            if (ImGui::Button("Pause Animation"))
                animating = false;
        }
        else
        {
            if (ImGui::Button("Resume Animation"))
                animating = true;
        }


        ImGui::InputInt("Steps per frame", &stepsPerFrame);

        if (ImGui::InputDouble("dt", &dt))
            reset = true;

        ImGui::Separator();

        if (ImGui::InputDouble("F", &F))
            reset = true;

        if (ImGui::InputDouble("k", &k))
            reset = true;
        
        if (ImGui::InputDouble("Du", &Du))
            reset = true;

        if (ImGui::InputDouble("Dv", &Dv))
            reset = true;

        if (reset)
        {
            initializeParams(Du, Dv, F, k, dt, params);            
            initializeUV(U, V);
        }

        if (animating)
        {
            for (int i = 0; i < stepsPerFrame; i++)
            {
                Eigen::VectorXd newU, newV;
                simulateOneStep(U, V, newU, newV, params);
                U = newU;
                V = newV;
            }

            mesh->addVertexScalarQuantity("U", U);
            mesh->addVertexScalarQuantity("V", V);
        }
    };

    // Show the gui
    polyscope::show();

    return 0;
}
