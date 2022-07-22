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
    Eigen::VectorXd s;
    Eigen::VectorXd alpha;
    Eigen::VectorXd beta;
    double da;
    double db;
    double dt;

    Eigen::SparseMatrix<double> M;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Asolver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Bsolver;
};

void makePerturbedAlphaBeta(double alpha, double beta, double s, double da, double db, double perturbPercent, double dt, RDParams &params)
{
    int n = meshV.rows();
    params.alpha.resize(n);
    params.alpha.setConstant(alpha);
    Eigen::VectorXd alphanoise(n);
    alphanoise.setRandom();
    params.alpha += alpha * perturbPercent * 0.01 * alphanoise;

    params.beta.resize(n);
    params.beta.setConstant(beta);
    Eigen::VectorXd betanoise(n);
    betanoise.setRandom();
    params.beta += beta * perturbPercent * 0.01 * betanoise;

    params.s.resize(n);
    params.s.setConstant(s);

    params.da = da;
    params.db = db;    

    params.dt = dt;
    
    igl::massmatrix(meshV, meshF, igl::MassMatrixType::MASSMATRIX_TYPE_BARYCENTRIC, params.M);
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::SparseMatrix<double> Amat = params.M - params.dt * params.da * L;
    Eigen::SparseMatrix<double> Bmat = params.M - params.dt * params.db * L;

    params.Asolver.compute(Amat);
    params.Bsolver.compute(Bmat);
}

void simulateOneStep(const Eigen::VectorXd& a, const Eigen::VectorXd& b, Eigen::VectorXd& newa, Eigen::VectorXd& newb, RDParams& params)
{
    int nverts = meshV.rows();

    Eigen::VectorXd F(nverts);
    Eigen::VectorXd G(nverts);
    for (int i = 0; i < nverts; i++)
    {
        F[i] = params.s[i] * (a[i] * b[i] - a[i] - params.alpha[i]);
        G[i] = params.s[i] * (params.beta[i] - a[i] * b[i]);
    }

    Eigen::VectorXd arhs = params.M * (a + params.dt * F);
    Eigen::VectorXd brhs = params.M * (b + params.dt * G);

    newa = params.Asolver.solve(arhs);
    newb = params.Bsolver.solve(brhs);

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

    Eigen::SparseMatrix<double> S;
    igl::loop(origV.rows(), origF, S, meshF);
    meshV = S * origV;

    double alpha = 12.0;
    double beta = 16.0;

    double s = 1.0 / 128.0;
    double da = 1e-4;
    double db = 1e-3;
    double percentNoise = 0.1;
    double dt = 0.1;

    // Register the mesh with Polyscope
    auto* mesh = polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

    RDParams params;
    int nverts = meshV.rows();
    makePerturbedAlphaBeta(alpha, beta, s, da, db, percentNoise, dt, params);

    Eigen::VectorXd a(nverts);
    Eigen::VectorXd b(nverts);
    a.setConstant(4.0);
    b.setConstant(4.0);

    auto *acolor = mesh->addVertexScalarQuantity("a", a);
    acolor->setEnabled(true);
    mesh->addVertexScalarQuantity("b", b);

    int stepsPerFrame = 1;

    // Add the callback
    polyscope::state::userCallback = [&]()
    {
        bool reset = false;
        if (ImGui::Button("Reset Animation"))
            reset = true;

        ImGui::InputInt("Steps per frame", &stepsPerFrame);

        ImGui::Separator();

        if (ImGui::InputDouble("dt", &dt))
            reset = true;

        if (ImGui::InputDouble("s", &s))
            reset = true;

        if (ImGui::InputDouble("da", &da))
            reset = true;
        
        if (ImGui::InputDouble("db", &db))
            reset = true;

        if (ImGui::InputDouble("alpha", &alpha))
            reset = true;
        
        if (ImGui::InputDouble("beta", &beta))
            reset = true;

        if (ImGui::InputDouble("% noise", &percentNoise))
            reset = true;

        if (reset)
        {
            makePerturbedAlphaBeta(alpha, beta, s, da, db, percentNoise, dt, params);
            a.setConstant(4.0);
            b.setConstant(4.0);
        }

        for (int i = 0; i < stepsPerFrame; i++)
        {
            Eigen::VectorXd newa, newb;
            simulateOneStep(a, b, newa, newb, params);
            a = newa;
            b = newb;
        }
        mesh->addVertexScalarQuantity("a", a);
        mesh->addVertexScalarQuantity("b", b);
        
    };

    // Show the gui
    polyscope::show();

    return 0;
}
