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

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <stdlib.h> 
#include <unordered_set>
#include <utility>
#include <unsupported/Eigen/MatrixFunctions>
#include<Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/SVD>

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::MatrixXd res;
Eigen::VectorXd rd;

float t=1;//0.1;
int iter = 0;
float s=(float)0.007812;
float al = 0.055;//12;
float be=0.062;//16;
float da=1.0;//0.00005;
float db=0.5;//0.0002;

// Options for algorithms
int iVertexSource = 7;

Eigen::VectorXd termbyterm(Eigen::VectorXd A, Eigen::VectorXd B){
  int r = A.rows();
  Eigen::VectorXd res(r);
  for(int i=0;i<r;i++){
    res(i)=A(i)*B(i);
  }
  return res;
}

Eigen::MatrixXd allvals(){
    using namespace Eigen;
    SparseMatrix<double> L, M, Minv;
    igl::massmatrix(meshV,meshF,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    igl::cotmatrix(meshV,meshF,L);
    //L = Minv*L;

    int r = meshV.rows();
    VectorXd A(r), A_old(r), B(r), B_old(r), C(r), beta(r), alpha(r);

    MatrixXd res(r,iter+1);
    res.setZero();


    beta = VectorXd::Constant(r,1,al);
    alpha = VectorXd::Constant(r,1,be);
   
    for(int i=0;i<r;i++){
        A(i)=4.0;
        B(i)=4.0;
      C(i)=1;
    }

    SparseMatrix<double> Y1(r,r), Y11(r,r), Y12(r,r);
    SparseMatrix<double> Y2(r,r), Y21(r,r), Y22(r,r);
    Y1.setIdentity();
    Y2.setIdentity();
    SimplicialLDLT<SparseMatrix<double>> solver1;
    SimplicialLDLT<SparseMatrix<double>> solver2;

    Y1 = (Y1 - da*t*L).eval();
    Y2 = (Y2 - db*t*L).eval();

    solver1.compute(Y1);
    solver2.compute(Y2);

    for(int i=0;i<r;i++){
      float rn = rand() % 201;
      float rn1 = rand() % 201;
      rn=(float)(rn-100)/10000;
      rn1=(float)(rn1-100)/10000;
      beta(i) = beta(i) + rn1;
      alpha(i)=alpha(i)+rn;
    }
    res.block(0,0,r,1)=A;
    for (int i=1;i<iter+1;i++){

    A_old = A;
    A=(A+t*s*(termbyterm(A,B)-A-alpha)).eval();
    B=(B+t*s*(beta-termbyterm(A_old,B))).eval();
    A=solver1.solve(A);
    B=solver2.solve(B);
    res.block(0,i,r,1)=A;
    }

    return res;
}

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

void addReactionDiffusion() {
    using namespace Eigen;

    SparseMatrix<double> L, M, Minv;
    igl::massmatrix(meshV,meshF,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    igl::cotmatrix(meshV,meshF,L);
    L = Minv*L;

    int r = meshV.rows();
    VectorXd A(r), A_old(r), B(r), C(r), temp(r), temp1(r), beta(r), alpha(r);

    beta = VectorXd::Constant(r,1,16);
    alpha = VectorXd::Constant(r,1,12);
   
    for(int i=0;i<r;i++){
      if(i<100){
        A(i)=1.0;
        B(i)=1.0;
      }else{
        A(i)=0.0;
        B(i)=1;
      }
      C(i)=1;
    }

    //temp.setZero();
    float da=1.0;//0.25*0.25;//*0.25;
    float db=0.5;//0.25;
    float k=0.062;//0.062;
    float f=0.0545;//0.055;
    float s =(float)128;//1.0;

    for (int i=0;i<iter;i++){
      A_old=A;
    A = (A+t*(da*L*A - termbyterm(A_old,termbyterm(B,B))+f*C-f*A)).eval();
    B = (B+t*(db*L*B + termbyterm(A_old,termbyterm(B,B))-(k+f)*B)).eval();
    }

    auto scalarQ = polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("reaction diffusion", A);
    float m1 = A.minCoeff();
    float m2 = A.maxCoeff();
    scalarQ->setMapRange({m1,m2});
}

void addReactionDiffusion2() {
    using namespace Eigen;

    int r = meshV.rows();

    VectorXd A = res.block(0,iter,r,1);

    polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("reaction diffusion 2", A);
}

Eigen::MatrixXd reactiondiffusionimplicit(int ite){
  using namespace Eigen;
  SparseMatrix<double> L, M, Minv;
  igl::massmatrix(meshV,meshF,igl::MASSMATRIX_TYPE_VORONOI,M);
  //igl::invert_diag(M,Minv);
  igl::cotmatrix(meshV,meshF,L);
  //MatrixXd ss;
  //ss=MatrixXd(L);
  BDCSVD<MatrixXd> svd(MatrixXd(L), ComputeThinU | ComputeThinV);
  //MatrixXd L_new = svd.matrixU() * svd.matrixV().transpose();
  //L = L_new.sparseView();
  //L=Minv*L;

  int r = meshV.rows();
  VectorXd A(r), A_old(r), B(r), B_old(r), C(r), beta(r), alpha(r); 
  alpha = VectorXd::Constant(r,1,al)+al*0.001*rd;
  beta = VectorXd::Constant(r,1,be)+be*0.001*rd;
  for(int i=0;i<r;i++){
      A(i) = 4.0;
      B(i) = 4.0;

      C(i)=1;
  }
  SparseMatrix<double> Y1(r,r), Y2(r,r);
  Y1.setIdentity();
  Y2.setIdentity();
  SimplicialLDLT<SparseMatrix<double>> solver1;
  SimplicialLDLT<SparseMatrix<double>> solver2;
  //da=0.00005;
  //db=0.0002;
  Y1 = (Y1 - da*t*L).eval();
  Y2 = (Y2 - db*t*L).eval();
  solver1.compute(Y1);
  solver2.compute(Y2);

  float s = (float)1/128;
  MatrixXd res(r,ite+1);
  res.setZero();
  res.block(0,0,r,1)=A;
  for (int i=1;i<ite;i++){
    // Turing
    //A_old = A;
    //A=M*(A+t*s*(termbyterm(A_old,B)-A_old-alpha)).eval();
    //B=M*(B+t*s*(beta-termbyterm(A_old,B))).eval();
    A_old=A;
    A = (A+t*(da*L*A - termbyterm(A_old,termbyterm(B,B))+al*C-al*A)).eval();
    B = (B+t*(db*L*B + termbyterm(A_old,termbyterm(B,B))-(be+al)*B)).eval();
    A=solver1.solve(A);
    B=solver2.solve(B);
    res.block(0,i,r,1)=A;
    }

  return res;
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


void callback() {

  ImGui::PushItemWidth(100);

//  ImGui::InputFloat("t", &t);
//  ImGui::SameLine();
    ImGui::SliderInt("iter", &iter, 0, 20000);
//  ImGui::SliderFloat("s", &s, 0, 1);
//  ImGui::SliderFloat("alpha", &al, 0, 30);
//  ImGui::SameLine();
//  ImGui::SliderFloat("beta", &be, 0, 30);
//  ImGui::SliderFloat("da", &da, 0, 1);
//  ImGui::SameLine();
//  ImGui::SliderFloat("db", &db, 0, 1);
  //reactiondiffusionimplicit();
  addReactionDiffusion2();

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
  Eigen::MatrixXd origV;
  Eigen::MatrixXi origF;

  // Read the mesh
  igl::readOBJ(filename, origV, origF);

  Eigen::SparseMatrix<double> S;
  igl::loop(origV.rows(), origF, S, meshF);
  meshV = S * origV;

  int r=meshV.rows();
  rd = Eigen::VectorXd::Random(r, 1);
  res=reactiondiffusionimplicit(20000);

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
