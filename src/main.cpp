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

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::MatrixXd res;
Eigen::VectorXd rd;

float t=0.05;
int iter = 0;
float s=(float)0;
float al = 0;
float be=0;
float da=0;
float db=0;

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
    // MatrixXd Bm(r,r), Am(r,r), Alpham(r,r);
    // Bm.setZero();
    // Alpham.setZero();
    // Am.setZero();

    MatrixXd res(r,101);
    res.setZero();


    beta = VectorXd::Constant(r,1,al);
    alpha = VectorXd::Constant(r,1,be);
   
    for(int i=0;i<r;i++){
      if(i==100){
        A(i)=0.0;
        B(i)=4.0;
      }else{
        A(i)=4.0;
        B(i)=0.0;
      }
      C(i)=1;
    }


    // float da=(float)1;
    // float db=(float)4;
    // float k=2;//0.062;
    // float f=6;//0.055;
    // float s = (float)1/128;//1.0;
    //float alpha = 12;
    //float beta=16;

    SparseMatrix<double> Y1(r,r), Y11(r,r), Y12(r,r);
    SparseMatrix<double> Y2(r,r), Y21(r,r), Y22(r,r);
    Y1.setIdentity();
    Y2.setIdentity();
    SimplicialLDLT<SparseMatrix<double>> solver1;
    SimplicialLDLT<SparseMatrix<double>> solver2;

    Y1 = (Y1 - da*t*L).eval();
    Y2 = (Y2 - db*t*L).eval();
    //Y11=(Y1-t*da*L+s*t*(f+k)*Y1).eval();
    //Y11=(Y1-t*da*L).eval();//+s*(f+k)*Y1
    //Y12=-t*s*Y1;
    //Y21 = (Y2 - db*t*L+s*t*f*Y2).eval();
    //Y21 = (Y2 - db*t*L).eval();//+s*f*Y2
    //Y22=t*s*Y2;
    //MatrixXd concy(2*r,3*r);
    // concy.setZero();
    // concy.block(0,0,r,r)=MatrixXd(Y11);
    // concy.block(0,r,r,r)=MatrixXd(Y12);
    // concy.block(r,r,r,r)=MatrixXd(Y22);
    // concy.block(r,2*r,r,r)=MatrixXd(Y21);
    // SparseMatrix<double> spconcy = concy.sparseView();
    // solver1.compute(spconcy);
    solver1.compute(Y1);
    solver2.compute(Y2);
    // VectorXd conca(2*r), AA(2*r);

    // MatrixXd concy(r,2*r); 
    // concy.setZero();   
    // concy.block(0,0,r,r)=MatrixXd(Y11);
    // concy.block(0,r,r,r)=MatrixXd(Y21);
    // SparseMatrix<double> spconcy = concy.sparseView();
    // solver1.compute(spconcy);
    //solver1.compute(Y11);
    // solver2.compute(Y21);
    //VectorXd conca(r), AA(2*r);
    //VectorXd AA(r), BB(r);

    for(int i=0;i<r;i++){
      float rn = rand() % 201;
      float rn1 = rand() % 201;
      rn=(float)(rn-100)/10000;
      rn1=(float)(rn1-100)/10000;
      //temp(i)=A(i)*B(i);//B(i);
      //temp1(i) = (12+rn)*A(i);
      beta(i) = beta(i) + rn1;
      alpha(i)=alpha(i)+rn;
    }
    res.block(0,0,r,1)=A;
    for (int i=1;i<101;i++){
      //std::cout << i << std::endl;
    //A = (A+t*(da*L*A - temp+f*C-f*A)).eval();
    //B = (B+t*(db*L*B + temp-(k+f)*B)).eval();
    //A=(A+t*(da*L*A+s*temp-s*alpha*A-s*alpha*C)).eval();
    //B=(B+t*(db*L*B+s*beta*C-s*temp)).eval();
    //Bm.diagonal()=B;
    //Am.diagonal()=A;
    //Alpham.diagonal()=alpha;
    //Y11 = (Y1 - da*t*L-t*s*Bm+t*s*Alpham).eval();
    //Y22 = (Y2 - db*t*L+t*s*Am).eval();
    A_old = A;
    A=(A+t*s*(termbyterm(A,B)-A-alpha)).eval();
    B=(B+t*s*(beta-termbyterm(A_old,B))).eval();
    //A= solver1.solve(A);
    //B= solver2.solve(B);

    // conca.block(0,0,r,1) = A(Y1-t*s*Y1)*A-t*s*alpha;
    // conca.block(r,0,r,1) = B;
    //conca=A+B+t*s*f*C;//+t*s*beta-s*t*alpha;
    // AA=A+s*t*termbyterm(A,termbyterm(A,B))-s*t*(f+k)*A;
    // BB=B-s*t*termbyterm(A,termbyterm(A,B))+s*t*f*C-s*t*f*B;
    //AA=solver1.solve(conca);
    // A=AA.block(0,0,r,1);
    // B=AA.block(r,0,r,1);
    A=solver1.solve(A);
    B=solver2.solve(B);
    res.block(0,i,r,1)=A; //0,r

    //alpha=alpha*0.99;
    //beta=beta*0.99;
    // s=s*2;
    // if(s>1){
    //   s=1;
    // }
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
    // for(int i=0;i<r;i++){
    //   float rn = rand() % 201;
    //   float rn1 = rand() % 201;
    //   rn=(float)(rn-100)/10000;
    //   //std::cout << rn << std::endl;
    //   rn1=(float)(rn1-100)/10000;
    //   //temp(i)=A(i)*B(i);//B(i);
    //   //temp1(i) = (12+rn)*A(i);
    //   beta(i) = beta(i) + rn1;
    //   alpha(i)=alpha(i)+rn;
    // }
    //float alpha = 12;
    //float beta=16;
    for (int i=0;i<iter;i++){
      A_old=A;
    A = (A+t*(da*L*A - termbyterm(A_old,termbyterm(B,B))+f*C-f*A)).eval();
    B = (B+t*(db*L*B + termbyterm(A_old,termbyterm(B,B))-(k+f)*B)).eval();
    //A_old=A;
    //A=(A+t*(da*L*A+s*termbyterm(A_old,B)-s*A-s*alpha)).eval();
    //B=(B+t*(db*L*B+s*beta-s*termbyterm(A_old,B))).eval();
    //alpha=alpha*0.99;
    //beta=beta*0.99;
    //s=s*2;
    }

    auto scalarQ = polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("reaction diffusion", A);
    float m1 = A.minCoeff();
    float m2 = A.maxCoeff();
    scalarQ->setMapRange({m1,m2});
}

void addReactionDiffusion2() {
    using namespace Eigen;

    // SparseMatrix<double> L, M, Minv;
    // igl::massmatrix(meshV,meshF,igl::MASSMATRIX_TYPE_VORONOI,M);
    // igl::invert_diag(M,Minv);
    // igl::cotmatrix(meshV,meshF,L);
    // L = Minv*L;

    int r = meshV.rows();
    // VectorXd A(r), A_old(r), B(r), B_old(r), C(r), temp(r), temp1(r), beta(r), alpha(r);
    // MatrixXd Bm(r,r), Am(r,r), Alpham(r,r);
    // Bm.setZero();
    // Alpham.setZero();
    // Am.setZero();


    // beta = VectorXd::Constant(r,1,16);
    // alpha = VectorXd::Constant(r,1,12);
   
    // for(int i=0;i<r;i++){
    //   if(i==100){
    //     A(i)=4.0;
    //     B(i)=4.0;
    //   }else{
    //     A(i)=4.0;
    //     B(i)=4.0;
    //   }
    //   C(i)=1;
    // }

    // temp.setZero();
    // float da=0.25*0.25;
    // float db=0.25;
    // float k=0.062;//0.062;
    // float f=0.0545;//0.055;
    // float s = (float)1/128;//1.0;
    // //float alpha = 12;
    // //float beta=16;

    // int size = meshV.rows();
    // SparseMatrix<double> Y1(size,size), Y11(size,size);
    // SparseMatrix<double> Y2(size,size),Y22(size,size);
    // SimplicialLDLT<SparseMatrix<double>> solver1;
    // SimplicialLDLT<SparseMatrix<double>> solver2;
    // for(int i=0;i<size;i++){
    //    Y1.coeffRef(i,i)=1;
    //    Y2.coeffRef(i,i)=1;
    // }

    // for(int i=0;i<r;i++){
    //   float rn = rand() % 201;
    //   float rn1 = rand() % 201;
    //   rn=(float)(rn-100)/10000;
    //   rn1=(float)(rn1-100)/10000;
    //   //temp(i)=A(i)*B(i);//B(i);
    //   //temp1(i) = (12+rn)*A(i);
    //   beta(i) = beta(i) + rn1;
    //   alpha(i)=alpha(i)+rn;
    // }

    // for (int i=0;i<iter;i++){
    // //A = (A+t*(da*L*A - temp+f*C-f*A)).eval();
    // //B = (B+t*(db*L*B + temp-(k+f)*B)).eval();
    // //A=(A+t*(da*L*A+s*temp-s*alpha*A-s*alpha*C)).eval();
    // //B=(B+t*(db*L*B+s*beta*C-s*temp)).eval();
    // Bm.diagonal()=B;
    // Am.diagonal()=A;
    // Alpham.diagonal()=alpha;
    // Y11 = (Y1 - da*t*L-t*s*Bm+t*s*Alpham).eval();
    // Y22 = (Y2 - db*t*L+t*s*Am).eval();
    // solver1.compute(Y11);
    // solver2.compute(Y22);
    
    // A = (A-t*s*alpha).eval();
    // B = (B+t*s*beta).eval();
    // A= solver1.solve(A);
    // B= solver2.solve(B);
    // //alpha=alpha*0.99;
    // //beta=beta*0.99;
    // //s=s*2;
    // }

    //res=allvals();

    VectorXd A = res.block(0,iter,r,1);

    polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("reaction diffusion 2", A);
    //float m1 = A.minCoeff();
    //float m2 = A.maxCoeff();
    //scalarQ->setMapRange({m1,m2});
}

void reactiondiffusionimplicit(){
  using namespace Eigen;
  SparseMatrix<double> L, M, Minv;
  igl::massmatrix(meshV,meshF,igl::MASSMATRIX_TYPE_VORONOI,M);
  igl::invert_diag(M,Minv);
  igl::cotmatrix(meshV,meshF,L);
  //L=Minv*L;

  int r = meshV.rows();
  VectorXd A(r), A_old(r), B(r), B_old(r), C(r), beta(r), alpha(r); 
  beta = VectorXd::Constant(r,1,al)+rd;
  alpha = VectorXd::Constant(r,1,be)+rd; 
  for(int i=0;i<r;i++){
      A(i) = 1 + rd(i);
      B(i) = 1 + rd(i);;

      C(i)=1;
  }
  SparseMatrix<double> Y1(r,r), Y2(r,r);
  Y1.setIdentity();
  Y2.setIdentity();
  SimplicialLDLT<SparseMatrix<double>> solver1;
  SimplicialLDLT<SparseMatrix<double>> solver2;
  Y1 = (Y1 - da*t*L).eval();
  Y2 = (Y2 - db*t*L).eval();
  solver1.compute(Y1);
  solver2.compute(Y2);

  float s = (float)1/128;

  for (int i=1;i<iter;i++){
    /* Turing
    A_old = A;
    A=(A+t*s*(termbyterm(A_old,B)-A_old-alpha)).eval();
    B=(B+t*s*(beta-termbyterm(A_old,B))).eval();
    */
    A_old = A;
    A=(A+t*(-termbyterm(A_old,termbyterm(B,B))+0.055*(C-A))).eval();
    B=(B+t*(-(0.117)*B+termbyterm(A_old,termbyterm(B,B)))).eval();
    A=solver1.solve(A);
    B=solver2.solve(B);
    }

    polyscope::getSurfaceMesh("input mesh")
    ->addVertexScalarQuantity("reaction diffusion turing", A);

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

  ImGui::InputFloat("t", &t);
  ImGui::SameLine();
  ImGui::SliderInt("Offset", &iter, 0, 1000);
  ImGui::SliderFloat("s", &s, 0, 30);
  ImGui::SliderFloat("alpha", &al, 0, 30);
  ImGui::SameLine();
  ImGui::SliderFloat("beta", &be, 0, 30);
  ImGui::SliderFloat("da", &da, 0, 4);
  ImGui::SameLine();
  ImGui::SliderFloat("db", &db, 0, 4);
  // addLaplacianExplicit();
  // addLaplacianImplicit();
  // addReactionDiffusion();
  // addReactionDiffusion2();
  reactiondiffusionimplicit();
  addReactionDiffusion();

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
  //res=Eigen::MatrixXd::Zero(r,50);
  res=allvals();
  std::cout << res.block(79,0,1,100) << std::endl;
  //t = 0.01;

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
