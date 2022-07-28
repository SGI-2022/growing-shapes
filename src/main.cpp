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
#include <igl/procrustes.h>
#include <igl/rotate_vectors.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/adjacency_matrix.h>

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

double t=1;//0.1;
int iter = 0;
double s=(double)0.007812;
double al = 0.018;//12;
double be=0.051;//16;
double da=1.0;//0.00005;
double db=0.5;//0.0002;

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
      double rn = rand() % 201;
      double rn1 = rand() % 201;
      rn=(double)(rn-100)/10000;
      rn1=(double)(rn1-100)/10000;
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
    double da=1.0;//0.25*0.25;//*0.25;
    double db=0.5;//0.25;
    double k=0.062;//0.062;
    double f=0.0545;//0.055;
    double s =(double)128;//1.0;

    for (int i=0;i<iter;i++){
      A_old=A;
    A = (A+t*(da*L*A - termbyterm(A_old,termbyterm(B,B))+f*C-f*A)).eval();
    B = (B+t*(db*L*B + termbyterm(A_old,termbyterm(B,B))-(k+f)*B)).eval();
    }

    auto scalarQ = polyscope::getSurfaceMesh("input mesh")
            ->addVertexScalarQuantity("reaction diffusion", A);
    double m1 = A.minCoeff();
    double m2 = A.maxCoeff();
    scalarQ->setMapRange({m1,m2});
}

void addReactionDiffusion2() {
    using namespace Eigen;

    int r = meshV.rows();

    VectorXd A = res.block(0,iter,r,1);

    // polyscope::getSurfaceMesh("input mesh")
    //          ->addVertexScalarQuantity("reaction diffusion 2", A);
    auto scalarQ = polyscope::getSurfaceMesh("input mesh")
           ->addVertexScalarQuantity("reaction diffusion 2", A);
    scalarQ->setMapRange({0.0,255.0});
}

Eigen::MatrixXd reactiondiffusionimplicit(int ite){
  using namespace Eigen;
  SparseMatrix<double> L;
  igl::cotmatrix(meshV,meshF,L);

  int r = meshV.rows();
  VectorXd A(r), A_old(r), B(r), B_old(r), C(r), beta(r), alpha(r); 
  alpha = VectorXd::Constant(r,1,al)+al*0.001*rd;
  beta = VectorXd::Constant(r,1,be)+be*0.001*rd;
  for(int i=0;i<r;i++){
      A(i) = 1.0;
      B(i) = 0.0;
      if(((meshV(i,0)-meshV(100,0))*(meshV(i,0)-meshV(100,0))+(meshV(i,1)-meshV(100,1))*(meshV(i,1)-meshV(100,1))+(meshV(i,2)-meshV(100,2))*(meshV(i,2)-meshV(100,2)))<0.01){
          B(i)=1.0;
      }

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

  double s = (double)1/128;
  MatrixXd res(r,ite+1);
  res.setZero();
  res.block(0,0,r,1)=(A-B)*255;
  for (int i=1;i<ite+1;i++){
    A_old=A;
    A = (A+t*( - termbyterm(A_old,termbyterm(B,B))+al*C-al*A)).eval();
    B = (B+t*( termbyterm(A_old,termbyterm(B,B))-(be+al)*B)).eval();
    A=solver1.solve(A);
    B=solver2.solve(B);
    res.block(0,i,r,1)=(A-B)*255;
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
}


void callback() {

  ImGui::PushItemWidth(100);

//  ImGui::Inputdouble("t", &t);
//  ImGui::SameLine();
    ImGui::SliderInt("iter", &iter, 0, 5000);
//  ImGui::Sliderdouble("s", &s, 0, 1);
//  ImGui::Sliderdouble("alpha", &al, 0, 30);
//  ImGui::SameLine();
//  ImGui::Sliderdouble("beta", &be, 0, 30);
//  ImGui::Sliderdouble("da", &da, 0, 1);
//  ImGui::SameLine();
//  ImGui::Sliderdouble("db", &db, 0, 1);
  //reactiondiffusionimplicit();
  addReactionDiffusion2();

  ImGui::SameLine();
  ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

void create_triangles_set(Eigen::MatrixXi F, Eigen::MatrixXd V, Eigen::VectorXd G, Eigen::MatrixXd &T){
    T.setZero();
    float a, b, c, x, y;
    int ind1, ind2, ind3;
    int rF = F.rows();
    for(int i=0;i<rF;i++){
        ind1 = F(i,0);
        ind2 = F(i,1);
        ind3 = F(i,2);
        a = 0.5*(G(ind1)+G(ind2))*(V.row(ind1) - V.row(ind2)).norm();
        b = 0.5*(G(ind2)+G(ind3))*(V.row(ind2) - V.row(ind3)).norm();
        c = 0.5*(G(ind3)+G(ind1))*(V.row(ind3) - V.row(ind1)).norm();
        x = (c*c-b*b+a*a)/(2*a);
        y = sqrt(c*c-x*x);

        T.row(i) << 0,0,0,a,0,0,x,y,0;
    }
}

void create_diamond_faces(Eigen::MatrixXi F, Eigen::MatrixXi &diamondF){
    Eigen::SparseMatrix<int> adj2;
    int sizeF = F.rows();
    Eigen::MatrixXi diamondFtemp(sizeF*3,4);
    diamondFtemp.setZero();
    int ind1, ind2, ind3;
    igl::adjacency_matrix(F,adj2);
    Eigen::MatrixXi adj = Eigen::MatrixXi(adj2);
    int sizeA = adj.rows();
    int c=0;
    for(int i=0;i<sizeF;i++){
        ind1 = F(i,0);
        ind2 = F(i,1);
        ind3 = F(i,2);      
        for(int j=0;j<sizeA;j++){
            //std::cout << "here" << std::endl; 
            if (adj(ind1,j)==1 && adj(ind2,j)==1 && j!=ind3){
                diamondFtemp.row(c) << ind3, ind1, j, ind2;
                c=c+1;
            }else if(adj(ind2,j)==1 && adj(ind3,j)==1 && j!=ind1){
                diamondFtemp.row(c) << ind3, ind1, ind2, j;
                c=c+1;
            } else if(adj(ind3,j)==1 && adj(ind1,j)==1 && j!=ind2){
                diamondFtemp.row(c) << ind3, j, ind1, ind2;
                c=c+1;
            }
        }
    }

    diamondF = diamondFtemp.block(0,0,c-1,4);
}

void create_diamonds_set(Eigen::MatrixXi F, Eigen::MatrixXd V, Eigen::MatrixXd &D){
    // NO GROWTH?
    D.setZero();
    float a, b, c, d, e, g, x1, y1, x2, y2;
    int ind1, ind2, ind3, ind4;
    int rF = F.rows();
    for(int i=0;i<rF;i++){
        //std::cout << i << std::endl;
        ind1 = F(i,0);
        ind2 = F(i,1);
        ind3 = F(i,2);
        ind4 = F(i,3);
//std::cout << rF<< " " << ind1 <<" "<< ind2<<" "<< ind3<<" "<< ind4 << std::endl;
        a = (V.row(ind1) - V.row(ind2)).norm();
        b = (V.row(ind2) - V.row(ind3)).norm();
        c = (V.row(ind3) - V.row(ind1)).norm();
        d = (V.row(ind4) - V.row(ind3)).norm();
        e = (V.row(ind4) - V.row(ind1)).norm();
        g = (V.row(ind4) - V.row(ind2)).norm();

        x1 = (c*c-b*b+a*a)/(2*a);
        y1 = sqrt(c*c-x1*x1);
        
        x2= (e*e-g*g+a*a)/(2*a);
        y2 = sqrt(e*e-x2*x2);

        D.row(i) << 0,0,0,a,0,0,x1,y1,0,x2,y2,0;
    }
}

void update(Eigen::MatrixXd orig, Eigen::MatrixXd set, float alpha, Eigen::MatrixXd &updated_triangle_piece ){
    Eigen::MatrixXd R;
    Eigen::MatrixXd res;
    Eigen::VectorXd t;
    Eigen::MatrixXd set1;

    int fd = (int)set.cols()/3;
    set.resize(3,fd);
    set1=set.transpose();
    igl::procrustes(set1,orig,false,false,R,t);
    res = (set1*R).rowwise() + t.transpose();
    updated_triangle_piece = (1-alpha)*orig+alpha*res;

}

void transform(Eigen::MatrixXi Ft, Eigen::MatrixXi Fd, Eigen::MatrixXd V, Eigen::MatrixXd T, Eigen::MatrixXd D, int nbIter, float alpha, float beta, Eigen::MatrixXd &newV){
    newV = V;
    Eigen::MatrixXd set_piece;
    Eigen::MatrixXd original_piece;
    Eigen::MatrixXd updated_piece;
    Eigen::VectorXi inds;
    Eigen::VectorXi inds2(1);


    int sizeT = T.rows();
    int sizeD;

    for(int i=0;i<nbIter;i++){

        std::cout << "Iteration: "<< i << std::endl;

        for(int j=0;j<sizeT;j++){
            // Alpha
            inds = Ft.row(j);
            inds2(0)=j;
            igl::slice(newV,inds,1,original_piece);

            igl::slice(T,inds2,1,set_piece);
            update(original_piece, set_piece, alpha, updated_piece);

            igl::slice_into(updated_piece,inds,1,newV);
        }
        create_diamonds_set(Fd, newV, D);
        sizeD = D.rows();
        for(int j=0;j<sizeD;j++){
            // Beta
            inds = Fd.row(j);
            inds2(0)=j;
            igl::slice(newV,inds,1,original_piece);
            igl::slice(D,inds2,1,set_piece);
            update(original_piece, set_piece, beta, updated_piece);
            igl::slice_into(updated_piece,inds,1,newV);
        }


    }

}

int main(int argc, char **argv){
  // Options
  //polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string filename = "../this.obj";
  std::cout << "loading: " << filename << std::endl;

  Eigen::MatrixXd meshV, newV;
  Eigen::MatrixXi meshF;
  igl::readOBJ(filename, meshV, meshF);

  int rF = meshF.rows();
  int rV = meshV.rows();
  int cV = meshV.cols();

  Eigen::MatrixXi diamondF;

  Eigen::MatrixXd setT(rF,cV*3);
  Eigen::VectorXd G(rV);
  G.setConstant(2.2);
  for(int i=0;i<rV;i++){
      if(i%30==0){
          G(i)=1.4;
      }
  }

  int nbIter=10;
  float alpha=1;
  float beta=0.0;

  create_diamond_faces(meshF, diamondF);
  int dF = diamondF.rows();
  Eigen::MatrixXd setD(dF,12);
  create_triangles_set(meshF, meshV, G, setT);
  //create_diamonds_set(diamondF, meshV, setD);

  transform(meshF,diamondF,meshV,setT,setD,nbIter,alpha,beta,newV);

  polyscope::registerSurfaceMesh("input mesh", newV,meshF);
  polyscope::registerSurfaceMesh("input mesh 2", meshV,meshF);

  polyscope::show();

  return 0;

}

// int main(int argc, char **argv) {

//   // Options
//   polyscope::options::autocenterStructures = true;
//   polyscope::view::windowWidth = 1024;
//   polyscope::view::windowHeight = 1024;

//   // Initialize polyscope
//   polyscope::init();

//   std::string filename = "../spot.obj";
//   std::cout << "loading: " << filename << std::endl;

//   // Read the mesh
//   Eigen::MatrixXd origV, meshV1;
//   Eigen::MatrixXi origF, meshF1;

//   // Read the mesh
//   igl::readOBJ(filename, origV, origF);

//   Eigen::SparseMatrix<double> S, S1;
//   igl::loop(origV.rows(), origF, S1, meshF1);
//   meshV1 = S1 * origV;
//   igl::loop(meshV1.rows(), meshF1, S, meshF);
//   meshV = S * meshV1;

//   int r=meshV.rows();
//   rd = Eigen::VectorXd::Random(r, 1);
//   res=reactiondiffusionimplicit(5000);

//   // Register the mesh with Polyscope
//   polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

//   // Add the callback
//   polyscope::state::userCallback = callback;

//   // Show the gui
//   polyscope::show();

//   return 0;
// }