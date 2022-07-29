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
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
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

int iter=0;

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

void run_rd_step(Eigen::SparseMatrix<double> Y1, Eigen::SparseMatrix<double> Y2, float t, float al, float be, float da, float db, Eigen::VectorXd &A, Eigen::VectorXd &B, Eigen::VectorXd &G){
  using namespace Eigen;
  int rV = meshV.rows();

  SimplicialLDLT<SparseMatrix<double>> solver1;
  SimplicialLDLT<SparseMatrix<double>> solver2;
  solver1.compute(Y1);
  solver2.compute(Y2);
  VectorXd  A_old(rV), C(rV); 
  C.setConstant(1.0);
  std::cout << rV << std::endl;

  A_old=A;
  A = (A+t*( - termbyterm(A_old,termbyterm(B,B))+al*C-al*A)).eval();
  B = (B+t*( termbyterm(A_old,termbyterm(B,B))-(be+al)*B)).eval();
  A=solver1.solve(A);
  B=solver2.solve(B);
  for(int i=0;i<rV;i++){
    if(A(i)>B(i)){
      G(i)=0.9;
    }else if(A(i)<B(i)){
      G(i)=1.8;
    }else{
      G(i)=1.0;
    }
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

    D.setZero();
    float a, b, c, d, e, g, x1, y1, x2, y2;
    int ind1, ind2, ind3, ind4;
    int rF = F.rows();
    for(int i=0;i<rF;i++){

        ind1 = F(i,0);
        ind2 = F(i,1);
        ind3 = F(i,2);
        ind4 = F(i,3);

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

void transform(Eigen::MatrixXi Ft, Eigen::MatrixXi Fd, Eigen::MatrixXd V, Eigen::MatrixXd T, int nbIter, float alpha, float beta, Eigen::MatrixXd &newV){
    newV = V;
    Eigen::MatrixXd set_piece;
    Eigen::MatrixXd original_piece;
    Eigen::MatrixXd updated_piece;
    Eigen::VectorXi inds;
    Eigen::VectorXi inds2(1);
    int sizeT = T.rows();
    int dF = Fd.rows();
    int sizeD;
    Eigen::MatrixXd D(dF,12);


    for(int i=0;i<nbIter;i++){

        // std::cout << "Iteration: "<< i << std::endl;

        for(int j=0;j<sizeT;j++){
            //Alpha
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
            //create_diamonds_set(Fd, newV, D);
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

  std::string filename = "../original_grid.obj";
  std::cout << "loading: " << filename << std::endl;

  Eigen::MatrixXd origV, meshV1,newV;
  Eigen::MatrixXi origF, meshF1;
  igl::readOBJ(filename, origV, origF);
  double rn;
  for(int i=0;i<origV.rows();i++){
      rn = rand() % 5;
      rn = (rn-1)/10;
      origV(i,2) = origV(i,2)+rn;
  }

  Eigen::SparseMatrix<double> S, S1;
  igl::loop(origV.rows(), origF, S1, meshF1);
  meshV1 = S1 * origV;
  igl::loop(meshV1.rows(), meshF1, S, meshF);
  meshV = S * meshV1;
  int rF = meshF.rows();
  int rV = meshV.rows();
  int cV = meshV.cols();
  
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(meshV,meshF,L);

  Eigen::MatrixXi diamondF;
  Eigen::MatrixXd setT(rF,cV*3);
  Eigen::VectorXd G(rV), A(rV), B(rV);
  int nbIter=10;
  float alpha=1.0;
  float beta=0.4;
  double t=1;//0.1;
  double al = 0.018;//12;
  double be=0.051;//16;
  double da=1.0;//0.00005;
  double db=0.5;//0.0002;

  B.setConstant(0.0);
  A.setConstant(1.0);
  G.setConstant(1.0);
   for(int i=0;i<rV;i++){
      if(((meshV(i,0)-meshV(100,0))*(meshV(i,0)-meshV(100,0))+(meshV(i,1)-meshV(100,1))*(meshV(i,1)-meshV(100,1))+(meshV(i,2)-meshV(100,2))*(meshV(i,2)-meshV(100,2)))<0.2){
          B(i)=1.0;
      }
  }

  Eigen::SparseMatrix<double> Y1(rV,rV), Y2(rV,rV);
  Y1.setIdentity();
  Y2.setIdentity();
  Y1 = (Y1 - da*t*L).eval();
  Y2 = (Y2 - db*t*L).eval();

  create_diamond_faces(meshF, diamondF);
  for(int i=0;i<10;i++){
    std::cout << "Iteration: "<< i << std::endl;
    run_rd_step(Y1,Y2,t,al,be,da,db,A,B,G);
    create_triangles_set(meshF, meshV, G, setT);
    transform(meshF,diamondF,meshV,setT,nbIter,alpha,beta,newV);
  }
    polyscope::registerSurfaceMesh("original mesh", origV,origF);
    polyscope::registerSurfaceMesh("input mesh", meshV,meshF);
    polyscope::registerSurfaceMesh("output mesh", newV,meshF);
    polyscope::show();
  return 0;

}