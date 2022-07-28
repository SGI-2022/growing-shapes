// void create_triangles_set(Eigen::MatrixXi F, Eigen::MatrixXd V, Eigen::VectorXd G, Eigen::MatrixXd &T){
//     T.setZero();
//     float a, b, c, x, y;
//     int ind1, ind2, ind3;
//     int rF = F.rows();
//     for(int i=0;i<rF;i++){
//         ind1 = F(i,0);
//         ind2 = F(i,1);
//         ind3 = F(i,2);
//         a = 0.5*(G(ind1)+G(ind2))*(V.row(ind1) - V.row(ind2)).norm();
//         b = 0.5*(G(ind2)+G(ind3))*(V.row(ind2) - V.row(ind3)).norm();
//         c = 0.5*(G(ind3)+G(ind1))*(V.row(ind3) - V.row(ind1)).norm();
//         x = (c*c-b*b+a*a)/(2*a);
//         y = sqrt(c*c-x*x);

//         T.row(i) << 0,0,0,a,0,0,x,y,0;
//     }
// }

// void create_diamond_faces(Eigen::MatrixXi F, Eigen::MatrixXi &diamondF){
//     Eigen::SparseMatrix<int> adj2;
//     int sizeF = F.rows();
//     Eigen::MatrixXi diamondFtemp(sizeF*3,4);
//     diamondFtemp.setZero();
//     int ind1, ind2, ind3;
//     igl::adjacency_matrix(F,adj2);
//     Eigen::MatrixXi adj = Eigen::MatrixXi(adj2);
//     int sizeA = adj.rows();
//     int c=0;
//     for(int i=0;i<sizeF;i++){
//         ind1 = F(i,0);
//         ind2 = F(i,1);
//         ind3 = F(i,2);      
//         for(int j=0;j<sizeA;j++){
//             //std::cout << "here" << std::endl; 
//             if (adj(ind1,j)==1 && adj(ind2,j)==1 && j!=ind3){
//                 diamondFtemp.row(c) << ind3, ind1, j, ind2;
//                 c=c+1;
//             }else if(adj(ind2,j)==1 && adj(ind3,j)==1 && j!=ind1){
//                 diamondFtemp.row(c) << ind3, ind1, ind2, j;
//                 c=c+1;
//             } else if(adj(ind3,j)==1 && adj(ind1,j)==1 && j!=ind2){
//                 diamondFtemp.row(c) << ind3, j, ind1, ind2;
//                 c=c+1;
//             }
//         }
//     }

//     diamondF = diamondFtemp.block(0,0,c-1,4);
// }

// void create_diamonds_set(Eigen::MatrixXi F, Eigen::MatrixXd V, Eigen::MatrixXd &D){
//     // NO GROWTH?
//     D.setZero();
//     float a, b, c, d, e, g, x1, y1, x2, y2;
//     int ind1, ind2, ind3, ind4;
//     int rF = F.rows();
//     for(int i=0;i<rF;i++){
//         //std::cout << i << std::endl;
//         ind1 = F(i,0);
//         ind2 = F(i,1);
//         ind3 = F(i,2);
//         ind4 = F(i,3);
// //std::cout << rF<< " " << ind1 <<" "<< ind2<<" "<< ind3<<" "<< ind4 << std::endl;
//         a = (V.row(ind1) - V.row(ind2)).norm();
//         b = (V.row(ind2) - V.row(ind3)).norm();
//         c = (V.row(ind3) - V.row(ind1)).norm();
//         d = (V.row(ind4) - V.row(ind3)).norm();
//         e = (V.row(ind4) - V.row(ind1)).norm();
//         g = (V.row(ind4) - V.row(ind2)).norm();

//         x1 = (c*c-b*b+a*a)/(2*a);
//         y1 = sqrt(c*c-x1*x1);
        
//         x2= (e*e-g*g+a*a)/(2*a);
//         y2 = sqrt(e*e-x2*x2);

//         D.row(i) << 0,0,0,a,0,0,x1,y1,0,x2,y2,0;
//     }
// }

// void update(Eigen::MatrixXd orig, Eigen::MatrixXd set, float alpha, Eigen::MatrixXd &updated_triangle_piece ){
//     Eigen::MatrixXd R;
//     Eigen::MatrixXd res;
//     Eigen::VectorXd t;
//     Eigen::MatrixXd set1;

//     int fd = (int)set.cols()/3;
//     set.resize(3,fd);
//     set1=set.transpose();
//     igl::procrustes(set1,orig,false,false,R,t);
//     res = (set1*R).rowwise() + t.transpose();
//     updated_triangle_piece = (1-alpha)*orig+alpha*res;

// }

// void transform(Eigen::MatrixXi Ft, Eigen::MatrixXi Fd, Eigen::MatrixXd V, Eigen::MatrixXd T, Eigen::MatrixXd D, int nbIter, float alpha, float beta, Eigen::MatrixXd &newV){
//     newV = V;
//     Eigen::MatrixXd set_piece;
//     Eigen::MatrixXd original_piece;
//     Eigen::MatrixXd updated_piece;
//     Eigen::VectorXi inds;
//     Eigen::VectorXi inds2(1);


//     int sizeT = T.rows();
//     int sizeD;

//     for(int i=0;i<nbIter;i++){

//         std::cout << "Iteration: "<< i << std::endl;

//         for(int j=0;j<sizeT;j++){
//             // Alpha
//             inds = Ft.row(j);
//             inds2(0)=j;
//             igl::slice(newV,inds,1,original_piece);

//             igl::slice(T,inds2,1,set_piece);
//             update(original_piece, set_piece, alpha, updated_piece);

//             igl::slice_into(updated_piece,inds,1,newV);
//         }
//         // create_diamonds_set(Fd, newV, D);
//         // sizeD = D.rows();
//         // for(int j=0;j<sizeD;j++){
//         //     // Beta
//         //     inds = Fd.row(j);
//         //     inds2(0)=j;
//         //     igl::slice(newV,inds,1,original_piece);
//         //     igl::slice(D,inds2,1,set_piece);
//         //     update(original_piece, set_piece, beta, updated_piece);
//         //     igl::slice_into(updated_piece,inds,1,newV);
//         // }


//     }

// }

// int main(int argc, char **argv){
//   // Options
//   //polyscope::options::autocenterStructures = true;
//   polyscope::view::windowWidth = 1024;
//   polyscope::view::windowHeight = 1024;

//   // Initialize polyscope
//   polyscope::init();

//   std::string filename = "../this.obj";
//   std::cout << "loading: " << filename << std::endl;

//   Eigen::MatrixXd meshV, newV;
//   Eigen::MatrixXi meshF;
//   igl::readOBJ(filename, meshV, meshF);

//   int rF = meshF.rows();
//   int rV = meshV.rows();
//   int cV = meshV.cols();

//   Eigen::MatrixXi diamondF;

//   Eigen::MatrixXd setT(rF,cV*3);
//   Eigen::VectorXd G(rV);
//   G.setConstant(2.2);
//   for(int i=0;i<rV;i++){
//       if(i%30==0){
//           G(i)=1.4;
//       }
//   }

//   int nbIter=10;
//   float alpha=1;
//   float beta=0.0;

//   create_diamond_faces(meshF, diamondF);
//   int dF = diamondF.rows();
//   Eigen::MatrixXd setD(dF,12);
//   create_triangles_set(meshF, meshV, G, setT);
//   //create_diamonds_set(diamondF, meshV, setD);

//   transform(meshF,diamondF,meshV,setT,setD,nbIter,alpha,beta,newV);

//   polyscope::registerSurfaceMesh("input mesh", newV,meshF);
//   polyscope::registerSurfaceMesh("input mesh 2", meshV,meshF);

//   polyscope::show();

//   return 0;

// }
