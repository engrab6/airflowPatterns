/*
Lattice Boltzmann method (LBM)
*/

#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef float real;
#define ROUND(x)( (x)-int(x) <= real(0.5) ? int(x) : (int(x)+1) )  //Define Round
#define makeEven(number) ((number%2==0)?(number):(number+1))  //Make Even Number


// problem parameters
/*
const int L = 256*2;  // lattices for X direction
const int M = 128*1;  // lattices for y direction
const int N = 128*1;  //lattices for z direction)
*/

       const int L=360;                           // lattices for X direction

      const int M_temp=ROUND((0.305/0.914)*L);  //0.457=width of the geometry in y directon
      const int M=makeEven(M_temp);             // lattices for y direction

      const int N_temp=ROUND((0.457/0.914)*L);  //0.305=height of the geometry in z directon
      const int N=makeEven(N_temp);                //lattices for z direction




const double inlet_dim=.101;           // Inlet size
const double outlet_dim=.101;          // Outlet Size

const double x_length=.914;           // Length of the geometry in x direction
const int outlet_diameter=ROUND((outlet_dim/x_length)*L);  //Nondimensionalization of outlet size
const int inlet_diameter=ROUND((inlet_dim/x_length)*L);  //Nondimensionalization of outlet size



const int TIME_STEPS = 2000;  // number of time steps for which the simulation is run
const int avg_time=500;
//const int NDIR = 19;           // number of discrete velocity directions used in the D2Q9 model
const double al = 0.2;
const double DENSITY = 1.0;          // fluid density in lattice units
const double u0 = .1;    // lid velocity in lattice units
const double re = 5000;  // Re =
const double nu=(u0*double(inlet_diameter))/re;
const double cs=0.1;
//const double nu =0.02;
const double pr=0.71;
const double ra=100000.0;
const double alpha=nu/pr;
//const double gbeta=ra*nu*alpha/(double(M*M*M));
//const double uref=sqrt(gbeta*double(N)/(ra*pr));
//const double uref=sqrt(gbeta*double(M));
//const double Ma=sqrt(3.0)*uref;
 const double tw=1.;
 const double tc=.0;
// const double u0 = 0.1;//uref;


// /*
      /*
      const double inlet_dim=.101;           // Inlet size
      const double outlet_dim=.101;          // Outlet Size

      const double x_length=.914;           // Length of the geometry in x direction
      const int outlet_diameter=ROUND((outlet_dim/x_length)*L);  //Nondimensionalization of outlet size
      const int inlet_diameter=ROUND((inlet_dim/x_length)*L);  //Nondimensionalization of outlet size

    */


//Partition 

		const double pw=4;  
		const int x1=((L+0)/2)-pw;                               //  starting coordinate of parttion along x direction
      		const int x2=((L+0)/2);                                  //  ending coordinate of parttion along x direction

    //--When inlet is in xy-plane --//
      const int inlet_size= inlet_diameter/2.0;
      // Here z=0 for all coordinate meaning the south/front wall 
      const int x3=(L+0)/4-inlet_size;             //  starting  coordinate of inlet along x direction
      const int y3=(M+0)/2-inlet_size;             //  starting  coordinate of inlet along y direction
      const int x4=(L+0)/4+inlet_size;             //  starting  coordinate of inlet along x direction
      const int y4=(M+0)/2+inlet_size;             //  starting  coordinate of inlet along y direction
       
          //--When inlet is in xy-plane --//
      const int outlet_size= outlet_diameter/2.0;
      // Here z=N for all coordinate meaning the north/back wall 
      const int x5=L/2+L/4-outlet_size;             //  starting  coordinate of outlet along x direction
      const int y5=(M+0)/2-outlet_size;             //  starting  coordinate of outlet along y direction
      const int x6=L/2+L/4+outlet_size;             //  starting  coordinate of outlet along x direction
      const int y6=(M+0)/2+outlet_size;             //  starting  coordinate of outlet along y direction

     //  */

      const int z2=N-1;                             //  starting coordinate of parttion along z direction
     /* const int z2=N/2;                           //  ending coordinate of parttion along z direction
      const int y4=0;
      const int y5=M-1; 
	*/

	const int y7=M/2 ;
        






#include "initial.h"
#include "kernel.h"
int main(int argc, char *argv[])
{
   //double *ux, *uy, rho for CPU;
cudaSetDevice(1);
    double * ux = (double *) malloc (L*M*N*sizeof(double));
    double * uy = (double *) malloc (L*M*N*sizeof(double));
    double * uz = (double *) malloc (L*M*N*sizeof(double));
    double * rho = (double *) malloc (L*M*N*sizeof(double)); 
    double * mean_u = (double *) malloc (L*M*N*sizeof(double));
    double * mean_v = (double *) malloc (L*M*N*sizeof(double));
    double * mean_w = (double *) malloc (L*M*N*sizeof(double));

   //--Random Number Genaration--//
	 double * rands= (double *) malloc (L*M*N*sizeof(double));
        
	srand ( time ( NULL));
        int i;
        int s=L*M*N;
        for (i=0;i<s;i++){
        rands[i]=(double)rand()/RAND_MAX*2.0-1.0;
         }

     //--Random Number BlockEND--// 

    double * ups_h = (double *) malloc (L*M*N*sizeof(double));
    double * vps_h = (double *) malloc (L*M*N*sizeof(double));
    double * wps_h = (double *) malloc (L*M*N*sizeof(double));
    double * uvp_h = (double *) malloc (L*M*N*sizeof(double));
    double * vwp_h = (double *) malloc (L*M*N*sizeof(double));
    double * uwp_h = (double *) malloc (L*M*N*sizeof(double));
    double * t_h = (double *) malloc (L*M*N*sizeof(double));
    double * mean_t = (double *) malloc (L*M*N*sizeof(double));
    double * tps_h = (double *) malloc (L*M*N*sizeof(double));

    // allocate memory on the GPU for probability distribution function f_i
    double *d_f0, *d_f1, *d_f2, *d_f3, *d_f4, *d_f5, *d_f6, *d_f7,*d_f8,*d_f9,*d_f10, *d_f11, *d_f12, *d_f13, *d_f14, *d_f15, *d_f16, *d_f17,*d_f18;
    cudaMalloc((void **)&d_f0,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f1,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f2,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f3,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f4,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f5,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f6,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f7,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f8,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f9,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f10,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f11,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f12,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f13,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f14,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f15,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f16,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f17,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_f18,L*M*N*sizeof(double));

double *mf0, *mf1, *mf2, *mf3, *mf4, *mf5, *mf6, *mf7,*mf8,*mf9,*mf10, *mf11, *mf12, *mf13, *mf14, *mf15, *mf16, *mf17,*mf18;
    cudaMalloc((void **)&mf0,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf1,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf2,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf3,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf4,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf5,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf6,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf7,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf8,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf9,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf10,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf11,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf12,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf13,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf14,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf15,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf16,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf17,L*M*N*sizeof(double));
    cudaMalloc((void **)&mf18,L*M*N*sizeof(double));

   double  *d_g1, *d_g2, *d_g3, *d_g4, *d_g5,*d_g6;

        cudaMalloc((void **)&d_g1,L*M*N*sizeof(double));
        cudaMalloc((void **)&d_g2,L*M*N*sizeof(double));
        cudaMalloc((void **)&d_g3,L*M*N*sizeof(double));
        cudaMalloc((void **)&d_g4,L*M*N*sizeof(double));
        cudaMalloc((void **)&d_g5,L*M*N*sizeof(double));
        cudaMalloc((void **)&d_g6,L*M*N*sizeof(double));
double *gpost1,*gpost2,*gpost3,*gpost4,*gpost5,*gpost6;

     cudaMalloc((void **)&gpost1,L*M*N*sizeof(double));
     cudaMalloc((void **)&gpost2,L*M*N*sizeof(double));
     cudaMalloc((void **)&gpost3,L*M*N*sizeof(double));
     cudaMalloc((void **)&gpost4,L*M*N*sizeof(double));
     cudaMalloc((void **)&gpost5,L*M*N*sizeof(double));
     cudaMalloc((void **)&gpost6,L*M*N*sizeof(double));

// allocate memory on the GPU for post collision distribution function fpost_i

double *fpost0,*fpost1,*fpost2,*fpost3,*fpost4,*fpost5,*fpost6,*fpost7,*fpost8,*fpost9, *fpost10,*fpost11,*fpost12,*fpost13,*fpost14,*fpost15,*fpost16,
         *fpost17,*fpost18;
    cudaMalloc((void **)&fpost0,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost1,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost2,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost3,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost4,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost5,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost6,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost7,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost8,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost9,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost10,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost11,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost12,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost13,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost14,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost15,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost16,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost17,L*M*N*sizeof(double));
    cudaMalloc((void **)&fpost18,L*M*N*sizeof(double));

// allocate memory on the GPU for velocity, density and relaxation parameter tau

    double *d_rho, *d_ux, *d_uy,*d_uz,*omega,*d_t,*rand_d;
    cudaMalloc((void **)&d_rho,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_ux,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_uy,L*M*N*sizeof(double));
    cudaMalloc((void **)&d_uz,L*M*N*sizeof(double));
    cudaMalloc((void **)&omega,L*M*N*sizeof(double));
     cudaMalloc((void **)&d_t,L*M*N*sizeof(double));
    cudaMalloc((void **)&rand_d,L*M*N*sizeof(double));

  //  cudaMalloc((void **)&omega,L*M*N*sizeof(double));
    double *umean ,*vmean,*wmean,*ups,*vps,*wps,*uvp,*vwp,*uwp,*tps,*tmean;
    cudaMalloc((void **)&umean,L*M*N*sizeof(double));
    cudaMalloc((void **)&vmean,L*M*N*sizeof(double));
    cudaMalloc((void **)&wmean,L*M*N*sizeof(double));
    cudaMalloc((void **)&ups,L*M*N*sizeof(double));
    cudaMalloc((void **)&vps,L*M*N*sizeof(double));
    cudaMalloc((void **)&wps,L*M*N*sizeof(double));
    cudaMalloc((void **)&uvp,L*M*N*sizeof(double));
    cudaMalloc((void **)&vwp,L*M*N*sizeof(double));
    cudaMalloc((void **)&uwp,L*M*N*sizeof(double));
    cudaMalloc((void **)&tmean,L*M*N*sizeof(double));
    cudaMalloc((void **)&tps,L*M*N*sizeof(double));
    
    //Assigning value in the random variable array

     cudaMemcpy(rand_d,rands,L*M*N * sizeof(double),cudaMemcpyHostToDevice);

    // assign a 3D distribution of CUDA "threads" within each CUDA "block"
    dim3 grid_size(M,N,1);
    dim3 block_size(L,1,1);
   

initialize<<<grid_size, block_size>>>(d_rho, d_ux, d_uy,d_uz,d_t,
       d_f0, d_f1, d_f2, d_f3, d_f4, d_f5, d_f6, d_f7, d_f8,d_f9, d_f10, d_f11, d_f12, d_f13, d_f14, d_f15, d_f16, d_f17, d_f18,
       umean ,vmean,wmean,ups,vps,wps,uvp,vwp,uwp,tps,tmean,d_g1,d_g2,d_g3,d_g4,d_g5,d_g6,rand_d);
    // time integration
    int time=0;
   // int avg_step;
    while(time<TIME_STEPS) {

        time++;
       // avg_step=TIME_STEPS-time;
//std::cout<< time << "\n";
 kernel<<<grid_size, block_size>>>(d_rho, d_ux, d_uy,d_uz,d_t,
               d_f0, d_f1, d_f2, d_f3, d_f4, d_f5, d_f6, d_f7, d_f8,d_f9, d_f10, d_f11, d_f12, d_f13, d_f14, d_f15, d_f16, d_f17, d_f18,
         fpost0,fpost1,fpost2,fpost3,fpost4,fpost5,fpost6,fpost7,fpost8,fpost9,fpost10,fpost11,fpost12,fpost13,fpost14,fpost15,fpost16,fpost17,fpost18
      ,umean ,vmean,wmean,ups,vps,wps,uvp,vwp,uwp,tps,time,mf0, mf1, mf2, mf3, mf4, mf5, mf6, mf7, mf8,mf9, mf10, mf11, mf12, mf13, mf14, mf15, mf16, mf17, mf18,omega
      ,d_g1,d_g2,d_g3,d_g4,d_g5,d_g6, gpost1,gpost2, gpost3,gpost4,gpost5,gpost6,tmean,rand_d);


std::cout<<"Time step = " << time << "\n";
    }
    cudaMemcpy(t_h,d_t,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
// cudaMemcpy(ux,d_ux,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
// cudaMemcpy(uy,d_uy,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
// cudaMemcpy(uz,d_uz,L*M*N*sizeof(double),cudaMemcpyDeviceToHost); 
 //cudaMemcpy(rho,d_rho,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(mean_u,umean,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(mean_v,vmean,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(mean_w,wmean,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(mean_t,tmean,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);

cudaMemcpy(ups_h,ups,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(vps_h,vps,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(wps_h,wps,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(uvp_h,uvp,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
 cudaMemcpy(vwp_h,vwp,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(uwp_h,uwp,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(tps_h,tps,L*M*N*sizeof(double),cudaMemcpyDeviceToHost);
    
cudaThreadSynchronize();



FILE *fp,*fp1,*fp2,*fp3,*fp4,*fp5,*fp6,*fp7,*fp8,*fp9,*fp10,*fp11,*fp12,*fp13,*fp14,*fp15,*fp16,*fp17,*fp18,*fp19,*fp20,*fp21;


// Velocity profiles for different x positions ...

    fp1=fopen("Mean_vel_x_point1.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp1,"%10.7f %16.14f %16.14f %16.14f",float(j)/float(M-1),mean_u[L/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/10+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/10+L*j+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp1, "\n");
      }
    fclose(fp1);
    
 fp2=fopen("Mean_vel_x_point25.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp2," %10.7f %16.14f %16.14f %16.14f",float(j)/float(M-1),mean_u[L/4+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/4+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/4+L*j+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp2, "\n");
      }
    fclose(fp2);
 
 fp3=fopen("Mean_vel_x_point40.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp3," %10.7f %16.14f %16.14f %16.14f",float(j)/float(M-1),mean_u[(2*L)/5+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[(2*L)/5+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[(2*L)/5+L*j+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp3, "\n");
      }
    fclose(fp3);

 fp4=fopen("Mean_vel_x_point75.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp4," %10.7f %16.14f %16.14f %16.14f",float(j)/float(M-1),mean_u[(3*L)/4+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[(3*L)/4+L*j+M*L*N/2]/(u0*(avg_time)),mean_w[(3*L)/4+L*j+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp4, "\n");
      }
    fclose(fp4);


fp5=fopen("Mean_vel_x_point90.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp5," %10.7f %16.14f %16.14f %16.14f",float(j)/float(M-1),mean_u[(9*L)/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[(9*L)/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_w[(9*L)/10+L*j+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp5, "\n");
      }
    fclose(fp5);




fp6=fopen("Mean_vel_y_point125.dat","w");
    for(int i=0;i<=L-1;i++){
    fprintf(fp6," %10.7f %16.14f %16.14f %16.14f",float(i)/float(L-1),mean_u[i+L*(4*L)/5+M*L*N/2]/(u0*(avg_time)),mean_v[i+L*(4*L)/5+M*L*N/2]/(u0*(avg_time)),mean_w[i+L*(4*L)/5+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp6, "\n");
      }
    fclose(fp6);


fp7=fopen("Mean_vel_y_point25.dat","w");
    for(int i=0;i<=L-1;i++){
    fprintf(fp7," %10.7f %16.14f %16.14f %16.14f",float(i)/float(L-1),mean_u[i+L/4*L+M*L*N/2]/(u0*(avg_time)),mean_v[i+L/4*L+M*L*N/2]/(u0*(avg_time)),mean_w[i+L/4*L+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp7, "\n");
      }
    fclose(fp7);



 fp=fopen("Mean_vel_x_point1.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp,"%16.14f %16.14f %16.14f %10.7f ",mean_u[L/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/10+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/10+L*j+M*L*N/2]/(u0*(avg_time)), float(j)/float(M-1));
    fprintf(fp, "\n");
      }
    fclose(fp);
 fp=fopen("Mean_vel_x_point1.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp,"%16.14f %16.14f %16.14f %10.7f ",mean_u[L/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/10+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/10+L*j+M*L*N/2]/(u0*(avg_time)), float(j)/float(M-1));
    fprintf(fp, "\n");
      }
    fclose(fp); fp=fopen("Mean_vel_x_point1.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp,"%16.14f %16.14f %16.14f %10.7f ",mean_u[L/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/10+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/10+L*j+M*L*N/2]/(u0*(avg_time)), float(j)/float(M-1));
    fprintf(fp, "\n");
      }
    fclose(fp); fp=fopen("Mean_vel_x_point1.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp,"%16.14f %16.14f %16.14f %10.7f ",mean_u[L/10+L*j+M*L*N/2]/(u0*(avg_time)),mean_v[L/10+L*j+M*L*N/2]/(u0*(avg_time)),
						mean_w[L/10+L*j+M*L*N/2]/(u0*(avg_time)), float(j)/float(M-1));
    fprintf(fp, "\n");
      }
    fclose(fp);	
  



// end of files

    fp=fopen("Mean_UV_vector.dat","w");
    fprintf(fp, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp, "zone i=" "%5d",M);
    fprintf(fp, ", j=" "%5d",L);
    fprintf(fp,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int j=0;j<=M-1;j++){
//    fprintf(fp,"%f %f %15e %15e", float(i)/float(M-1), float(j)/float(M-1),ux[i+L*j+M*L*(N/2)]/u0,uy[i+L*j+M*L*(N/2)]/u0);
    fprintf(fp,"%5f %5f %15e %15e %15e", float(i)/float(M), float(j)/float(M),mean_u[i+L*j+M*L*(N/2)]/(u0*(avg_time)),mean_v[i+L*j+M*L*(N/2)]/(u0*(avg_time)),mean_w[i+L*j+M*L*(N/2)]/(u0*(avg_time)));
    fprintf(fp, "\n");
       }
      }
    fprintf(fp, "GEOMETRY M=GRID, FC=WHITE,F=POINT" "\n");
    fprintf(fp, "2" "\n");
    fprintf(fp, "4" "\n");
    fprintf(fp,"%f %f ",(x1/float(M/1)),0.0) ;
    fprintf(fp, "\n");
    fprintf(fp,"%f %f ",(x1/float(M/1)),(M/2)/float(M/1)) ;
    fprintf(fp, "\n");
    fprintf(fp,"%f %f ",(x2/float(M/1)),(M/2)/float(M/1)) ;
    fprintf(fp, "\n");
    fprintf(fp,"%f %f ",(x2/float(M/1)),0.0) ;
    fprintf(fp, "\n");
    fprintf(fp, "2" "\n");
    fprintf(fp,"%f %f ",(x1/float(M/1)),0.0) ;
    fprintf(fp, "\n");
    fprintf(fp,"%f %f ",(x2/float(M/1)),0.0) ;
    fprintf(fp, "\n");


    fclose(fp);


fp15=fopen("Mean_UW_vector_point50M.dat","w");
    fprintf(fp15, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp15, "zone i=" "%5d",N);
    fprintf(fp15, ", j=" "%5d",L);
    fprintf(fp15,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int k=0;k<=N-1;k++){
//    fprintf(fp,"%f %f %15e %15e", float(i)/float(M-1), float(j)/float(M-1),ux[i+L*j+M*L*(N/2)]/u0,uy[i+L*j+M*L*(N/2)]/u0);
    fprintf(fp15,"%5f %5f %15e %15e %15e", float(i)/float(N), float(k)/float(N),mean_u[i+L*(M/2)+M*L*k]/(u0*(avg_time)),mean_v[i+L*(M/2)+M*L*k]/(u0*(avg_time)),mean_w[i+L*(M/2)+M*L*k]/(u0*(avg_time)));
    fprintf(fp15, "\n");
       }
      }
      fclose(fp15);



fp16=fopen("Mean_UW_vector_point25M.dat","w");
    fprintf(fp16, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp16, "zone i=" "%5d",N);
    fprintf(fp16, ", j=" "%5d",L);
    fprintf(fp16,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int k=0;k<=N-1;k++){
//    fprintf(fp,"%f %f %15e %15e", float(i)/float(M-1), float(j)/float(M-1),ux[i+L*j+M*L*(N/2)]/u0,uy[i+L*j+M*L*(N/2)]/u0);
    fprintf(fp16,"%5f %5f %15e %15e %15e", float(i)/float(M), float(k)/float(N),mean_u[i+L*(M/4)+M*L*k]/(u0*(avg_time)),mean_v[i+L*(M/4)+M*L*k]/(u0*(avg_time)),mean_w[i+L*(M/4)+M*L*k]/(u0*(avg_time)));
    fprintf(fp16, "\n");
       }
      }
      fclose(fp16);



fp17=fopen("Mean_UW_vector_point75M.dat","w");
    fprintf(fp17, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp17, "zone i=" "%5d",N);
    fprintf(fp17, ", j=" "%5d",L);
    fprintf(fp17,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int k=0;k<=N-1;k++){
//    fprintf(fp,"%f %f %15e %15e", float(i)/float(M-1), float(j)/float(M-1),ux[i+L*j+M*L*(N/2)]/u0,uy[i+L*j+M*L*(N/2)]/u0);
    fprintf(fp17,"%5f %5f %15e %15e %15e", float(i)/float(M), float(k)/float(N),mean_u[i+L*(3*M/4)+M*L*k]/(u0*(avg_time)),mean_v[i+L*(3*M/4)+M*L*k]/(u0*(avg_time)),mean_w[i+L*(3*M/4)+M*L*k]/(u0*(avg_time)));
    fprintf(fp17, "\n");
       }
      }
      fclose(fp17);

 fp8=fopen("isotherms_str_point50.dat","w");
    fprintf(fp8, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp8, "zone i=" "%5d",M);
    fprintf(fp8, ", j=" "%5d",N);
    fprintf(fp8,  ", f=point""\n");
    for(int j=0;j<=M-1;j++){
    for(int k=0;k<=N-1;k++){
    fprintf(fp8,"%5f %5f %15e", float(j)/float(M-1), float(k)/float(N-1),mean_t[(L/2)+L*j+M*L*k]/(u0*(avg_time)));
    fprintf(fp8, "\n");
       }
      }
    fclose(fp8);


 fp18=fopen("isotherms_str_point25.dat","w");
    fprintf(fp18, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp18, "zone i=" "%5d",M);
    fprintf(fp18, ", j=" "%5d",N);
    fprintf(fp18,  ", f=point""\n");
    for(int j=0;j<=M-1;j++){
    for(int k=0;k<=N-1;k++){
    fprintf(fp18,"%5f %5f %15e", float(j)/float(M-1), float(k)/float(N-1),mean_t[(L/4)+L*j+M*L*k]/(u0*(avg_time)));
    fprintf(fp18, "\n");
       }
      }
    fclose(fp18);

 fp19=fopen("isotherms_str_point75.dat","w");
    fprintf(fp19, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp19, "zone i=" "%5d",M);
    fprintf(fp19, ", j=" "%5d",N);
    fprintf(fp19,  ", f=point""\n");
    for(int j=0;j<=M-1;j++){
    for(int k=0;k<=N-1;k++){
    fprintf(fp19,"%5f %5f %15e", float(j)/float(M-1), float(k)/float(N-1),mean_t[(3*L/4)+L*j+M*L*k]/(u0*(avg_time)));
    fprintf(fp19, "\n");
       }
      }
    fclose(fp19);





  /* fp1=fopen("Mean_WV_vector.dat","w");
    fprintf(fp1, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp1, "zone i=" "%5d",M);
    fprintf(fp1, ", j=" "%5d",N);
    fprintf(fp1,  ", f=point""\n");
    for(int k=0;k<=N-1;k++){
    for(int j=0;j<=M-1;j++){    
    fprintf(fp1,"%f %f %15e %15e", float(k)/float(M-1), float(j)/float(M-1),mean_w[L/4+L*j+M*L*k]/(u0*(avg_time)),mean_v[L/4+L*j+M*L*k]/(u0*(avg_time)));
//    fprintf(fp1,"%f %f %15e %15e", float(j)/float(N-1), float(k)/float(N-1),vps_h[L/2+L*j+M*L*k]/(u0*avg_time),uvp_h[L/2+L*j+M*L*k]/(u0*avg_time));
//    fprintf(fp,"%f %f %15e %15e", float(i)/float(N-1), float(j)/float(N-1),ux[i+L*j+M*L*(N/2)]/u0,u0);
    fprintf(fp1, "\n");
       }
      }
    fclose(fp1);
    fp101=fopen("Mean_WU_vector.dat","w");
    fprintf(fp101, "variable='x','y' ,'u','v' ""\n");
    fprintf(fp101, "zone i=" "%5d",L);
    fprintf(fp101, ", j=" "%5d",N);
    fprintf(fp101,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int k=0;k<=N-1;k++){
    fprintf(fp101,"%f %f %15e %15e", float(k)/float(M-1), float(i)/float(M-1),mean_w[i+L*M/2+M*L*k]/(u0*(avg_time)),mean_u[i+L*M/2+M*L*k]/(u0*(avg_time)));
    fprintf(fp101, "\n");
     }
    }
   fclose(fp101);

    FILE *fp2,*fp3;  
    fp2=fopen("Mean_vel_u_midx.dat","w");
    for(int j=0;j<=M-1;j++){
    fprintf(fp2,"%16.14f %10.7f ",mean_u[L/2+L*j+M*L*N/2]/(u0*(avg_time)), float(j)/float(M-1));
    fprintf(fp2, "\n");
      }
    fclose(fp2);
    
    fp3=fopen("Mean_vel_v_midy.dat","w");
    for(int i=0;i<=L-1;i++){
    fprintf(fp3,"%16.14f  %10.7f ", float(i)/float(M-1),mean_v[i+L*M/2+M*L*N/2]/(u0*(avg_time)));
    fprintf(fp3, "\n");
      }
    fclose(fp3);
*/
    FILE *fp04,*fp05;
    fp04=fopen("rms_uvw_quarter_xy.dat","w");
    for(int k=1;k<=N-1;k++){
   fprintf(fp04,"%5f %15e %15e %15e", float(k)/float(N-1),sqrt(ups_h[L/4+L*M/2+M*L*k])/(u0*(avg_time)),sqrt(vps_h[L/4+L*M/2+M*L*k])/(u0*(avg_time)),sqrt(wps_h[L/4+L*M/2+M*L*k])/(u0*(avg_time))); 
    fprintf(fp04, "\n");
      }
    fclose(fp04);
    fp05=fopen("rms_uvw_quarter_yz.dat","w");
    for(int i=1;i<=L-1;i++){
   fprintf(fp05,"%5f %15e %15e %15e", float(i)/float(N-1),sqrt(ups_h[i+L*M/2+M*L*N/4])/(u0*(avg_time)), sqrt(vps_h[i+L*M/2+M*L*N/4])/(u0*(avg_time)),sqrt(wps_h[i+L*M/2+M*L*N/4])/(u0*(avg_time)));
   fprintf(fp05, "\n");
      }
    fclose(fp05);


   FILE *fp06,*fp07;
    fp06=fopen("temperature_tps_midx.dat","w");
    for(int k=1;k<=N-1;k++){
    fprintf(fp06,"%5f %15e ",float(k)/float(N-1),tps_h[L/2+L*M/2+M*L*k]/(u0*(avg_time)));
    fprintf(fp06, "\n");
      }
    fclose(fp06);


/*
    FILE *fp06,*fp07;
    fp06=fopen("Reynolds_uv_midx.dat","w");
    for(int k=1;k<=N-1;k++){
    fprintf(fp06,"%5f %15e ",float(k)/float(N-1),uwp_h[L/2+L*M/2+M*L*k]/(u0*u0*(avg_time)));
    fprintf(fp06, "\n");
      }
    fclose(fp06);
    fp07=fopen("Reynolds_uv_midy.dat","w");
    for(int i=1;i<L-1;i++){
   fprintf(fp07,"%5f %15e ", float(i)/float(N-1),uwp_h[i+L*M/2+M*L*N/2]/(u0*u0*(avg_time)));
    fprintf(fp07, "\n");
      }
    fclose(fp07);
*/

FILE *fp03,*fp002;
double snul=0.0;
double snur=0.0;
double snur1=0.0;
for(int j=0;j<=M-1;j++){
     for(int k=0;k<=N-1;k++){
         double rnul=(t_h[0+L*j+L*M*k]-t_h[1+L*j+L*M*k])*double(M);
         //double rnur=(t[(M-2)+M*j]-t[(M-1)+j*M])*double(M-1);
         float rnur=-0.5*(4.0*t_h[1+j*L+L*M*k]-3.0*t_h[0+L*j+L*M*k]-t_h[2+L*j+M*N*k])*double(M);
         //fprintf(fp002,"%d %f %f ", , rnul,rnur);
         //fprintf(fp002, "\n");
         snul=snul+rnul;
         snur=snur+rnur;
     }
}
for(int j=0;j<=M-1;j++){
    double rnul1=(t_h[0+L*j+L*M*N/2]-t_h[1+L*j+L*M*N/2])*double(M);
    //double rnur=(t[(M-2)+M*j]-t[(M-1)+j*M])*double(M-1);
    //float rnur=-0.5*(4.0*theta[(x2+1)+j*M]-3.0*theta[x2+M*j]-theta[(x2+2)+M*j])*float(N);
    //fprintf(fp002,"%d %f %f ", , rnul,rnur);
    //fprintf(fp002, "\n");
    snur1=snur1+rnul1;
    //snur=snur+rnur;
}

fp03=fopen("Avg_Nu_left.dat","w");
 //fprintf(fp0,"%f %f %f %f", ra, snul/double(x2-x1),snur/double(x2-x1),((snul)/1.0)/double(x2-1-x1));
  fprintf(fp03,"%f %f %f %f ", ra,1*(snul/double((M)*(M))), 1*(snur/double((M)*(M))),1*(snur1/double(M)));
   fprintf(fp03, "\n");
    fclose(fp03);
 FILE *fp005;
  fp005=fopen("Mean_Inst_temp.dat","w");
  fprintf(fp005, "variable='x','y' ,'u','v' ""\n");
  fprintf(fp005, "zone i=" "%5d",M);
  fprintf(fp005, ", j=" "%5d",L);
  fprintf(fp005,  ", f=point""\n");
  for(int i=0;i<=L-1;i++){
  for(int j=0;j<=M-1;j++){
  fprintf(fp005,"%8f %8f %15e %15e ", float(i)/float(M), float(j)/float(M),mean_t[i+L*j+M*L*N/2]/(avg_time),t_h[i+L*j+M*L*N/2]);
  fprintf(fp005, "\n");
  }
  }
 fclose(fp005);


 FILE *fp102;
  fp102=fopen("Mean_WV_vector_point25.dat","w");
  fprintf(fp102, "variable='x','y' ,'u','v' ""\n");
  fprintf(fp102, "zone i=" "%5d",N);
  fprintf(fp102, ", j=" "%5d",M);
  fprintf(fp102,  ", f=point""\n");
  for(int k=0;k<=N-1;k++){
  for(int j=0;j<=M-1;j++){
  fprintf(fp102,"%8f %8f %15e %15e", float(k)/float(N), float(j)/float(M),mean_v[(L/4)+L*j+M*L*k]/(u0*(avg_time)),mean_w[(L/4)+L*j+M*L*k]/(u0*(avg_time)));
  fprintf(fp102, "\n");
  }
  }
 fclose(fp102);
 
  fp14=fopen("Mean_WV_vector_point50.dat","w");
  fprintf(fp14, "variable='x','y' ,'u','v' ""\n");
  fprintf(fp14, "zone i=" "%5d",N);
  fprintf(fp14, ", j=" "%5d",M);
  fprintf(fp14,  ", f=point""\n");
  for(int k=0;k<=N-1;k++){
  for(int j=0;j<=M-1;j++){
  fprintf(fp14,"%8f %8f %15e %15e", float(k)/float(N), float(j)/float(M), mean_v[(L/2)+L*j+M*L*k]/(u0*(avg_time)),mean_w[(L/2)+L*j+M*L*k]/(u0*(avg_time)));
  fprintf(fp14, "\n");
  }
  }
 fclose(fp14);


  fp20=fopen("Mean_WV_vector_point75.dat","w");
  fprintf(fp20, "variable='x','y' ,'u','v' ""\n");
  fprintf(fp20, "zone i=" "%5d",N);
  fprintf(fp20, ", j=" "%5d",M);
  fprintf(fp20,  ", f=point""\n");
  for(int k=0;k<=N-1;k++){
  for(int j=0;j<=M-1;j++){
  fprintf(fp20,"%8f %8f %15e %15e", float(k)/float(N), float(j)/float(M), mean_v[(3*L/4)+L*j+M*L*k]/(u0*(avg_time)),mean_w[(3*L/4)+L*j+M*L*k]/(u0*(avg_time)));
  fprintf(fp20, "\n");
  }
  }
 fclose(fp20);


fp21=fopen("3DContour_temp.dat","w");
    fprintf(fp21, "variable='x','z' ,'y','T' " "\n");
    fprintf(fp21, "zone i=" "%5d",M);
    fprintf(fp21, ", k=" "%5d",N);
    fprintf(fp21, ", j=" "%5d",L);
    fprintf(fp21,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int j=0;j<=M-1;j++){
     for(int k=0;k<=N-1;k++){
      fprintf(fp21,"%f %f %f %15e", float(i)/float(M),float(j)/float(M), float(k)/float(M),t_h[i+L*j+M*L*k]);
 
 
	fprintf(fp21, "\n");
        }
    	}
	}


        fprintf(fp21,"GEOMETRY  M=GRID, FC=BLUE, T=LINE3D, F=POINT" "\n");
        fprintf(fp21,"5" "\n");
	
	 fprintf(fp21,"5" "\n");
         fprintf(fp21,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),0.0,0.0) ;
         fprintf(fp21, "\n");
         fprintf(fp21,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),0.0) ;
         fprintf(fp21, "\n");
         fprintf(fp21,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),0.0) ;
         fprintf(fp21, "\n");
	fprintf(fp21,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp21, "\n");


        fprintf(fp21,"4" "\n");
         fprintf(fp21,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp21, "\n");
         fprintf(fp21,"%f %f %f ",(x1/float(M/1)),0.0,z2/float(M/1)) ;
         fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
         fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),0.0) ;
        fprintf(fp21, "\n");
        
         fprintf(fp21,"4" "\n");


        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),0.0) ;
        fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),0.0,z2/float(M/1)) ;
        fprintf(fp21, "\n");

        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),0.0,0.0) ;
        fprintf(fp21, "\n");

        fprintf(fp21,"2" "\n");
        fprintf(fp21,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp21, "\n");

        fprintf(fp21,"2" "\n");
        fprintf(fp21,"%f %f %f ",(x1/float(M/1)),0.0,z2/float(N/1)) ;
        fprintf(fp21, "\n");
        fprintf(fp21,"%f %f %f ",(x2/float(M/1)),0.0,z2/float(N/1)) ;
        fprintf(fp21, "\n");
fclose(fp21);




FILE *fp22;

fp22=fopen("3DContour_velocity.dat","w");
    fprintf(fp22, "variable='x','z' ,'y','V' " "\n");
    fprintf(fp22, "zone i=" "%5d",M);
    fprintf(fp22, ", k=" "%5d",N);
    fprintf(fp22, ", j=" "%5d",L);
    fprintf(fp22,  ", f=point""\n");
    for(int i=0;i<=L-1;i++){
    for(int j=0;j<=M-1;j++){
     for(int k=0;k<=N-1;k++){
      fprintf(fp22,"%f %f %f %15e", float(i)/float(M),float(j)/float(M), float(k)/float(M),mean_w[i+L*j+M*L*k]/(u0*(avg_time)));
        fprintf(fp22, "\n");
        }
        }
        }


        fprintf(fp22,"GEOMETRY  M=GRID, FC=BLUE, T=LINE3D, F=POINT" "\n");
       fprintf(fp22,"5" "\n");

         fprintf(fp22,"5" "\n");
         fprintf(fp22,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),0.0,0.0) ;
         fprintf(fp22, "\n");
         fprintf(fp22,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),0.0) ;
         fprintf(fp22, "\n");
         fprintf(fp22,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),0.0) ;
         fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp22, "\n");

 

         fprintf(fp22,"4" "\n");
         fprintf(fp22,"%f %f %f ",(x1/float(M/1)),0.0,0.0) ;
         fprintf(fp22, "\n");
         fprintf(fp22,"%f %f %f ",(x1/float(M/1)),0.0,z2/float(M/1)) ;
         fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
         fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),0.0) ;
        fprintf(fp22, "\n");

        fprintf(fp22,"4" "\n");
	fprintf(fp22,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),0.0) ;
        fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),0.0,z2/float(M/1)) ;
        fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),0.0,0.0) ;

	fprintf(fp22,"2" "\n");
        fprintf(fp22,"%f %f %f ",(x1/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),y7/float(M/1),z2/float(M/1)) ;
        fprintf(fp22, "\n");

        fprintf(fp22,"2" "\n");
        fprintf(fp22,"%f %f %f ",(x1/float(M/1)),0.0,z2/float(N/1)) ;
        fprintf(fp22, "\n");
        fprintf(fp22,"%f %f %f ",(x2/float(M/1)),0.0,z2/float(N/1)) ;
        fprintf(fp22, "\n");




fclose(fp22);

    return 0;
}
