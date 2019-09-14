__global__ void kernel(double *d_rho, double *d_ux, double *d_uy, double *d_uz,double *d_t,
                          double *d_f0,double *d_f1,double *d_f2,double *d_f3,double *d_f4,double *d_f5,double *d_f6,double *d_f7,double *d_f8,
                          double *d_f9,double *d_f10,double *d_f11,double *d_f12,double *d_f13,double *d_f14,double *d_f15,double *d_f16,double *d_f17,double *d_f18,
                          double *fpost0,double *fpost1,double *fpost2,double *fpost3,double *fpost4,double *fpost5,double *fpost6,double *fpost7,double *fpost8,
                          double *fpost9,double *fpost10,double *fpost11,double *fpost12,double *fpost13,double *fpost14,double *fpost15,double *fpost16,
                          double *fpost17, double *fpost18, double *umean,double *vmean,double *wmean,double *ups,double *vps,double *wps
                          ,double *uvp,double *vwp,double *uwp, double *tps, int time, double *mf0,double *mf1,double *mf2,double *mf3,double *mf4,double *mf5,double *mf6,double *mf7,double *mf8,
   double *mf9,double *mf10,double *mf11,double *mf12,double *mf13,double *mf14,double *mf15,double *mf16,double *mf17,double *mf18,double *omega,double *d_g1,double *d_g2,double *d_g3,double *d_g4,double *d_g5,double *d_g6,
                             double *gpost1,double *gpost2, double *gpost3,double *gpost4,double *gpost5,double *gpost6,double *tmean,double *random)


{
    // calculate fluid viscosity based on the Reynolds number

    // calculate relaxation time tau
//   double tau =  0.5 + 3.0 * nu;
 //  double omega1=1.0/tau;
// double omega_temp=omega1;

    // compute the "i" and "j" location and the "dir"
    // handled by this thread

     int i = threadIdx.x;
     int j = blockIdx.x;
     int k = blockIdx.y;
         //if((i>=1) && (i<L-1)&&(k>=1)&&(k<N-1)) {

         //         d_ux[i+L*(M-1)+M*L*k] = u0;
        // }

// if((i>0) && (i<L-1)&&(j>0)&&(j<M-1) &&(k>0) && (k<N-1)) {
     int ixyz = i+L*j+M*L*k;
/*
  double ts=d_ux[ixyz]*d_ux[ixyz]+d_uy[ixyz]*d_uy[ixyz]+d_uz[ixyz]*d_uz[ixyz];
        //double t0=0;
        double t1=d_ux[ixyz];
        double t2=-d_ux[ixyz];
        double t3=d_uy[ixyz];
        double t4=-d_uy[ixyz];
        double t5=d_uz[ixyz];
        double t6=-d_uz[ixyz];
        double t7=d_ux[ixyz]+d_uy[ixyz];
        double t8=-d_ux[ixyz]+d_uy[ixyz];
        double t9=d_ux[ixyz]-d_uy[ixyz];
        double t10=-d_ux[ixyz]-d_uy[ixyz];
        double t11=d_ux[ixyz]+d_uz[ixyz];
        double t12=-d_ux[ixyz]+d_uz[ixyz];
        double t13=d_ux[ixyz]-d_uz[ixyz];
        double t14=-d_ux[ixyz]-d_uz[ixyz];
        double t15=d_uy[ixyz]+d_uz[ixyz];
        double t16=-d_uy[ixyz]+d_uz[ixyz];
        double t17=d_uy[ixyz]-d_uz[ixyz];
        double t18=-d_uy[ixyz]-d_uz[ixyz];

       double  feq0=d_rho[ixyz]*(1./3.)*(1.0-1.5*ts);
       double  feq1=d_rho[ixyz]*(1./18.)*(1.0+3.0*t1+4.5*t1*t1-1.5*ts);
       double  feq2=d_rho[ixyz]*(1./18.)*(1.0+3.0*t2+4.5*t2*t2-1.5*ts);
       double  feq3=d_rho[ixyz]*(1./18.)*(1.0+3.0*t3+4.5*t3*t3-1.5*ts);
       double  feq4=d_rho[ixyz]*(1./18.)*(1.0+3.0*t4+4.5*t4*t4-1.5*ts);
       double  feq5=d_rho[ixyz]*(1./18.)*(1.0+3.0*t5+4.5*t5*t5-1.5*ts);
       double  feq6=d_rho[ixyz]*(1./18.)*(1.0+3.0*t6+4.5*t6*t6-1.5*ts);
       double  feq7=d_rho[ixyz]*(1./36.)*(1.0+3.0*t7+4.5*t7*t7-1.5*ts);
       double  feq8=d_rho[ixyz]*(1./36.)*(1.0+3.0*t8+4.5*t8*t8-1.5*ts);
       double  feq9=d_rho[ixyz]*(1./36.)*(1.0+3.0*t9+4.5*t9*t9-1.5*ts);
       double  feq10=d_rho[ixyz]*(1./36.)*(1.0+3.0*t10+4.5*t10*t10-1.5*ts);
       double  feq11=d_rho[ixyz]*(1./36.)*(1.0+3.0*t11+4.5*t11*t11-1.5*ts);
       double  feq12=d_rho[ixyz]*(1./36.)*(1.0+3.0*t12+4.5*t12*t12-1.5*ts);
       double  feq13=d_rho[ixyz]*(1./36.)*(1.0+3.0*t13+4.5*t13*t13-1.5*ts);
       double  feq14=d_rho[ixyz]*(1./36.)*(1.0+3.0*t14+4.5*t14*t14-1.5*ts);
       double  feq15=d_rho[ixyz]*(1./36.)*(1.0+3.0*t15+4.5*t15*t15-1.5*ts);
       double  feq16=d_rho[ixyz]*(1./36.)*(1.0+3.0*t16+4.5*t16*t16-1.5*ts);
       double  feq17=d_rho[ixyz]*(1./36.)*(1.0+3.0*t17+4.5*t17*t17-1.5*ts);
       double  feq18=d_rho[ixyz]*(1./36.)*(1.0+3.0*t18+4.5*t18*t18-1.5*ts);
// SGS for LES
        double den = d_rho[ixyz];
   double f1diff  =  d_f1[ixyz] -  feq1;
   double f2diff  =  d_f2[ixyz] -  feq2;
   double f3diff  =  d_f3[ixyz] -  feq3;
   double f4diff  =  d_f4[ixyz] -  feq4;
   double f5diff  =  d_f5[ixyz] -  feq5;
   double f6diff  =  d_f6[ixyz] -  feq6;
   double f7diff  =  d_f7[ixyz] -  feq7;
   double f8diff  =  d_f8[ixyz] -  feq8;
   double f9diff  =  d_f9[ixyz] -  feq9;
   double f10diff = d_f10[ixyz] -  feq10;
   double f11diff = d_f11[ixyz] -  feq11;
   double f12diff = d_f12[ixyz] -  feq12;
   double f13diff = d_f13[ixyz] -  feq13;
   double f14diff = d_f14[ixyz] -  feq14;
   double f15diff = d_f15[ixyz] -  feq15;
   double f16diff = d_f16[ixyz] -  feq16;
   double f17diff = d_f17[ixyz] -  feq17;
   double f18diff = d_f18[ixyz] -  feq18;

    // non equilibrium stress-tensor for velocity
    double Pi_x_x = f1diff + f2diff + f7diff + f8diff + f9diff + f10diff + f11diff + f12diff + f13diff + f14diff;
    double Pi_x_y = f7diff - f8diff - f9diff + f10diff;
    double Pi_x_z = f11diff - f12diff - f13diff + f14diff;
    double Pi_y_y = f3diff + f4diff + f7diff + f8diff + f9diff + f10diff + f15diff + f16diff + f17diff + f18diff;
    double Pi_y_z = f15diff - f16diff - f17diff + f18diff;
    double Pi_z_z = f5diff + f6diff + f11diff + f12diff + f13diff + f14diff + f15diff + f16diff + f17diff + f18diff;

 //variance

    double dxx=-(3.0/2.0)*omega[ixyz]*Pi_x_x/den;
    double dxy=-(3.0/2.0)*omega[ixyz]*Pi_x_y/den;
    double dxz=-(3.0/2.0)*omega[ixyz]*Pi_x_z/den;
    double dyy=-(3.0/2.0)*omega[ixyz]*Pi_y_y/den;
    double dyz=-(3.0/2.0)*omega[ixyz]*Pi_y_z/den;
    double dzz=-(3.0/2.0)*omega[ixyz]*Pi_z_z/den;

    double pai = sqrt(2.0*dxx*dxx + 4.*dxy*dxy+4.*dxz*dxz+4.*dyz*dyz + 2.0*dyy*dyy+2.0*dzz*dzz);
    double temp=sqrt(nu*nu+18.0*cs*cs*pai)-nu;

    double nu_sgs=temp/6.0;
    //omega[ixyz]=1.0/(3.*(nu+nu_sgs)+0.5);
      omega[ixyz]=1.0/(3.*(nu+nu_sgs)+0.5);
     double omegat=1.0/(3.*(alpha+nu_sgs/0.4)+0.5);

*/
//if((i>0) && (i<=L-1) && (j>0) && (j<=M-1)&& (k>0) && (k<=N-1)) {
mf0[ixyz]=d_f0[ixyz]+d_f1[ixyz]+d_f2[ixyz]+d_f3[ixyz]+d_f4[ixyz]+d_f5[ixyz]+d_f6[ixyz]+d_f7[ixyz]+d_f8[ixyz]
                                +d_f9[ixyz]+d_f10[ixyz]+d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]
                                +d_f17[ixyz]+d_f18[ixyz];

mf1[ixyz]=-30.0*d_f0[ixyz]-11.0*(d_f1[ixyz]+d_f2[ixyz]+d_f3[ixyz]+d_f4[ixyz]+d_f5[ixyz]+d_f6[ixyz])+8.0*(d_f7[ixyz]+d_f8[ixyz]
                                +d_f9[ixyz]+d_f10[ixyz]+d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]
                                +d_f17[ixyz]+d_f18[ixyz]);

mf2[ixyz]=12.0*d_f0[ixyz]-4.0*(d_f1[ixyz]+d_f2[ixyz]+d_f3[ixyz]+d_f4[ixyz]+d_f5[ixyz]+d_f6[ixyz])+d_f7[ixyz]+d_f8[ixyz]
                                +d_f9[ixyz]+d_f10[ixyz]+d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]
                                +d_f17[ixyz]+d_f18[ixyz];

mf3[ixyz]=d_f1[ixyz]-d_f2[ixyz]+d_f7[ixyz]-d_f8[ixyz] +d_f9[ixyz]-d_f10[ixyz]+d_f11[ixyz]-d_f12[ixyz]+d_f13[ixyz]-d_f14[ixyz];

mf4[ixyz]=-4.0*d_f1[ixyz]+4.0*d_f2[ixyz]+d_f7[ixyz]-d_f8[ixyz] +d_f9[ixyz]-d_f10[ixyz]+d_f11[ixyz]-d_f12[ixyz]+d_f13[ixyz]-d_f14[ixyz];

mf5[ixyz]=d_f3[ixyz]-d_f4[ixyz]+d_f7[ixyz]+d_f8[ixyz]-d_f9[ixyz]-d_f10[ixyz]+d_f15[ixyz]-d_f16[ixyz]+d_f17[ixyz]-d_f18[ixyz];

mf6[ixyz]=-4.0*d_f3[ixyz]+4.0*d_f4[ixyz]+d_f7[ixyz]+d_f8[ixyz]-d_f9[ixyz]-d_f10[ixyz]+d_f15[ixyz]-d_f16[ixyz]+d_f17[ixyz]-d_f18[ixyz];

mf7[ixyz]=d_f5[ixyz]-d_f6[ixyz]+d_f11[ixyz]+d_f12[ixyz]-d_f13[ixyz]-d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]-d_f17[ixyz]-d_f18[ixyz];

mf8[ixyz]=-4.0*d_f5[ixyz]+4.0*d_f6[ixyz]+d_f11[ixyz]+d_f12[ixyz]-d_f13[ixyz]-d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]-d_f17[ixyz]-d_f18[ixyz];

mf9[ixyz]=2.0*(d_f1[ixyz]+d_f2[ixyz])-(d_f3[ixyz]+d_f4[ixyz]+d_f5[ixyz]+d_f6[ixyz])+d_f7[ixyz]+d_f8[ixyz]
         +d_f9[ixyz]+d_f10[ixyz]+d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]-2.0*(d_f15[ixyz]+d_f16[ixyz]
                               +d_f17[ixyz]+d_f18[ixyz]);

mf10[ixyz]=-4.0*(d_f1[ixyz]+d_f2[ixyz])+2*(d_f3[ixyz]+d_f4[ixyz]+d_f5[ixyz]+d_f6[ixyz])+d_f7[ixyz]+d_f8[ixyz]
         +d_f9[ixyz]+d_f10[ixyz]+d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]-2.0*(d_f15[ixyz]+d_f16[ixyz]
                               +d_f17[ixyz]+d_f18[ixyz]);

mf11[ixyz]=d_f3[ixyz]+d_f4[ixyz]-(d_f5[ixyz]+d_f6[ixyz])+d_f7[ixyz]+d_f8[ixyz]
                                +d_f9[ixyz]+d_f10[ixyz]-(d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]);
mf12[ixyz]=-2.0*(d_f3[ixyz]+d_f4[ixyz])+2.0*(d_f5[ixyz]+d_f6[ixyz])+d_f7[ixyz]+d_f8[ixyz]
                                +d_f9[ixyz]+d_f10[ixyz]-(d_f11[ixyz]+d_f12[ixyz]+d_f13[ixyz]+d_f14[ixyz]);
mf13[ixyz]=d_f7[ixyz]-d_f8[ixyz]-d_f9[ixyz]+d_f10[ixyz];
mf14[ixyz]=d_f15[ixyz]-d_f16[ixyz]-d_f17[ixyz]+d_f18[ixyz];
mf15[ixyz]=d_f11[ixyz]-d_f12[ixyz]-d_f13[ixyz]+d_f14[ixyz];
mf16[ixyz]=d_f7[ixyz]-d_f8[ixyz]+d_f9[ixyz]-d_f10[ixyz]-d_f11[ixyz]+d_f12[ixyz]-d_f13[ixyz]+d_f14[ixyz];
mf17[ixyz]=-d_f7[ixyz]-d_f8[ixyz]+d_f9[ixyz]+d_f10[ixyz]+d_f15[ixyz]-d_f16[ixyz]+d_f17[ixyz]-d_f18[ixyz];
mf18[ixyz]=d_f11[ixyz]+d_f12[ixyz]-d_f13[ixyz]-d_f14[ixyz]-d_f15[ixyz]-d_f16[ixyz]+d_f17[ixyz]+d_f18[ixyz];

        double jx=d_ux[ixyz]*d_rho[ixyz];
        double jy=d_uy[ixyz]*d_rho[ixyz];//+force;
        double jz=d_uz[ixyz]*d_rho[ixyz];
        double tref=0.5;//(tw+tc)/2.0;
        double force=0.0;//gbeta*(d_t[ixyz]-tref);

       double mfeq0=d_rho[ixyz];
       double mfeq1=-11.0*d_rho[ixyz]+19.0*(jx*jx+jy*jy+jz*jz);
       double mfeq2=3.0*d_rho[ixyz]-(11.0/2.0)*(jx*jx+jy*jy+jz*jz);
       double mfeq3=jx;
       double mfeq4=-(2.0/3.0)*jx;
       double mfeq5=jy;
       double mfeq6=-(2.0/3.0)*jy;
       double mfeq7=jz;
       double mfeq8=-(2.0/3.0)*jz;
       double mfeq9=(2.0*jx*jx-jy*jy-jz*jz);
       double mfeq10=-0.5*(2.0*jx*jx-jy*jy-jz*jz);
       double mfeq11=(jy*jy-jz*jz);
       double mfeq12=-0.5*(jy*jy-jz*jz);
       double mfeq13=jx*jy;
       double mfeq14=jy*jz;
       double mfeq15=jz*jx;
       double mfeq16=0.0;
       double mfeq17=0.0;
       double mfeq18=0.0;
       double fx=0.0;
       double fy=force;
       double fz=0.0;
       double s0=0.0;
       double s1=38.0*(d_ux[ixyz]*fx+d_uy[ixyz]*fy+d_uz[ixyz]*fz);
       double s2=-11.0*(d_ux[ixyz]*fx+d_uy[ixyz]*fy+d_uz[ixyz]*fz);
       double s3=fx;
       double s4=-(2.0/3.0)*fx;
       double s5=fy;
       double s6=-(2.0/3.0)*fy;
       double s7=fz;
       double s8=-(2.0/3.0)*fz;
       double s9=2.0*(2.0*d_ux[ixyz]*fx-d_uy[ixyz]*fy-d_uz[ixyz]*fz);
       double s10=-(2.0*d_ux[ixyz]*fx-d_uy[ixyz]*fy-d_uz[ixyz]*fz);
       double s11=2.0*(d_uy[ixyz]*fy-d_uz[ixyz]*fz);
       double s12=-(d_uy[ixyz]*fy-d_uz[ixyz]*fz);
       double s13=(d_ux[ixyz]*fy+d_uy[ixyz]*fx);
       double s14=(d_uy[ixyz]*fz+d_uz[ixyz]*fy);
       double s15=(d_ux[ixyz]*fz+d_uz[ixyz]*fx);
       double s16=0.0;
       double s17=0.0;
       double s18=0.0;
// for LEST in MRT

 double den = d_rho[ixyz];
double h1=mf1[ixyz]-mfeq1+0.5*s1;
double h9=mf9[ixyz]-mfeq9+0.5*s9;
double h11=mf11[ixyz]-mfeq11+0.5*s11;
double h13=mf13[ixyz]-mfeq13+0.5*s13;
double h14=mf14[ixyz]-mfeq14+0.5*s14;
double h15=mf15[ixyz]-mfeq15+0.5*s15;

    double dxx=-(1.0/38.0)*(1.19*h1+19.0*omega[ixyz]*h9)/den;
    double dyy=-(1.0/76.0)*(2*1.19*h1-19.0*(omega[ixyz]*h9-3.0*s11*h11))/den;
    double dzz=-(1.0/76.0)*(2*1.19*h1-19.0*(omega[ixyz]*h9+3.0*s11*h11))/den;
    double dxy=-(3.0/2.0)*omega[ixyz]*h13/den;
    double dxz=-(3.0/2.0)*omega[ixyz]*h15/den;
    double dyz=-(3.0/2.0)*omega[ixyz]*h14/den;

    double pai = sqrt(2.0*dxx*dxx + 4.*dxy*dxy+4.*dxz*dxz+4.*dyz*dyz + 2.0*dyy*dyy+2.0*dzz*dzz);
    double temp=sqrt(nu*nu+18.0*cs*cs*pai)-nu;

    double nu_sgs=temp/6.0;
    //omega[ixyz]=1.0/(3.*(nu+nu_sgs)+0.5);
     omega[ixyz]=1.0/(3.*(nu+nu_sgs)+0.5);
     double omegat=1.0/(3.*(alpha+nu_sgs/0.3)+0.5);
     //double tref=0.5;//(tw+tc)/2.0;
     //double force=0.0;//gbeta*(d_t[ixyz]-tref);


mf0[ixyz]=mf0[ixyz]- 1.0 *(mf0[ixyz]-mfeq0)+d_rho[ixyz]*s0*0.5;
mf1[ixyz]=mf1[ixyz]- 1.19*(mf1[ixyz]-mfeq1)+d_rho[ixyz]*s1*(1.0-1.19/2.0);
mf2[ixyz]=mf2[ixyz]- 1.4 *(mf2[ixyz]-mfeq2)+d_rho[ixyz]*s2*(1.0-1.4/2.0);
mf3[ixyz]=mf3[ixyz]- 1.0 *(mf3[ixyz]-mfeq3)+d_rho[ixyz]*s3*0.5;
mf4[ixyz]=mf4[ixyz]- 1.2 *(mf4[ixyz]-mfeq4)+d_rho[ixyz]*s4*(1.0-1.2/2.0);
mf5[ixyz]=mf5[ixyz]- 1.0 *(mf5[ixyz]-mfeq5)+d_rho[ixyz]*s5*0.5;
mf6[ixyz]=mf6[ixyz]- 1.2 *(mf6[ixyz]-mfeq6)+d_rho[ixyz]*s6*(1.0-1.2/2.0);
mf7[ixyz]=mf7[ixyz]- 1.0 *(mf7[ixyz]-mfeq7)+d_rho[ixyz]*s7*0.5;
mf8[ixyz]=mf8[ixyz]- 1.2 *(mf8[ixyz]-mfeq8)+d_rho[ixyz]*s8*(1.0-1.2/2.0);
mf9[ixyz]=mf9[ixyz]- omega[ixyz] *(mf9[ixyz]-mfeq9)+d_rho[ixyz]*s9*(1.0-omega[ixyz]/2.0);
mf10[ixyz]=mf10[ixyz]- 1.4 *(mf10[ixyz]-mfeq10)+d_rho[ixyz]*s10*(1.0-1.4/2.0);
mf11[ixyz]=mf11[ixyz]- omega[ixyz] *(mf11[ixyz]-mfeq11)+d_rho[ixyz]*s11*(1.0-omega[ixyz]/2.0);
mf12[ixyz]=mf12[ixyz]- 1.4 *(mf12[ixyz]-mfeq12)+d_rho[ixyz]*s12*(1.0-1.4/2.0);
mf13[ixyz]=mf13[ixyz]- omega[ixyz] *(mf13[ixyz]-mfeq13)+d_rho[ixyz]*s13*(1.0-omega[ixyz]/2.0);
mf14[ixyz]=mf14[ixyz]- omega[ixyz] *(mf14[ixyz]-mfeq14)+d_rho[ixyz]*s14*(1.0-omega[ixyz]/2.0);
mf15[ixyz]=mf15[ixyz]- omega[ixyz] *(mf15[ixyz]-mfeq15)+d_rho[ixyz]*s15*(1.0-omega[ixyz]/2.0);
mf16[ixyz]=mf16[ixyz]- 1.98 *(mf16[ixyz]-mfeq16)+d_rho[ixyz]*s16;
mf17[ixyz]=mf17[ixyz]- 1.98 *(mf17[ixyz]-mfeq17)+d_rho[ixyz]*s17;
mf18[ixyz]=mf18[ixyz]- 1.98 *(mf18[ixyz]-mfeq18)+d_rho[ixyz]*s18;


mf0[ixyz]=mf0[ixyz]/19.0;
mf1[ixyz]=mf1[ixyz]/2394.0;
mf2[ixyz]=mf2[ixyz]/252.0;
mf3[ixyz]=mf3[ixyz]/10.0;
mf4[ixyz]=mf4[ixyz]/40.0;
mf5[ixyz]=mf5[ixyz]/10.0;
mf6[ixyz]=mf6[ixyz]/40.0;
mf7[ixyz]=mf7[ixyz]/10.0;
mf8[ixyz]=mf8[ixyz]/40.0;
mf9[ixyz]=mf9[ixyz]/36.0;
mf10[ixyz]=mf10[ixyz]/72.0;
mf11[ixyz]=mf11[ixyz]/12.0;
mf12[ixyz]=mf12[ixyz]/24.0;
mf13[ixyz]=mf13[ixyz]/4.0;
mf14[ixyz]=mf14[ixyz]/4.0;
mf15[ixyz]=mf15[ixyz]/4.0;
mf16[ixyz]=mf16[ixyz]/8.0;
mf17[ixyz]=mf17[ixyz]/8.0;
mf18[ixyz]=mf18[ixyz]/8.0;

fpost0[ixyz]=mf0[ixyz]-30.0*mf1[ixyz]+12.0*mf2[ixyz];

fpost1[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]+mf3[ixyz]-4.0*mf4[ixyz]+2.0*mf9[ixyz]-4.0*mf10[ixyz];

fpost2[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]-mf3[ixyz]+4.0*mf4[ixyz]+2.0*mf9[ixyz]-4.0*mf10[ixyz];

fpost3[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]+mf5[ixyz]-4.0*mf6[ixyz]-mf9[ixyz]+2.0*mf10[ixyz]+mf11[ixyz]-2.0*mf12[ixyz];

fpost4[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]-mf5[ixyz]+4.0*mf6[ixyz]-mf9[ixyz]+2.0*mf10[ixyz]+mf11[ixyz]-2.0*mf12[ixyz];

fpost5[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]+mf7[ixyz]-4.0*mf8[ixyz]-mf9[ixyz]+2.0*mf10[ixyz]-mf11[ixyz]+2.0*mf12[ixyz];

fpost6[ixyz]=mf0[ixyz]-11.0*mf1[ixyz]-4.0*mf2[ixyz]-mf7[ixyz]+4.0*mf8[ixyz]-mf9[ixyz]+2.0*mf10[ixyz]-mf11[ixyz]+2.0*mf12[ixyz];

fpost7[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf3[ixyz]+mf4[ixyz]+mf5[ixyz]+mf6[ixyz]+mf9[ixyz]
                      +mf10[ixyz]+mf11[ixyz]+mf12[ixyz]+mf13[ixyz]+mf16[ixyz]-mf17[ixyz];

fpost8[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf3[ixyz]-mf4[ixyz]+mf5[ixyz]+mf6[ixyz]+mf9[ixyz]
                      +mf10[ixyz]+mf11[ixyz]+mf12[ixyz]-mf13[ixyz]-mf16[ixyz]-mf17[ixyz];

fpost9[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf3[ixyz]+mf4[ixyz]-mf5[ixyz]-mf6[ixyz]+mf9[ixyz]
                      +mf10[ixyz]+mf11[ixyz]+mf12[ixyz]-mf13[ixyz]+mf16[ixyz]+mf17[ixyz];

fpost10[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf3[ixyz]-mf4[ixyz]-mf5[ixyz]-mf6[ixyz]+mf9[ixyz]
                      +mf10[ixyz]+mf11[ixyz]+mf12[ixyz]+mf13[ixyz]-mf16[ixyz]+mf17[ixyz];

fpost11[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf3[ixyz]+mf4[ixyz]+mf7[ixyz]+mf8[ixyz]+mf9[ixyz]
                      +mf10[ixyz]-mf11[ixyz]-mf12[ixyz]+mf15[ixyz]-mf16[ixyz]+mf18[ixyz];


fpost12[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf3[ixyz]-mf4[ixyz]+mf7[ixyz]+mf8[ixyz]+mf9[ixyz]
                      +mf10[ixyz]-mf11[ixyz]-mf12[ixyz]-mf15[ixyz]+mf16[ixyz]+mf18[ixyz];

fpost13[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf3[ixyz]+mf4[ixyz]-mf7[ixyz]-mf8[ixyz]+mf9[ixyz]
                      +mf10[ixyz]-mf11[ixyz]-mf12[ixyz]-mf15[ixyz]-mf16[ixyz]-mf18[ixyz];


fpost14[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf3[ixyz]-mf4[ixyz]-mf7[ixyz]-mf8[ixyz]+mf9[ixyz]
                      +mf10[ixyz]-mf11[ixyz]-mf12[ixyz]+mf15[ixyz]+mf16[ixyz]-mf18[ixyz];


fpost15[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf5[ixyz]+mf6[ixyz]+mf7[ixyz]+mf8[ixyz]-2.0*mf9[ixyz]
                      -2.0*mf10[ixyz]+mf14[ixyz]+mf17[ixyz]-mf18[ixyz];


fpost16[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf5[ixyz]-mf6[ixyz]+mf7[ixyz]+mf8[ixyz]-2.0*mf9[ixyz]
                      -2.0*mf10[ixyz]-mf14[ixyz]-mf17[ixyz]-mf18[ixyz];

fpost17[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]+mf5[ixyz]+mf6[ixyz]-mf7[ixyz]-mf8[ixyz]-2.0*mf9[ixyz]
                      -2.0*mf10[ixyz]-mf14[ixyz]+mf17[ixyz]+mf18[ixyz];

fpost18[ixyz]=mf0[ixyz]+8.0*mf1[ixyz]+mf2[ixyz]-mf5[ixyz]-mf6[ixyz]-mf7[ixyz]-mf8[ixyz]-2.0*mf9[ixyz]
                      -2.0*mf10[ixyz]+mf14[ixyz]-mf17[ixyz]+mf18[ixyz];

  int ip = ((i==L-1) ? (0) : (1+i));
  int im = ((i==0) ? (L-1) : (-1+i));
  int jp = ((j==M-1) ? (0) : (1+j));
  int jm = ((j==0) ? (M-1) : (-1+j));
  int kp = ((k==N-1) ? (0) : (1+k));
  int km = ((k==0) ? (N-1) : (-1+k));

  d_f0[ixyz]  = fpost0[ ixyz ];
  d_f1[ixyz]  = fpost1[im+j*L+M*L*k];
  d_f2[ixyz]  = fpost2[ip+j*L+M*L*k ];
  d_f3[ixyz]  = fpost3[i+jm*L+M*L*k ];
  d_f4[ixyz]  = fpost4[i+jp*L+M*L*k];
  d_f5[ixyz]  = fpost5[i+j*L+M*L*km ];
  d_f6[ixyz]  = fpost6[i+j*L+M*L*kp ];
  d_f7[ixyz]  = fpost7[im+jm*L+M*L*k];
  d_f8[ixyz]  = fpost8[ip+jm*L+M*L*k ];
  d_f9[ixyz]  = fpost9[im+jp*L+M*L*k ];
  d_f10[ixyz]  = fpost10[ip+jp*L+M*L*k ];
  d_f11[ixyz]  = fpost11[im+j*L+M*L*km];
  d_f12[ixyz]  = fpost12[ip+j*L+M*L*km ];
  d_f13[ixyz]  = fpost13[im+j*L+M*L*kp ];
  d_f14[ixyz]  = fpost14[ip+j*L+M*L*kp];
  d_f15[ixyz]  = fpost15[i+jm*L+M*L*km ];
  d_f16[ixyz]  = fpost16[i+jp*L+M*L*km ];
  d_f17[ixyz]  = fpost17[i+jm*L+M*L*kp];
  d_f18[ixyz]  = fpost18[i+jp*L+M*L*kp ];
// Boundary conditions
        if((j>=0) && (j<=M-1)&& (k>=0)&& (k<=N-1)) {
 // west wall
       d_f1[0+j*L+M*L*k]=fpost2[0+j*L+M*L*k];
       d_f7[0+j*L+M*L*k]=fpost10[0+j*L+M*L*k];
       d_f9[0+j*L+M*L*k]=fpost8[0+j*L+M*L*k];
       d_f11[0+j*L+M*L*k]=fpost14[0+j*L+M*L*k];
       d_f13[0+j*L+M*L*k]=fpost12[0+j*L+M*L*k];
// east wall
       d_f2[(L-1)+j*L+M*L*k]=fpost1[(L-1)+j*L+M*L*k];
       d_f8[(L-1)+j*L+M*L*k]=fpost9[(L-1)+j*L+M*L*k];
       d_f10[(L-1)+j*L+M*L*k]=fpost7[(L-1)+j*L+M*L*k];
       d_f12[(L-1)+j*L+M*L*k]=fpost13[(L-1)+j*L+M*L*k];
       d_f14[(L-1)+j*L+M*L*k]=fpost11[(L-1)+j*L+M*L*k];

}
 if((i>=0) && (i<=L-1)&&(j>=0)&&(j<=M-1)) {
// south wall
       d_f5[i+j*M+M*L*0]=fpost6[i+j*M+M*L*0];
       d_f12[i+j*M+M*L*0]=fpost13[i+j*M+M*L*0];
       d_f16[i+j*M+M*L*0]=fpost17[i+j*M+M*L*0];
       d_f15[i+j*M+M*L*0]=fpost18[i+j*M+M*L*0];
       d_f11[i+j*M+M*L*0]=fpost14[i+j*M+M*L*0];


// North wall
       d_f6[i+j*L+M*L*(N-1)]=fpost5[i+j*L+M*L*(N-1)];
       d_f13[i+j*L+M*L*(N-1)]=fpost12[i+j*L+M*L*(N-1)];
       d_f17[i+j*L+M*L*(N-1)]=fpost16[i+j*L+M*L*(N-1)];
       d_f14[i+j*L+M*L*(N-1)]=fpost11[i+j*L+M*L*(N-1)];
       d_f18[i+j*L+M*L*(N-1)]=fpost15[i+j*L+M*L*(N-1)];
}
 //top and bottom walls
     
     if((i>=0) && (i<=L-1)&&(k>=0)&&(k<=N-1)) {
// bottom wall
       d_f3[i+0*L+M*L*k]=fpost4[i+0*L+M*L*k];
       d_f7[i+0*L+M*L*k]=fpost10[i+0*L+M*L*k];
       d_f8[i+0*L+M*L*k]=fpost9[i+0*L+M*L*k];
       d_f17[i+0*L+M*L*k]=fpost16[i+0*L+M*L*k];
       d_f15[i+0*L+M*L*k]=fpost18[i+0*L+M*L*k];
}




//-- Boundary conditions for Lid driven cavity--//


 if((i>=0) && (i<=L-1)&&(k>=0)&&(k<=N-1)) {
// top wall
       int ixn=i+L*(M-1)+M*L*k;
       // double vy =0.0;
      //  double vz = 0.0;
        double vx=0;
       // double rhon=(1./(1.+vz))*(fpost0[ixn]+fpost1[ixn]+fpost2[ixn]+fpost3[ixn]+fpost4[ixn]+fpost7[ixn]+fpost8[ixn]+fpost9[ixn]+fpost10[ixn]
        //+2.0*(fpost5[ixn]+fpost11[ixn]+fpost14[ixn]+fpost15[ixn]+fpost18[ixn]));
       d_f4[ixn]=fpost3[ixn];
       d_f10[ixn]=fpost7[ixn]-vx/6.0;
       d_f9[ixn]=fpost8[ixn]+vx/6.0;
       d_f16[ixn]=fpost17[ixn];
       d_f18[ixn]=fpost15[ixn];

}







if((i>=x3) && (i<=x4)&&(j>=y3)&&(j<=y4)) {
       
      
       int ixn=i+L*j+M*L*0;
      double vz=u0* (1+ al * random[ixn]); 
      
     double rhon=(1./(1.+vz))*(fpost0[ixn]+fpost1[ixn]+fpost2[ixn]+fpost3[ixn]+fpost4[ixn]+fpost7[ixn]+fpost8[ixn]+fpost9[ixn]+fpost10[ixn]
        +2.0*(fpost5[ixn]+fpost11[ixn]+fpost12[ixn]+fpost15[ixn]+fpost16[ixn]));
      

      // double rhon=1.0;
	
      
	
       d_f5[ixn]=fpost6[ixn]+rhon*vz/3.0;
       d_f11[ixn]=fpost14[ixn]+rhon*vz/6.0;
       d_f12[ixn]=fpost13[ixn]+rhon*vz/6.0;
       d_f15[ixn]=fpost18[ixn]+rhon*vz/6.0;
       d_f16[ixn]=fpost17[ixn]+rhon*vz/6.0;

}

if((i>=x5) && (i<=x6)&&(j>=y5)&&(j<=y6)) {
       int ixn=i+L*j+M*L*(N-1);
       int ixno=i+L*j+M*L*(N-2);
       d_f6[ixn]=fpost6[ixno];
       d_f13[ixn]=fpost13[ixno];
       d_f14[ixn]=fpost14[ixno];
       d_f17[ixn]=fpost17[ixno];
       d_f18[ixn]=fpost18[ixno];

}

//Partition

if((i>=x1) && (i<=x2)){
      int p=M/2;
       d_f4[i+p*L+M*L*k]=fpost3[i+p*L+M*L*k];
       d_f10[i+p*L+M*L*k]=fpost7[i+p*L+M*L*k];
       d_f9[i+p*L+M*L*k]=fpost8[i+p*L+M*L*k];
       d_f16[i+p*L+M*L*k]=fpost17[i+p*L+M*L*k];
       d_f18[i+p*L+M*L*k]=fpost15[i+p*L+M*L*k];

    }
   if((j>=0) &&(j<=(M/2))){
/*
//right wall
       d_f2[x2+j*L+M*L*k]=fpost1[x2+j*L+M*L*k];
       d_f12[x2+j*L+M*L*k]=fpost13[x2+j*L+M*L*k];
       d_f14[x2+j*L+M*L*k]=fpost11[x2+j*L+M*L*k];
       d_f8[x2+j*L+M*L*k]=fpost9[x2+j*L+M*L*k];
       d_f10[x2+j*L+M*L*k]=fpost7[x2+j*L+M*L*k];
// left wall
       d_f1[x1+j*L+M*L*k]=fpost2[x1+j*L+M*L*k];
       d_f13[x1+j*L+M*L*k]=fpost12[x1+j*L+M*L*k];
       d_f11[x1+j*L+M*L*k]=fpost14[x1+j*L+M*L*k];
       d_f7[x1+j*L+M*L*k]=fpost10[x1+j*L+M*L*k];
       d_f9[x1+j*L+M*L*k]=fpost8[x1+j*L+M*L*k];     
*/
//right wall
       d_f2[x1+j*L+M*L*k]=fpost1[x1+j*L+M*L*k];
       d_f12[x1+j*L+M*L*k]=fpost13[x1+j*L+M*L*k];
       d_f14[x1+j*L+M*L*k]=fpost11[x1+j*L+M*L*k];
       d_f8[x1+j*L+M*L*k]=fpost9[x1+j*L+M*L*k];
       d_f10[x1+j*L+M*L*k]=fpost7[x1+j*L+M*L*k];
// left wall
       d_f1[x2+j*L+M*L*k]=fpost2[x2+j*L+M*L*k];
       d_f13[x2+j*L+M*L*k]=fpost12[x2+j*L+M*L*k];
       d_f11[x2+j*L+M*L*k]=fpost14[x2+j*L+M*L*k];
       d_f7[x2+j*L+M*L*k]=fpost10[x2+j*L+M*L*k];
       d_f9[x2+j*L+M*L*k]=fpost8[x2+j*L+M*L*k];

    }




        // update density at interior nodes
      if((i>0) && (i<L-1) && (j>0) && (j<M-1)&& (k>0) && (k<N-1)) {
        int ixyz=i+j*L+M*L*k;
           d_rho[ixyz] =d_f0[ixyz]+d_f1[ixyz]+ d_f2[ixyz]+ d_f3[ixyz]+ d_f4[ixyz]+ d_f5[ixyz]+ d_f6[ixyz]+ d_f7[ixyz]+ d_f8[ixyz]
                +d_f9[ixyz] +d_f10[ixyz]+d_f11[ixyz]+ d_f12[ixyz]+ d_f13[ixyz]+ d_f14[ixyz]+ d_f15[ixyz]+ d_f16[ixyz]+ d_f17[ixyz]+ d_f18[ixyz];

  // update velocity at interior nodes
     d_ux[ixyz] = (d_f1[ixyz]-d_f2[ixyz]+d_f7[ixyz]-d_f8[ixyz]+d_f9[ixyz]-d_f10[ixyz]+d_f11[ixyz]-d_f12[ixyz]+d_f13[ixyz]-d_f14[ixyz])/d_rho[ixyz]+0.5*fx;
     d_uy[ixyz] = (d_f3[ixyz]-d_f4[ixyz]+d_f7[ixyz]+d_f8[ixyz]-d_f9[ixyz]-d_f10[ixyz]+d_f15[ixyz]-d_f16[ixyz]+d_f17[ixyz]-d_f18[ixyz])/d_rho[ixyz]+0.5*fy;
     d_uz[ixyz] = (d_f5[ixyz]-d_f6[ixyz]+d_f11[ixyz]+d_f12[ixyz]-d_f13[ixyz]-d_f14[ixyz]+d_f15[ixyz]+d_f16[ixyz]-d_f17[ixyz]-d_f18[ixyz])/d_rho[ixyz]+0.5*fz;
}
__syncthreads();

if((i>=x5) && (i<=x6)&&(j>=y5)&&(j<=y6)) {
d_ux[i+j*L+(N-1)*M*L]=0.0;
d_uy[i+j*L+(N-1)*M*L]=0.0;
//d_uz[i+j*L+(N-1)*M*L]=d_uz[i+j*L+(N-2)*M*L];
}

if((i>=x1) && (i<=x2)&& (j>=0) &&(j<=(M/2))){
      d_ux[i+j*L+k*M*L]=0.0;
      d_uy[i+j*L+k*M*L]=0.0;
      d_uz[i+j*L+k*M*L]=0.0;
      }

__syncthreads();


  double t1n=d_ux[ixyz];
  double t2n=-d_ux[ixyz];
  double t3n=d_uy[ixyz];
  double t4n=-d_uy[ixyz];
  double t5n=d_uz[ixyz];
  double t6n=-d_uz[ixyz];
  double  geq1=d_t[ixyz]*(1./6.)*(1.0+3.0*t1n);
  double  geq2=d_t[ixyz]*(1./6.)*(1.0+3.0*t2n);
  double  geq3=d_t[ixyz]*(1./6.)*(1.0+3.0*t3n);
  double  geq4=d_t[ixyz]*(1./6.)*(1.0+3.0*t4n);
  double  geq5=d_t[ixyz]*(1./6.)*(1.0+3.0*t5n);
  double  geq6=d_t[ixyz]*(1./6.)*(1.0+3.0*t6n);
  gpost1[ixyz]=omegat*geq1+(1.-omegat)*d_g1[ixyz];
  gpost2[ixyz]=omegat*geq2+(1.-omegat)*d_g2[ixyz];
  gpost3[ixyz]=omegat*geq3+(1.-omegat)*d_g3[ixyz];
  gpost4[ixyz]=omegat*geq4+(1.-omegat)*d_g4[ixyz];
  gpost5[ixyz]=omegat*geq5+(1.-omegat)*d_g5[ixyz];
  gpost6[ixyz]=omegat*geq6+(1.-omegat)*d_g6[ixyz];
  d_g1[ixyz]  = gpost1[im+j*L+M*L*k];
  d_g2[ixyz]  = gpost2[ip+j*L+M*L*k ];
  d_g3[ixyz]  = gpost3[i+jm*L+M*L*k ];
  d_g4[ixyz]  = gpost4[i+jp*L+M*L*k];
  d_g5[ixyz]  = gpost5[i+j*L+M*L*km ];
  d_g6[ixyz]  = gpost6[i+j*L+M*L*kp ];






// south and north wall
 if((j>y7) && (j<=M-1)&&(k>=0)&&(k<=N-1)) {
  d_g1[0+j*L+M*L*k]=gpost1[1+j*L+M*L*k];
  d_g2[0+j*L+M*L*k]=gpost2[1+j*L+M*L*k];
  d_g3[0+j*L+M*L*k]=gpost3[1+j*L+M*L*k];
  d_g4[0+j*L+M*L*k]=gpost4[1+j*L+M*L*k];
  d_g5[0+j*L+M*L*k]=gpost5[1+j*L+M*L*k];
  d_g6[0+j*L+M*L*k]=gpost6[1+j*L+M*L*k];

}

// left wall BC
if((j>=0) && (j<=y7)&& (k>=0)&& (k<=N-1)) {
 d_g1[0+j*L+M*L*k]=2.0*(1./6.0)*tw-gpost2[0+j*L+M*L*k];

}
//right wall BC  
if((j>=0) && (j<=M-1)&& (k>=0)&& (k<=N-1)) {
 d_g2[(L-1)+j*L+M*L*k]=-gpost1[(L-1)+j*L+M*L*k];
}
// south and north wall
 if((i>=0) && (i<=L-1)&&(j>=0)&&(j<=M-1)) {
  d_g1[i+j*L+M*L*0]=gpost1[i+j*L+M*L*1];
  d_g2[i+j*L+M*L*0]=gpost2[i+j*L+M*L*1];
  d_g3[i+j*L+M*L*0]=gpost3[i+j*L+M*L*1];
  d_g4[i+j*L+M*L*0]=gpost4[i+j*L+M*L*1];
  d_g5[i+j*L+M*L*0]=gpost5[i+j*L+M*L*1];
  d_g6[i+j*L+M*L*0]=gpost6[i+j*L+M*L*1];

 d_g1[i+j*L+M*L*(N-1)]=gpost1[i+j*L+M*L*(N-2)];
 d_g2[i+j*L+M*L*(N-1)]=gpost2[i+j*L+M*L*(N-2)];
 d_g3[i+j*L+M*L*(N-1)]=gpost3[i+j*L+M*L*(N-2)];
 d_g4[i+j*L+M*L*(N-1)]=gpost4[i+j*L+M*L*(N-2)];
 d_g5[i+j*L+M*L*(N-1)]=gpost5[i+j*L+M*L*(N-2)];
 d_g6[i+j*L+M*L*(N-1)]=gpost6[i+j*L+M*L*(N-2)];
 }
 __syncthreads();

 // top and bottom wall
if((i>=0) && (i<=L-1)&&(k>=0)&&(k<=N-1)) {
 d_g1[i+0*L+M*L*k]=gpost1[i+1*L+M*L*k];
 d_g2[i+0*L+M*L*k]=gpost2[i+1*L+M*L*k];
 d_g3[i+0*L+M*L*k]=gpost3[i+1*L+M*L*k];
 d_g4[i+0*L+M*L*k]=gpost4[i+1*L+M*L*k];
 d_g5[i+0*L+M*L*k]=gpost5[i+1*L+M*L*k];
  d_g6[i+0*L+M*L*k]=gpost6[i+1*L+M*L*k];

  d_g1[i+(M-1)*L+M*L*k]=gpost1[i+(M-2)*L+M*L*k];
  d_g2[i+(M-1)*L+M*L*k]=gpost2[i+(M-2)*L+M*L*k];
  d_g3[i+(M-1)*L+M*L*k]=gpost3[i+(M-2)*L+M*L*k];
  d_g4[i+(M-1)*L+M*L*k]=gpost4[i+(M-2)*L+M*L*k];
  d_g5[i+(M-1)*L+M*L*k]=gpost5[i+(M-2)*L+M*L*k];
  d_g6[i+(M-1)*L+M*L*k]=gpost6[i+(M-2)*L+M*L*k];

}
__syncthreads();

  if((i>=0) && (i<=L-1) && (j>=0) && (j<=M-1)&& (k>=0) && (k<=N-1)) {
      d_t[ixyz] =d_g1[ixyz]+ d_g2[ixyz]+ d_g3[ixyz]+ d_g4[ixyz]+ d_g5[ixyz]+ d_g6[ixyz];

  }
  __syncthreads();
   if((i>=0) && (i<=L-1)&&(j>=0)&&(j<=M-1)) {
       d_t[i+j*L+L*M*(N-1)]=d_t[i+j*L+L*M*(N-2)];
       d_t[i+j*L+L*M*0]=d_t[i+j*L+L*M*1];
   }
   if((i>=0) && (i<=L-1)&&(k>=0)&&(k<=N-1)) {
       d_t[i+0*L+L*M*k]=d_t[i+1*L+L*M*k];
       d_t[i+(M-1)*L+L*M*k]=d_t[i+(M-2)*L+L*M*k];
   }
   __syncthreads();

 //time average moments
     int timediff=TIME_STEPS-time;

    if(timediff<avg_time){
    // tot_step = tot_step+1;
     umean[ixyz]+= d_ux[ixyz];
     vmean[ixyz]+= d_uy[ixyz];
     wmean[ixyz]+= d_uz[ixyz];
     tmean[ixyz]+= d_t[ixyz];
     double fluc_u=d_ux[ixyz]-umean[ixyz]/(avg_time-timediff);
     double fluc_v=d_uy[ixyz]-umean[ixyz]/(avg_time-timediff);
     double fluc_w=d_uz[ixyz]-umean[ixyz]/(avg_time-timediff);
	double fluc_t=d_t[ixyz]-tmean[ixyz]/(avg_time-timediff);     
     ups[ixyz] +=fluc_u*fluc_u;
     vps[ixyz] +=fluc_v*fluc_v;
     wps[ixyz] +=fluc_w*fluc_w;
     uvp[ixyz] +=fluc_u*fluc_v;
     vwp[ixyz] +=fluc_v*fluc_w;
     uwp[ixyz] +=fluc_u*fluc_w;
     tps[ixyz] +=fluc_t*fluc_t;
}
__syncthreads();
//}
}

