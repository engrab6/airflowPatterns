__global__ void initialize(double *d_rho, double *d_ux, double *d_uy, double *d_uz,double *d_t,
                           double *d_f0,double *d_f1,double *d_f2,double *d_f3,double *d_f4,double *d_f5,double *d_f6,double *d_f7,double *d_f8,
                          double *d_f9,double *d_f10,double *d_f11,double *d_f12,double *d_f13,double *d_f14,double *d_f15,double *d_f16,double *d_f17,double *d_f18,
                          double *umean,double *vmean,double *wmean,double *ups,double *vps,double *wps
                          ,double *uvp,double *vwp,double *uwp,double *tps,double *tmean,double *d_g1,double *d_g2,double *d_g3,double *d_g4,double *d_g5,double *d_g6,double *random)

{

     int i = threadIdx.x;
     int j = blockIdx.x;
     int k = blockIdx.y;

    // initialize density and velocity fields inside the cavity
    d_rho[i+L*j+M*L*k] = DENSITY;
    d_ux[i+L*j+M*L*k] = 0;
    d_uy[i+L*j+M*L*k] = 0 ;
    d_uz[i+L*j+M*L*k] = 0 ;
    d_t[i+L*j+M*L*k] = 0 ;

         if(j>=0 && j<=y7 && k>=0 && k<=(N-1)) 
       {
        d_t[0+L*j+M*L*k] = tw;
        }
	 if(i==(L-1))d_t[i+L*j+M*L*k] = tc;
    //omega[i+L*j+M*L*k]=omega_temp;
if((i>=x3) && (i<=x4)&& (j>=y3)&& (j<=y4)) {

    
    if(k==0) d_uz[i+L*j+M*L*0] = u0*(1 + al * random[i+L*j+M*L*(N-1)]);
}

  // assign initial values for distribution functions
     //if((i>0) && (i<=L-1) && (j>0) && (j<=M-1)&& (k>0) && (k<=N-1)) {
       int ixyz = i+L*j+M*L*k;
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
        d_f0[ixyz]=d_rho[ixyz]*(1./3.)*(1.0-1.5*ts);
        d_f1[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t1+4.5*t1*t1-1.5*ts);
        d_f2[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t2+4.5*t2*t2-1.5*ts);
        d_f3[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t3+4.5*t3*t3-1.5*ts);
       d_f4[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t4+4.5*t4*t4-1.5*ts);
       d_f5[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t5+4.5*t5*t5-1.5*ts);
       d_f6[ixyz]=d_rho[ixyz]*(1./18.)*(1.0+3.0*t6+4.5*t6*t6-1.5*ts);
       d_f7[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t7+4.5*t7*t7-1.5*ts);
       d_f8[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t8+4.5*t8*t8-1.5*ts);
       d_f9[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t9+4.5*t9*t9-1.5*ts);
       d_f10[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t10+4.5*t10*t10-1.5*ts);
       d_f11[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t11+4.5*t11*t11-1.5*ts);
       d_f12[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t12+4.5*t12*t12-1.5*ts);
       d_f13[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t13+4.5*t13*t13-1.5*ts);
       d_f14[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t14+4.5*t14*t14-1.5*ts);
       d_f15[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t15+4.5*t15*t15-1.5*ts);
       d_f16[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t16+4.5*t16*t16-1.5*ts);
       d_f17[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t17+4.5*t17*t17-1.5*ts);
       d_f18[ixyz]=d_rho[ixyz]*(1./36.)*(1.0+3.0*t18+4.5*t18*t18-1.5*ts);
       d_g1[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t1);
       d_g2[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t2);
       d_g3[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t3);
       d_g4[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t4);
       d_g5[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t5);
       d_g6[ixyz]=d_t[ixyz]*(1./6.)*(1.0+3.0*t6);
       //statistics
        umean[ixyz]=0.0;
        vmean[ixyz]=0.0;
        wmean[ixyz]=0.0;
        tmean[ixyz]=0.0;
        ups[ixyz] =0.0;
        vps[ixyz] =0.0;
        wps[ixyz] =0.0;
        uvp[ixyz] =0.0;
        vwp[ixyz] =0.0;
        uwp[ixyz] =0.0;
        tps[ixyz]=0.0;

//}
}

