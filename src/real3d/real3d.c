/**   
 *    BEC-GP-OMP codes are developed and (c)opyright-ed by:
 *  
 *    Luis E. Young-S., Sadhan K. Adhikari
 *    (UNESP - Sao Paulo State University, Brazil)
 *  
 *    Paulsamy Muruganandam
 *    (Bharathidasan University, Tamil Nadu, India)
 *
 *    Dusan Vudragovic, Antun Balaz
 *    (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 *    Public use and modification of this code are allowed provided that the 
 *    following three papers are cited:
 *
 *    [1] L. E. Young-S. et al., Comput. Phys. Commun. 204 (2016) 209.
 *    [2] P. Muruganandam, S. K. Adhikari, Comput. Phys. Commun. 180 (2009) 1888.
 *    [3] D. Vudragovic et al., Comput. Phys. Commun. 183 (2012) 2021.
 *
 *    The authors would be grateful for all information and/or comments
 *    regarding the use of the code.
 *    
 *    This program solves the time-dependent Gross–Pitaevskii nonlinear
 *    partial differential equation in three space dimensions in a trap using
 *    the real-time propagation. The Gross–Pitaevskii equation describes the
 *    properties of a dilute trapped Bose–Einstein condensate. The equation is
 *    solved using the split-step Crank–Nicolson method by discretizing space
 *    and time. The discretized equation is then propagated in real time over
 *    small time steps.
 *    
 *    Description of variables used in the code:
 *    
 *    opt     - decides which rescaling of GP equation will be used
 *    par     - parameter for rescaling of GP equation
 *    psi     - array with the wave function values
 *    pot     - array with the values of the potential
 *    G0      - final nonlinearity
 *    Gpar    - coefficient that multiplies nonlinear term in non-stationary
 *              problem during final Nrun iterations
 *    norm    - wave function norm
 *    rms     - root mean square radius
 *    mu      - chemical potential
 *    en      - energy
 *    Nx      - number of discretization points in the x-direction
 *    Ny      - number of discretization points in the y-direction
 *    Nz      - number of discretization points in the z-direction
 *    x       - array with the space mesh values in the x-direction
 *    y       - array with the space mesh values in the y-direction
 *    z       - array with the space mesh values in the z-direction
 *    dx      - spatial discretization step in the x-direction
 *    dy      - spatial discretization step in the y-direction
 *    dz      - spatial discretization step in the z-direction
 *    dt      - time discretization step
 *    vgamma  - gamma coefficient of anisotropy of the trap
 *    vnu     - nu coefficient of anisotropy of the trap
 *    vlambda - lambda coefficient of anisotropy of the trap
 *    Nstp    - number of subsequent iterations with fixed nonlinearity G0
 *    Npas    - number of subsequent iterations with the fixed nonlinearity G0
 *    Nrun    - number of final iterations with the fixed nonlinearity G0
 *    output  - output file with the summary of final values of all physical
 *              quantities
 *    initout - output file with the initial wave function
 *    Nstpout - output file with the wave function obtained after the first
 *              Nstp iterations
 *    Npasout - output file with the wave function obtained after the
 *              subsequent Npas iterations, with the fixed nonlinearity G0
 *    Nrunout - output file with the final wave function obtained after the
 *              final Nrun iterations
 *    rmsout  - output file with the time dependence of RMS during the final
 *              Nrun iterations
 *    outstpx - discretization step in the x-direction used to save wave
 *              functions
 *    outstpy - discretization step in the y-direction used to save wave
 *              functions
 *    outstpz - discretization step in the z-direction used to save wave
 *              functions
 *    outstpt - time discretization step used to save RMS of the wave function
 */

#include "real3d.h"

int main(int argc, char **argv) {
   FILE *out;
   FILE *file;
   FILE *filerms;  
   FILE *dyna;     
   int nthreads;
   char filename[MAX_FILENAME_SIZE];
   long cnti;
   double norm, mu, en;
   double *rms;    
   double complex ***psi;
   double complex **cbeta;
   double ***dpsix, ***dpsiy, ***dpsiz;
   double **tmpxi, **tmpyi, **tmpzi, **tmpxj, **tmpyj, **tmpzj;
   double **tmpxk, **tmpyk, **tmpzk;   
   double *tmpx, *tmpy, *tmpz; 
   double psi2;
   double ***abc;
   
   time_t clock_beg, clock_end; 
   clock_beg = time(NULL);     
   pi = 4. * atan(1.);
   
   if((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      exit(EXIT_FAILURE);
   }

   if(! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      exit(EXIT_FAILURE);
   }
   
   readpar();
   
   #pragma omp parallel
      #pragma omp master
         nthreads = omp_get_num_threads();
	 
   rms = alloc_double_vector(RMS_ARRAY_SIZE); 	 
   
   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);
   z = alloc_double_vector(Nz);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);
   z2 = alloc_double_vector(Nz);

   pot = alloc_double_tensor(Nx, Ny, Nz);
   psi = alloc_complex_tensor(Nx, Ny, Nz);
   abc = alloc_double_tensor(Nx, Ny, Nz);   
   
   dpsix = alloc_double_tensor(Nx, Ny, Nz);
   dpsiy = alloc_double_tensor(Nx, Ny, Nz);
   dpsiz = alloc_double_tensor(Nx, Ny, Nz);
   
   calphax = alloc_complex_vector(Nx - 1);
   calphay = alloc_complex_vector(Ny - 1);
   calphaz = alloc_complex_vector(Nz - 1);
   cbeta =  alloc_complex_matrix(nthreads, MAX(Nx, Ny, Nz) - 1);
   cgammax = alloc_complex_vector(Nx - 1);
   cgammay = alloc_complex_vector(Ny - 1);
   cgammaz = alloc_complex_vector(Nz - 1);
   
   tmpxi = alloc_double_matrix(nthreads, Nx);
   tmpyi = alloc_double_matrix(nthreads, Ny);
   tmpzi = alloc_double_matrix(nthreads, Nz);
   tmpxj = alloc_double_matrix(nthreads, Nx);
   tmpyj = alloc_double_matrix(nthreads, Ny);
   tmpzj = alloc_double_matrix(nthreads, Nz);
   tmpxk = alloc_double_matrix(nthreads, Nx);
   tmpyk = alloc_double_matrix(nthreads, Ny);
   tmpzk = alloc_double_matrix(nthreads, Nz);   
   
   tmpx = alloc_double_vector(Nx);
   tmpy = alloc_double_vector(Ny);
   tmpz = alloc_double_vector(Nz);   
   
   if(output != NULL) {
      sprintf(filename, "%s.txt", output);
      out = fopen(filename, "w");
   }
   else out = stdout;
   
   if(rmsout != NULL) {
      sprintf(filename, "%s.txt", rmsout);
      filerms = fopen(filename, "w");
   }   
   else filerms = NULL; 
   
   fprintf(out, " Real time propagation 3D,   OPTION = %d\n\n", opt);
   fprintf(out, "  Number of Atoms N = %li, Unit of length AHO = %.8f m\n", Na, aho);
   fprintf(out, "  Scattering length a = %.2f*a0\n", as);
   fprintf(out, "  Nonlinearity G_3D = %.7f\n", G0);
   fprintf(out, "  Parameters of trap: GAMMA = %.5f, NU = %.5f, LAMBDA = %.5f\n", vgamma, vnu, vlambda);
   fprintf(out, " # Space Stp: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
   fprintf(out, "  Space Step: DX = %.4f, DY = %.4f, DZ = %.4f\n", dx, dy, dz);
   fprintf(out, " # Time Stp : NSTP = %li, NPAS = %li, NRUN = %li\n", Nstp, Npas, Nrun);
   fprintf(out, "   Time Step:   DT = %.6f\n\n",  dt);
   fprintf(out, " * Change for dynamics: GPAR = %.3f *\n\n", Gpar); 
   fprintf(out, "                  ----------------------------------------------------------\n");
   fprintf(out, "                    Norm      Chem        Ener/N      <r>      |Psi(0,0,0)|^2\n");
   fprintf(out, "                  ----------------------------------------------------------\n");   
   fflush(out);
   if(rmsout != NULL) {
     fprintf(filerms, " Real time propagation 3D,   OPTION = %d\n\n", opt);
     fprintf(filerms, "                  ---------------------------------------------------------\n");
     fprintf(filerms, "Values of rms size:       <r>          <x>          <y>          <z>\n");
     fprintf(filerms, "                  --------------------------------------------------------\n");      
     fflush(filerms);
   }   

   if(Nstp == 0) { 
     G = par * G0;  
   }
   else {
     G = 0.;
   }    
   
   init(psi, abc);
   gencoef();
   calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
   calcmuen(&mu, &en, psi, dpsix, dpsiy, dpsiz, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
   calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);
   psi2 = cabs(psi[Nx2][Ny2][Nz2]) * cabs(psi[Nx2][Ny2][Nz2]);
   fprintf(out, "Initial : %15.4f %11.3f %11.3f %10.3f %10.3f\n", norm, mu / par, en / par, *rms, psi2);
   fflush(out);   
   if(rmsout != NULL) {
     fprintf(filerms, "           Initial: %11.5f  %11.5f  %11.5f  %11.5f\n", rms[0], rms[1], rms[2], rms[3]);
     fflush(filerms);
   }    
   if(initout != NULL) {
      sprintf(filename, "%s.txt", initout);
      file = fopen(filename, "w");
      outdenxyz(psi, file);
      fclose(file);
     
      sprintf(filename, "%s1d_x.txt", initout);
      file = fopen(filename, "w");
      outdenx(psi, tmpy, tmpz, file);
      fclose(file);

      sprintf(filename, "%s1d_y.txt", initout);
      file = fopen(filename, "w");
      outdeny(psi, tmpx, tmpz, file);
      fclose(file);      
      
      sprintf(filename, "%s1d_z.txt", initout);
      file = fopen(filename, "w");
      outdenz(psi, tmpx, tmpy, file);
      fclose(file);
/*   
      sprintf(filename, "%s2d_xy.txt", initout);
      file = fopen(filename, "w");
      outdenxy(psi, tmpz, file);
      fclose(file);       

      sprintf(filename, "%s2d_xz.txt", initout);
      file = fopen(filename, "w");
      outdenxz(psi, tmpy, file);
      fclose(file);      
      
      sprintf(filename, "%s2d_yz.txt", initout);
      file = fopen(filename, "w");
      outdenyz(psi, tmpx, file);
      fclose(file);    

      sprintf(filename, "%s3d_x0z.txt", initout);
      file = fopen(filename, "w");
      outpsi2xz(psi, file);
      fclose(file);

      sprintf(filename, "%s3d_xy0.txt", initout);
      file = fopen(filename, "w");
      outpsi2xy(psi, file);
      fclose(file);     
*/   
   }
   
   if(Nstp != 0) {   
     double g_stp = par * G0 / (double) Nstp;          
     for(cnti = 0; cnti < Nstp; cnti ++) {
      G += g_stp;
      calcnu(psi);
      calclux(psi, cbeta);
      calcluy(psi, cbeta);
      calcluz(psi, cbeta);
     }
     calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
     calcmuen(&mu, &en, psi, dpsix, dpsiy, dpsiz, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
     calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);
     psi2 = cabs(psi[Nx2][Ny2][Nz2]) * cabs(psi[Nx2][Ny2][Nz2]);
     fprintf(out, "After NSTP iter.:%8.4f %11.3f %11.3f %10.3f %10.3f\n", norm, mu / par, en / par, *rms, psi2);
     fflush(out);   
     if(rmsout != NULL) {
       fprintf(filerms, "  After NSTP iter.: %11.5f  %11.5f  %11.5f  %11.5f\n", rms[0], rms[1], rms[2], rms[3]);
       fflush(filerms);
     }
     if(Nstpout != NULL) {
      sprintf(filename, "%s.txt", Nstpout);
      file = fopen(filename, "w");
      outdenxyz(psi, file);
      fclose(file);
     
      sprintf(filename, "%s1d_x.txt", Nstpout);
      file = fopen(filename, "w");
      outdenx(psi, tmpy, tmpz, file);
      fclose(file);

      sprintf(filename, "%s1d_y.txt", Nstpout);
      file = fopen(filename, "w");
      outdeny(psi, tmpx, tmpz, file);
      fclose(file);      
      
      sprintf(filename, "%s1d_z.txt", Nstpout);
      file = fopen(filename, "w");
      outdenz(psi, tmpx, tmpy, file);
      fclose(file);
/*   
      sprintf(filename, "%s2d_xy.txt", Nstpout);
      file = fopen(filename, "w");
      outdenxy(psi, tmpz, file);
      fclose(file);       

      sprintf(filename, "%s2d_xz.txt", Nstpout);
      file = fopen(filename, "w");
      outdenxz(psi, tmpy, file);
      fclose(file);      
      
      sprintf(filename, "%s2d_yz.txt", Nstpout);
      file = fopen(filename, "w");
      outdenyz(psi, tmpx, file);
      fclose(file);    

      sprintf(filename, "%s3d_x0z.txt", Nstpout);
      file = fopen(filename, "w");
      outpsi2xz(psi, file);
      fclose(file);

      sprintf(filename, "%s3d_xy0.txt", Nstpout);
      file = fopen(filename, "w");
      outpsi2xy(psi, file);
      fclose(file);     
*/     
     }     
   }

   if(dynaout != NULL) {
      sprintf(filename, "%s.txt", dynaout);
      dyna = fopen(filename, "w");
   }
   else dyna = NULL; 
 
   for(cnti = 0; cnti < Npas; cnti ++) {
      calcnu(psi);
      calclux(psi, cbeta);
      calcluy(psi, cbeta);
      calcluz(psi, cbeta);
      if((dynaout != NULL) && (cnti % outstpt == 0)) {
         calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);;
         fprintf(dyna, "%5le   %5le      %5le   %5le\n", cnti * dt * par,  rms[1], rms[2], rms[3]);
         fflush(dyna);
      }      
   }
   if(Npas != 0){      
      calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      calcmuen(&mu, &en, psi, dpsix, dpsiy, dpsiz, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
      calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);
      psi2 = cabs(psi[Nx2][Ny2][Nz2]) * cabs(psi[Nx2][Ny2][Nz2]);
      fprintf(out, "After NPAS iter.:%8.4f %11.3f %11.3f %10.3f %10.3f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);
      if(rmsout != NULL) {
        fprintf(filerms, "  After NPAS iter.: %11.5f  %11.5f  %11.5f  %11.5f\n", rms[0], rms[1], rms[2], rms[3]);
        fflush(filerms);
      }      
      if(Npasout != NULL) {
	sprintf(filename, "%s.txt", Npasout);
	file = fopen(filename, "w");
	outdenxyz(psi, file);
	fclose(file);
     
	sprintf(filename, "%s1d_x.txt", Npasout);
	file = fopen(filename, "w");
	outdenx(psi, tmpy, tmpz, file);
	fclose(file);

	sprintf(filename, "%s1d_y.txt", Npasout);
	file = fopen(filename, "w");
	outdeny(psi, tmpx, tmpz, file);
	fclose(file);      
      
	sprintf(filename, "%s1d_z.txt", Npasout);
	file = fopen(filename, "w");
	outdenz(psi, tmpx, tmpy, file);
	fclose(file);
/*   
	sprintf(filename, "%s2d_xy.txt", Npasout);
	file = fopen(filename, "w");
	outdenxy(psi, tmpz, file);
	fclose(file);       

	sprintf(filename, "%s2d_xz.txt", Npasout);
	file = fopen(filename, "w");
	outdenxz(psi, tmpy, file);
	fclose(file);      
      
	sprintf(filename, "%s2d_yz.txt", Npasout);
	file = fopen(filename, "w");
	outdenyz(psi, tmpx, file);
	fclose(file);    

	sprintf(filename, "%s3d_x0z.txt", Npasout);
	file = fopen(filename, "w");
	outpsi2xz(psi, file);
	fclose(file);

	sprintf(filename, "%s3d_xy0.txt", Npasout);
	file = fopen(filename, "w");
	outpsi2xy(psi, file);
	fclose(file);     
*/   
      }
   }
   
   G *= Gpar;
   
   for(cnti = 1; cnti <= Nrun; cnti ++) {
      calcnu(psi);
      calclux(psi, cbeta);
      calcluy(psi, cbeta);
      calcluz(psi, cbeta);
      if((dynaout != NULL) && (cnti % outstpt == 0)) {
         calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);;
         fprintf(dyna, "%5le   %5le      %5le   %5le\n", (cnti * dt  + dt * Npas) * par,  rms[1], rms[2], rms[3]);
         fflush(dyna);
      }      
   }
 if(dynaout != NULL) fclose(dyna);
   if(Nrun != 0){   
      calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      calcmuen(&mu, &en, psi, dpsix, dpsiy, dpsiz, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
      calcrms(rms, psi, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpxk, tmpyk, tmpzk);
      psi2 = cabs(psi[Nx2][Ny2][Nz2]) * cabs(psi[Nx2][Ny2][Nz2]);
      fprintf(out, "After NRUN iter.:%8.4f %11.3f %11.3f %10.3f %10.3f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);      
      if(rmsout != NULL) {
        fprintf(filerms, "  After NRUN iter.: %11.5f  %11.5f  %11.5f  %11.5f\n", rms[0], rms[1], rms[2], rms[3]);
        fprintf(filerms, "                  --------------------------------------------------------\n");        
        fflush(filerms);
      } 
      if(Nrunout != NULL) {
	sprintf(filename, "%s.txt", Nrunout);
	file = fopen(filename, "w");
	outdenxyz(psi, file);
	fclose(file);
     
	sprintf(filename, "%s1d_x.txt", Nrunout);
	file = fopen(filename, "w");
	outdenx(psi, tmpy, tmpz, file);
	fclose(file);

	sprintf(filename, "%s1d_y.txt", Nrunout);
	file = fopen(filename, "w");
	outdeny(psi, tmpx, tmpz, file);
	fclose(file);      
      
	sprintf(filename, "%s1d_z.txt", Nrunout);
	file = fopen(filename, "w");
	outdenz(psi, tmpx, tmpy, file);
	fclose(file);
/*   
	sprintf(filename, "%s2d_xy.txt", Nrunout);
	file = fopen(filename, "w");
	outdenxy(psi, tmpz, file);
	fclose(file);       
  
	sprintf(filename, "%s2d_xz.txt", Nrunout);
	file = fopen(filename, "w");
	outdenxz(psi, tmpy, file);
	fclose(file);      
	
	sprintf(filename, "%s2d_yz.txt", Nrunout);
	file = fopen(filename, "w");
	outdenyz(psi, tmpx, file);
	fclose(file);    

	sprintf(filename, "%s3d_x0z.txt", Nrunout);
	file = fopen(filename, "w");
	outpsi2xz(psi, file);
	fclose(file);
  
	sprintf(filename, "%s3d_xy0.txt", Nrunout);
	file = fopen(filename, "w");
	outpsi2xy(psi, file);
	fclose(file);     
*/  
      }
   }
      
   if(rmsout != NULL) fclose(filerms);    
   
   fprintf(out, "                  --------------------------------------------------------\n\n");
   
   free_double_vector(rms); 
   
   free_double_vector(x);
   free_double_vector(y);
   free_double_vector(z);

   free_double_vector(x2);
   free_double_vector(y2);
   free_double_vector(z2);
   
   free_double_tensor(pot);
   free_complex_tensor(psi);
   free_double_tensor(abc);
   
   free_double_tensor(dpsix);
   free_double_tensor(dpsiy);
   free_double_tensor(dpsiz);
   
   free_complex_vector(calphax);
   free_complex_vector(calphay);
   free_complex_vector(calphaz);
   free_complex_matrix(cbeta);
   free_complex_vector(cgammax);
   free_complex_vector(cgammay);
   free_complex_vector(cgammaz);
   
   free_double_matrix(tmpxi);
   free_double_matrix(tmpyi);
   free_double_matrix(tmpzi);
   free_double_matrix(tmpxj);
   free_double_matrix(tmpyj);
   free_double_matrix(tmpzj);
   free_double_matrix(tmpxk);
   free_double_matrix(tmpyk);
   free_double_matrix(tmpzk);   

   free_double_vector(tmpx);
   free_double_vector(tmpy);
   free_double_vector(tmpz);    
   
   clock_end = time(NULL);
   double wall_time = difftime(clock_end, clock_beg);
   double cpu_time = clock() / (double) CLOCKS_PER_SEC;
   fprintf(out, " Clock Time: %.f seconds\n", wall_time);
   fprintf(out, " CPU Time: %.f seconds\n", cpu_time);    
   
   if(output != NULL) fclose(out);   
   
   return(EXIT_SUCCESS);
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
   char *cfg_tmp;
   
   if((cfg_tmp = cfg_read("OPTION")) == NULL) {
      fprintf(stderr, "OPTION is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   opt = atol(cfg_tmp);
   
   if((cfg_tmp = cfg_read("G0")) == NULL) {
     
      if((cfg_tmp = cfg_read("NATOMS")) == NULL) {
	fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
	exit(EXIT_FAILURE);
      }
      Na = atol(cfg_tmp);
     
      if((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if((cfg_tmp = cfg_read("AS")) == NULL) {
         fprintf(stderr, "AS is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      as = atof(cfg_tmp);

      G0 = 4. * pi * as * Na * BOHR_RADIUS / aho;
   } else {
      G0 = atof(cfg_tmp);
   }

   if((cfg_tmp = cfg_read("GPAR")) == NULL) {
      fprintf(stderr, "GPAR is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Gpar = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("NX")) == NULL) {
      fprintf(stderr, "NX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nx = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("NY")) == NULL) {
      fprintf(stderr, "NY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Ny = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("NZ")) == NULL) {
      fprintf(stderr, "Nz is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nz = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("DX")) == NULL) {
      fprintf(stderr, "DX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dx = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("DY")) == NULL) {
      fprintf(stderr, "DY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dy = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("DZ")) == NULL) {
      fprintf(stderr, "DZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dz = atof(cfg_tmp);
   
   if((cfg_tmp = cfg_read("DT")) == NULL) {
      fprintf(stderr, "DT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dt = atof(cfg_tmp);
   
   if((cfg_tmp = cfg_read("GAMMA")) == NULL) {
      fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vgamma = atof(cfg_tmp);   
   
   if((cfg_tmp = cfg_read("NU")) == NULL) {
      fprintf(stderr, "NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vnu = atof(cfg_tmp);
   
   if((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
      fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vlambda = atof(cfg_tmp);
   
   if((cfg_tmp = cfg_read("NSTP")) == NULL) {
      fprintf(stderr, "NSTP is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nstp = atol(cfg_tmp);
   
   if((cfg_tmp = cfg_read("NPAS")) == NULL) {
      fprintf(stderr, "NPAS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Npas = atol(cfg_tmp);
   
   if((cfg_tmp = cfg_read("NRUN")) == NULL) {
      fprintf(stderr, "NRUN is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nrun = atol(cfg_tmp);
   
   output = cfg_read("OUTPUT");
   initout = cfg_read("INITOUT");
   rmsout = cfg_read("RMSOUT");
   dynaout = cfg_read("DYNAOUT");
   Nstpout = cfg_read("NSTPOUT");
   Npasout = cfg_read("NPASOUT");
   Nrunout = cfg_read("NRUNOUT");
   
   if((initout != NULL) || (Nstpout != NULL) || (Npasout != NULL) || (Nrunout != NULL)) {
      if((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
         fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpx = atol(cfg_tmp);
      
      if((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
         fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpy = atol(cfg_tmp);
      
      if((cfg_tmp = cfg_read("OUTSTPZ")) == NULL) {
         fprintf(stderr, "OUTSTPZ is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpz = atol(cfg_tmp);
   }
   
   if(rmsout != NULL) {
      if((cfg_tmp = cfg_read("OUTSTPT")) == NULL) {
         fprintf(stderr, "OUTSTPT is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpt = atol(cfg_tmp);
   }
   

  if(dynaout != NULL) {
      if((cfg_tmp = cfg_read("OUTSTPT")) == NULL) {
         fprintf(stderr, "OUTSTPT is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpt = atol(cfg_tmp);
   }   



   return;
}

/**
 *    Initialization of the space mesh, the potential, and the initial wave
 *    function.
 *    psi - array with the wave function values
 */
void init(double complex ***psi, double ***abc) {
   long cnti, cntj, cntk;
   double vgamma2, vnu2, vlambda2;
   double cpsi;
   double tmp;
   FILE *file;   
   
   if (opt == 1) par = 1.;
   else if (opt == 2) par = 2.;
   else {
      fprintf(stderr, "OPTION is not well defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   
   vgamma2 = vgamma * vgamma;   
   vnu2 = vnu * vnu;
   vlambda2 = vlambda * vlambda;
   
   Nx2 = Nx / 2; Ny2 = Ny / 2; Nz2 = Nz / 2;
   dx2 = dx * dx; dy2 = dy * dy; dz2 = dz * dz;
     
   for(cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
      for(cntj = 0; cntj < Ny; cntj ++) {
         y[cntj] = (cntj - Ny2) * dy;
         y2[cntj] = y[cntj] * y[cntj];
         for(cntk = 0; cntk < Nz; cntk ++) {
            z[cntk] = (cntk - Nz2) * dz;
            z2[cntk] = z[cntk] * z[cntk];
               pot[cnti][cntj][cntk] = (vgamma2 * x2[cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
         }
      }
   }
   
   if(Nstp!=0) {
      cpsi = sqrt(pi * sqrt(pi / (vgamma * vnu * vlambda)));
      for(cnti = 0; cnti < Nx; cnti ++) {
        for(cntj = 0; cntj < Ny; cntj ++) {
          for(cntk = 0; cntk < Nz; cntk ++) {
              tmp = exp(- 0.5 * (vgamma * x2[cnti] + vnu * y2[cntj] + vlambda * z2[cntk]));
              psi[cnti][cntj][cntk] = tmp / cpsi;
          }
        }
      }
   }   

   if(Nstp==0  ) {
     if((file = fopen("imag3d-den.txt", "r"))==NULL) {       /* open a text file for reading */
       printf("Run the program using the input file to read. i.g.: imag3d-den.txt\n");
       exit(1);					/*couldn't open the requested file!*/
     }
     file = fopen("imag3d-den.txt", "r");
      for(cnti = 0; cnti < Nx; cnti ++) {
        for(cntj = 0; cntj < Ny; cntj ++) {
          for(cntk = 0; cntk < Nz; cntk ++) {
	    if(fscanf(file,"%lf %lf %lf %lf\n", &x[cnti], &y[cntj], &z[cntk], &abc[cnti][cntj][cntk])) {
//             printf("%8le %8le %8le\n", x[cnti], y[cntj], z[cntk], abc[cnti][cntj][cntk]);
	    }
          }
        }
      }
      fclose(file);
      for(cnti = 0; cnti < Nx; cnti ++) {
        for(cntj = 0; cntj < Ny; cntj ++) {
          for(cntk = 0; cntk < Nz; cntk ++) {
              psi[cnti][cntj][cntk] = sqrt(abc[cnti][cntj][cntk]);
          }
        }
      }     
   }   
   
   return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(void) {
   long cnti;
   
   Ax0 = 1. + I * dt / dx2;
   Ay0 = 1. + I * dt / dy2;
   Az0 = 1. + I * dt / dz2;
   
   Ax0r = 1. - I * dt / dx2;
   Ay0r = 1. - I * dt / dy2;
   Az0r = 1. - I * dt / dz2;
   
   Ax = - 0.5 * I * dt / dx2;
   Ay = - 0.5 * I * dt / dy2;
   Az = - 0.5 * I * dt / dz2;
   
   calphax[Nx - 2] = 0.;
   cgammax[Nx - 2] = - 1. / Ax0;
   for (cnti = Nx - 2; cnti > 0; cnti --) {
      calphax[cnti - 1] = Ax * cgammax[cnti];
      cgammax[cnti - 1] = - 1. / (Ax0 + Ax * calphax[cnti - 1]);
   }
   
   calphay[Ny - 2] = 0.;
   cgammay[Ny - 2] = - 1. / Ay0;
   for (cnti = Ny - 2; cnti > 0; cnti --) {
      calphay[cnti - 1] = Ay * cgammay[cnti];
      cgammay[cnti - 1] = - 1. / (Ay0 + Ay * calphay[cnti - 1]);
   }
      
   calphaz[Nz - 2] = 0.;
   cgammaz[Nz - 2] = - 1. / Az0;
   for (cnti = Nz - 2; cnti > 0; cnti --) {
      calphaz[cnti - 1] = Az * cgammaz[cnti];
      cgammaz[cnti - 1] = - 1. / (Az0 + Az * calphaz[cnti - 1]);
   }
   
   return;
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm - wave function norm
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 *    tmpz - temporary array
 */
void calcnorm(double *norm, double complex ***psi, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;
   long cnti, cntj, cntk;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            for(cntk = 0; cntk < Nz; cntk ++) {
               tmpz[threadid][cntk] = cabs(psi[cnti][cntj][cntk]);
               tmpz[threadid][cntk] *= tmpz[threadid][cntk];
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         tmpx[0][cnti] = simpint(dy, tmpy[threadid], Ny);
      }
      #pragma omp barrier

      #pragma omp single
      *norm = sqrt(simpint(dx, tmpx[0], Nx));
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            for(cntk = 0; cntk < Nz; cntk ++) {
               psi[cnti][cntj][cntk] /= *norm;
            }
         }
      }
   }
   
   return;
}

/**
 *    Calculation of the chemical potential and energy.
 *    mu    - chemical potential
 *    en    - energy
 *    psi   - array with the wave function values
 *    dpsix - temporary array
 *    dpsiy - temporary array
 *    dpsiz - temporary array
 *    tmpxi - temporary array
 *    tmpyi - temporary array
 *    tmpzi - temporary array
 *    tmpxj - temporary array
 *    tmpyj - temporary array
 *    tmpzj - temporary array
 */
void calcmuen(double *mu, double *en, double complex ***psi, double ***dpsix, double ***dpsiy, double ***dpsiz, double **tmpxi, double **tmpyi, double **tmpzi, double **tmpxj, double **tmpyj, double **tmpzj) {
   int threadid;
   long cnti, cntj, cntk;
   double psi2, psi2lin, dpsi2;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2, psi2lin, dpsi2)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            for(cnti = 0; cnti < Nx; cnti ++) {
               tmpxi[threadid][cnti] = cabs(psi[cnti][cntj][cntk]);
            }
            diff(dx, tmpxi[threadid], tmpxj[threadid], Nx);
            for(cnti = 0; cnti < Nx; cnti ++) {
               dpsix[cnti][cntj][cntk] = tmpxj[threadid][cnti];
            }
         }
      }
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            for(cntj = 0; cntj < Ny; cntj ++) {
               tmpyi[threadid][cntj] = cabs(psi[cnti][cntj][cntk]);
            }
            diff(dy, tmpyi[threadid], tmpyj[threadid], Ny);
            for(cntj = 0; cntj < Ny; cntj ++) {
               dpsiy[cnti][cntj][cntk] = tmpyj[threadid][cntj];
            }
         }
      }
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            for(cntk = 0; cntk < Nz; cntk ++) {
               tmpzi[threadid][cntk] = cabs(psi[cnti][cntj][cntk]);
            }
            diff(dz, tmpzi[threadid], tmpzj[threadid], Nz);
            for(cntk = 0; cntk < Nz; cntk ++) {
               dpsiz[cnti][cntj][cntk] = tmpzj[threadid][cntk];
            }
         }
      }
      #pragma omp barrier

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            for(cntk = 0; cntk < Nz; cntk ++) {
               psi2 = cabs(psi[cnti][cntj][cntk]);
               psi2 *= psi2;
               psi2lin = psi2 * G;
               dpsi2 = dpsix[cnti][cntj][cntk] * dpsix[cnti][cntj][cntk] + 
                       dpsiy[cnti][cntj][cntk] * dpsiy[cnti][cntj][cntk] + 
                       dpsiz[cnti][cntj][cntk] * dpsiz[cnti][cntj][cntk];
               tmpzi[threadid][cntk] = (pot[cnti][cntj][cntk] + psi2lin) * psi2 + dpsi2;
               tmpzj[threadid][cntk] = (pot[cnti][cntj][cntk] + 0.5 * psi2lin) * psi2 + dpsi2;
            }
            tmpyi[threadid][cntj] = simpint(dz, tmpzi[threadid], Nz);
            tmpyj[threadid][cntj] = simpint(dz, tmpzj[threadid], Nz);
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
         tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
      }
   }
      
   *mu = simpint(dx, tmpxi[0], Nx);
   *en = simpint(dx, tmpxj[0], Nx);
   
   return;
}

/**
 *    Calculation of the root mean square radius.
 *    rms  - root mean square radius
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 *    tmpz - temporary array
 */
void calcrms(double *rms, double complex ***psi, double **tmpxi, double **tmpyi, double **tmpzi, double **tmpxj, double **tmpyj, double **tmpzj, double **tmpxk, double **tmpyk, double **tmpzk) {
   int threadid;
   long cnti, cntj, cntk;
   double psi2;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            for(cntk = 0; cntk < Nz; cntk ++) {
               psi2 = cabs(psi[cnti][cntj][cntk]) * cabs(psi[cnti][cntj][cntk]);
               tmpzi[threadid][cntk] = x2[cnti] * psi2;
               tmpzj[threadid][cntk] = y2[cntj] * psi2;
               tmpzk[threadid][cntk] = z2[cntk] * psi2;
            }
            tmpyi[threadid][cntj] = simpint(dz, tmpzi[threadid], Nz);
	    tmpyj[threadid][cntj] = simpint(dz, tmpzj[threadid], Nz);
            tmpyk[threadid][cntj] = simpint(dz, tmpzk[threadid], Nz);
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
	 tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
	 tmpxk[0][cnti] = simpint(dy, tmpyk[threadid], Ny);
      }
   }
   
   rms[1] = sqrt(simpint(dx, tmpxi[0], Nx));   
   rms[2] = sqrt(simpint(dx, tmpxj[0], Nx));   
   rms[3] = sqrt(simpint(dx, tmpxk[0], Nx));
   rms[0] = sqrt(rms[1] * rms[1] + rms[2] * rms[2] + rms[3] * rms[3]);
   
   return;
}


/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without spatial
 *    derivatives).
 *    psi - array with the wave function values
 */
void calcnu(double complex ***psi) {
   long cnti, cntj, cntk;
   double psi2, psi2lin, tmp;
   
   #pragma omp parallel for private(cnti, cntj, cntk, psi2, psi2lin, tmp)
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            psi2 = cabs(psi[cnti][cntj][cntk]);
            psi2 *= psi2;
            psi2lin = psi2 * G;
            tmp = dt * (pot[cnti][cntj][cntk] + psi2lin);
            psi[cnti][cntj][cntk] *= cexp(- I * tmp);
         }
      }
   }
   
   return;
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calclux(double complex ***psi, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Nx - 2] = psi[Nx - 1][cntj][cntk];
            for (cnti = Nx - 2; cnti > 0; cnti --) {
               c = - Ax * psi[cnti + 1][cntj][cntk] + Ax0r * psi[cnti][cntj][cntk] - Ax * psi[cnti - 1][cntj][cntk];
               cbeta[threadid][cnti - 1] =  cgammax[cnti] * (Ax * cbeta[threadid][cnti] - c);
            }
            psi[0][cntj][cntk] = 0.;
            for (cnti = 0; cnti < Nx - 2; cnti ++) {
               psi[cnti + 1][cntj][cntk] = calphax[cnti] * psi[cnti][cntj][cntk] + cbeta[threadid][cnti];
            }
            psi[Nx - 1][cntj][cntk] = 0.;
         }
      }
   }
   
   return;
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluy(double complex ***psi, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Ny - 2] = psi[cnti][Ny - 1][cntk];
            for (cntj = Ny - 2; cntj > 0; cntj --) {
               c = - Ay * psi[cnti][cntj + 1][cntk] + Ay0r * psi[cnti][cntj][cntk] - Ay * psi[cnti][cntj - 1][cntk];
               cbeta[threadid][cntj - 1] =  cgammay[cntj] * (Ay * cbeta[threadid][cntj] - c);
            }
            psi[cnti][0][cntk] = 0.;
            for (cntj = 0; cntj < Ny - 2; cntj ++) {
               psi[cnti][cntj + 1][cntk] = calphay[cntj] * psi[cnti][cntj][cntk] + cbeta[threadid][cntj];
            }
            psi[cnti][Ny - 1][cntk] = 0.;
         }
      }
   }
   
   return;
}

/**
 *    Time propagation with respect to H4 (z-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluz(double complex ***psi, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;
   
   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();
      
      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            cbeta[threadid][Nz - 2] = psi[cnti][cntj][Nz - 1];
            for (cntk = Nz - 2; cntk > 0; cntk --) {
               c = - Az * psi[cnti][cntj][cntk + 1] + Az0r * psi[cnti][cntj][cntk] - Az * psi[cnti][cntj][cntk - 1];
               cbeta[threadid][cntk - 1] =  cgammaz[cntk] * (Az * cbeta[threadid][cntk] - c);
            }
            psi[cnti][cntj][0] = 0.;
            for (cntk = 0; cntk < Nz - 2; cntk ++) {
               psi[cnti][cntj][cntk + 1] = calphaz[cntk] * psi[cnti][cntj][cntk] + cbeta[threadid][cntk];
            }
            psi[cnti][cntj][Nz - 1] = 0.;
         }
      }
   }
   
   return;
}


void outdenxyz(double complex ***psi, FILE *file) {
   long cnti, cntj, cntk;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj += outstpy) {
	 for(cntk = 0; cntk < Nz; cntk += outstpz) {
	    fprintf(file, "%8le %8le %8le %8le\n", x[cnti], y[cntj], z[cntk],  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]));
// 	    fprintf(file, "%8le\n", psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk]);
	    fflush(file);
         }
      }
   }

   return;
}

void outdenx(double complex ***psi, double *tmpy, double *tmpz, FILE *file) {
   long cnti, cntj, cntk;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
         }
         tmpy[cntj] = simpint(dz, tmpz, Nz);
      }
      fprintf(file, "%8le %8le\n", x[cnti], simpint(dy, tmpy, Ny));
      fflush(file);
   }
}

void outdeny(double complex ***psi, double *tmpx, double *tmpz, FILE *file) {
   long cnti, cntj, cntk;

   for(cntj = 0; cntj < Ny; cntj += outstpy) {
       for(cnti = 0; cnti < Nx; cnti ++){
         for(cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
         }
         tmpx[cnti] = simpint(dz, tmpz, Nz);
      }
      fprintf(file, "%8le %8le\n", y[cntj], simpint(dx, tmpx, Nx));
      fflush(file);
   }
}

void outdenz(double complex ***psi, double *tmpx, double *tmpy, FILE *file) {
   long cnti, cntj, cntk;

   for(cntk = 0; cntk < Nz; cntk += outstpz) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cnti = 0; cnti < Nx; cnti ++) {
            tmpx[cnti] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
         }
         tmpy[cntj] = simpint(dx, tmpx, Nx);
      }
      fprintf(file, "%8le %8le\n", z[cntk], simpint(dy, tmpy, Ny));
      fflush(file);
   }
}

void outdenxy(double complex ***psi, double *tmpz, FILE *file) {
   long cnti, cntj, cntk;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj += outstpy) {
         for(cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
         }
         fprintf(file, "%8le %8le %8le\n", x[cnti], y[cntj], simpint(dz, tmpz, Nz));
         fflush(file);
      }
   }

   return;
}

void outdenxz(double complex ***psi, double *tmpy, FILE *file) {
   long cnti, cntj, cntk;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntk = 0; cntk < Nz; cntk += outstpz) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmpy[cntj] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
         }
         fprintf(file, "%8le %8le %8le\n", x[cnti], z[cntk], simpint(dy, tmpy, Ny));
         fflush(file);
      }
   }

   return;
}

void outdenyz(double complex ***psi, double *tmpx, FILE *file) {
   long cnti, cntj, cntk;

   for(cntj = 0; cntj < Ny; cntj += outstpy) {
      for(cntk = 0; cntk < Nz; cntk += outstpz) {
         for(cnti = 0; cnti < Nx; cnti ++) {
            tmpx[cnti] =  cabs(psi[cnti][cntj][cntk]) *  cabs(psi[cnti][cntj][cntk]);
	    fprintf(file, "%8le %8le %8le\n", y[cntj], z[cntk], simpint(dx, tmpx, Nx));
	    fflush(file);
         }
      }
   }

   return;
}

void outpsi2xz(double complex ***psi, FILE *file) {
   long cnti, cntk;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntk = 0; cntk < Nz; cntk += outstpz) {
         fprintf(file, "%8le %8le %8le\n", x[cnti], z[cntk],  cabs(psi[cnti][Ny2][cntk]) *  cabs(psi[cnti][Ny2][cntk]));
      }
      fprintf(file, "\n");
      fflush(file);
   }

   return;
}

void outpsi2xy(double complex ***psi, FILE *file) {
   long cnti, cntj;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj += outstpy) {
         fprintf(file, "%8le %8le %8le\n", x[cnti], y[cntj],  cabs(psi[cnti][cntj][Nz2]) *  cabs(psi[cnti][cntj][Nz2]));
      }
      fprintf(file, "\n");
      fflush(file);
   }

   return;
}
