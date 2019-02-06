#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RAD_CONV_FAC (1.0f/60.0f)*(M_PI/180.0f)
#define DEG_CONV_FAC 180.0f/M_PI

/*

Compile with: nvcc -O3 -Xptxas="-v" -arch=sm_30  galaxy_distribution.cu
Run with: time ./a.out real.txt sim.txt

*/

/* ---------- Device code ---------- */

/*   
     N = number of galaxies (in one list)
     a1 = ascension (alpha) for first galaxy list
     d1 = declination (delta) for first galaxy list
     a2 = ascension for second glaxy list
     d2 = declination for second galaxy list
*/
__global__ void theta_calc(int N, float* a1, float* d1, float* a2, float* d2, int* hist) {
     
     // Thread index
     int i = blockDim.x * blockIdx.x + threadIdx.x;

     if (i < N) {
          float angle, temp;
          int j, bin_nr;
          // Each thread multiplies one value (their index) of the first galaxy list against all the values in the second galaxy list
          for (j = 0; j < N; j++) {
               temp = sinf(d1[i])*sinf(d2[j])+cosf(d1[i])*cosf(d2[j])*cosf(a1[i]-a2[j]);
               angle = acosf(fminf(temp, 1.0f));
               // Convert to degree and determine bin number
               bin_nr = floor((angle*DEG_CONV_FAC)/0.25);
               // Increment histogram
               atomicAdd(&hist[bin_nr], 1);
          }
     }
}

void debug (int* dd, int* dr, int* rr, float* omega_hist, int bins) {

      // Save histograms to files for analysis
     FILE *out_file = fopen("output.txt", "w");
     FILE *out_file2 = fopen("output2.txt", "w");
     FILE *out_file3 = fopen("output3.txt", "w");
     FILE *out_file4 = fopen("output4.txt", "w");
     
     long int dd_tot = 0;
     long int dr_tot = 0;
     long int rr_tot = 0;

     for(int k = 0; k < bins; k++) {
          fprintf(out_file, "%d\n", dd[k]);
          fprintf(out_file2, "%d\n", dr[k]);
          fprintf(out_file3, "%d\n", rr[k]);
          fprintf(out_file4, "%f\n", omega_hist[k]);
          dd_tot += dd[k];
          dr_tot += dr[k];
          rr_tot += rr[k];
     }
     
     printf("Total entries in histogram dd: %ld\n", dd_tot);
     printf("Total entries in histogram dr: %ld\n", dr_tot);
     printf("Total entries in histogram rr: %ld\n", rr_tot);
}

void omega_calc(int* dd, int* dr, int* rr, float* omega_hist, int bins) {
     
     // Calculate omega (difference between two equally big sets) with the three histograms
     for (int m = 0; m < bins; m++) {
          omega_hist[m] = (float)((float)dd[m]-2.0f*(float)dr[m]+(float)rr[m])/(float)rr[m];
          if (m < 15) {
               printf("Omega %d: %f\n", m, omega_hist[m]);
          }
     }
}

/* ---------- Host code ---------- */

int main (int agrc, char *argv[]) {

     // Allocate arrays for the values
     int N = 100000;
     float* real_values_asc;
     float* real_values_dec;
     float* sim_values_asc;
     float* sim_values_dec;

     cudaMallocManaged(&real_values_asc, N*sizeof(float)); 
     cudaMallocManaged(&real_values_dec, N*sizeof(float));      
     cudaMallocManaged(&sim_values_asc, N*sizeof(float));      
     cudaMallocManaged(&sim_values_dec, N*sizeof(float));

     // Allocate arrays for the histograms
     int bins = 180*4;
     int* dd;
     int* dr;
     int* rr;
     float* omega_hist = (float*)malloc(bins*sizeof(float));

     cudaMallocManaged(&dd, bins*sizeof(int));
     cudaMallocManaged(&dr, bins*sizeof(int));
     cudaMallocManaged(&rr, bins*sizeof(int));

     
     // Read values from the files
     FILE * file_real = fopen(argv[1], "r");
     FILE * file_sim = fopen(argv[2], "r");
     if(!file_real || !file_sim) {
          printf("Something went wrong with the file reading...\n");
          exit(-1);
     }
     
     for (int i = 0; i < 2*N; i++) { // The values are read one by one, not in pairs, so we need to double the iteration value
          if(i%2 == 0) { // First column, right ascension
               fscanf(file_real, "%f", &real_values_asc[i/2]);
               // Convert to radians
               real_values_asc[i/2] *= RAD_CONV_FAC;
               fscanf(file_sim, "%f", &sim_values_asc[i/2]);
               // Convert to radians
               sim_values_asc[i/2] *= RAD_CONV_FAC;
          }
          else { // Second column, declination
               fscanf(file_real, "%f", &real_values_dec[i/2]);
               // Convert to radians
               real_values_dec[i/2] *= RAD_CONV_FAC;
               fscanf(file_sim, "%f", &sim_values_dec[i/2]);
               // Convert to radians
               sim_values_dec[i/2] *= RAD_CONV_FAC;
          }

          // Initialize the histograms
          if(i < bins) {
               dd[i] = 0;
               dr[i] = 0;
               rr[i] = 0;
          }
     }
     
     fclose(file_real);
     fclose(file_sim);
     
     int threads_in_block = 512;
     int blocks_in_grid = (N+threads_in_block-1)/threads_in_block;

     // GPU function
     theta_calc<<<blocks_in_grid, threads_in_block>>>(N, real_values_asc, real_values_dec, real_values_asc, real_values_dec, dd);
     theta_calc<<<blocks_in_grid, threads_in_block>>>(N, real_values_asc, real_values_dec, sim_values_asc, sim_values_dec, dr);
     theta_calc<<<blocks_in_grid, threads_in_block>>>(N, sim_values_asc, sim_values_dec, sim_values_asc, sim_values_dec, rr);

     // Wait for GPU to finish
     cudaDeviceSynchronize();

     // Calculate omega values
     omega_calc(dd, dr, rr, omega_hist, bins);
     
     //debug(dd, dr, rr, omega_hist, bins);

     // Free all the things
     cudaFree(real_values_asc);
     cudaFree(real_values_dec);
     cudaFree(sim_values_asc);
     cudaFree(sim_values_dec);
     cudaFree(dd);
     cudaFree(dr);
     cudaFree(rr);
     free(omega_hist);
     
}