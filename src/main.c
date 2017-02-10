/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <gif_lib.h>
#include "filters.h"
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>

void dispatchImages(int processRank, int totalProcess, animated_gif *image) {
  MPI_Request request;
  MPI_Status status;
  if (processRank == 0) {
  int numberOfImages = image->n_images;
  int numberOfProcessesPerImage = totalProcess / numberOfImages;
  int remainingProcesses = totalProcess - numberOfProcessesPerImage * numberOfImages;
  fprintf(stderr, "Number of Processes Per Image is : %d. \n", numberOfProcessesPerImage);
  fprintf(stderr, "Remaining processes : %d. \n", remainingProcesses);

  int i,j;
  // Sending images to processes.
  for (i = 0; i < numberOfImages; i++) {
  int size = image->width[i] * image->height[i];
  int *red = malloc(size * sizeof(int));
  int *blue = malloc(size * sizeof(int));
  int *green = malloc(size * sizeof(int));
    /* Copying image pixels into three arrays */
    for (j = 0; j < size; j++) {
      red[j] = image->p[i][j].r;
      blue[j] = image->p[i][j].b;
      green[j] = image->p[i][j].g;
    }
    /* Sending information to processes */
    for (j = 0; j < numberOfProcessesPerImage; j++) {
     if (i == 0 && j == 0) continue;
      fprintf(stderr, "Sending image number %d to process %d.\n", i, i * numberOfProcessesPerImage + j);
      MPI_Isend(&image->width[i], 1, MPI_INT, i * numberOfProcessesPerImage + j, 0, MPI_COMM_WORLD, &request);
      MPI_Isend(&image->height[i], 1, MPI_INT, i * numberOfProcessesPerImage + j, 1, MPI_COMM_WORLD, &request);
      MPI_Isend(red, size, MPI_INT, i * numberOfProcessesPerImage + j, 2, MPI_COMM_WORLD, &request);
      MPI_Isend(blue, size, MPI_INT, i * numberOfProcessesPerImage + j, 3, MPI_COMM_WORLD, &request);
      MPI_Isend(green, size, MPI_INT, i * numberOfProcessesPerImage + j, 4, MPI_COMM_WORLD, &request);
    }
  }
  /* Sending remaining info to last processes */
   int size = image->width[numberOfImages - 1] * image->height[numberOfImages - 1];
   int *red = malloc(size * sizeof(int));
   int *blue = malloc(size * sizeof(int));
   int *green = malloc(size * sizeof(int));
    /* Copying image pixels into three arrays */
    for (j = 0; j < size; j++) {
      red[j] = image->p[numberOfImages - 1][j].r;
      blue[j] = image->p[numberOfImages - 1][j].b;
      green[j] = image->p[numberOfImages - 1][j].g;
    }

    int count = (numberOfProcessesPerImage) * (numberOfImages) + 1;
    fprintf(stderr, "Processes %d.\n", count);
    while (count < totalProcess) {
      fprintf(stderr, "Sending image number %d to process %d.\n", numberOfImages - 1, count);
      MPI_Isend(&image->width[numberOfImages - 1], 1, MPI_INT, count, 0, MPI_COMM_WORLD, &request);
      MPI_Isend(&image->height[numberOfImages - 1], 1, MPI_INT, count, 1, MPI_COMM_WORLD, &request);
      MPI_Isend(red, size, MPI_INT, count, 2, MPI_COMM_WORLD, &request);
      MPI_Isend(blue, size, MPI_INT, count, 3, MPI_COMM_WORLD, &request);
      MPI_Isend(green, size, MPI_INT, count, 4, MPI_COMM_WORLD, &request);
      count++;
    }
  } else {
    int width, height;
    MPI_Recv(&width, 1, MPI_INT, 0, 0 , MPI_COMM_WORLD, &status);
    MPI_Recv(&height, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    int size = width * height;
    int *red = malloc(size * sizeof(int));
    int *blue = malloc(size * sizeof(int));
    int *green = malloc(size * sizeof(int));
    MPI_Recv(red, width * height, MPI_INT, 0, 2 , MPI_COMM_WORLD, &status);
    MPI_Recv(blue, width * height, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
    MPI_Recv(green, width * height, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
    fprintf(stderr, "Process number : %d received width d and height %d. \n", processRank, width, height);
  }
}

int main( int argc, char ** argv )
{
    char * input_filename ;
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;
    int totalProcess, processRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    //printf("Rank is %d.\n", processRank);

    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }

    if(processRank == 0) {
    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "GIF loaded from file %s with %d image(s) in %lf s\n",
            input_filename, image->n_images, duration ) ;
    }
       // Distributing images to each group of processes.
   dispatchImages(processRank, totalProcess, image);

    if(false) {

    /* GRAY_FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    apply_gray_filter( image ) ;

    /* GRAY_FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "GRAY_FILTER done in %lf s\n", duration ) ;

    /* BLUR_FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Apply blur filter with convergence value */
    apply_blur_filter( image, 5, 20 ) ;

    /* BLUR_FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "BLUR_FILTER done in %lf s\n", duration ) ;

    /* SOBEL_FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Apply sobel filter on pixels */
    apply_sobel_filter( image ) ;

    /* SOBEL_FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "SOBEL_FILTER done in %lf s\n", duration ) ;

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;
}
    MPI_Finalize();
    return 0 ;
}
