/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <mpi.h>
#include "filters.h"
#include <mpi.h>
#include <unistd.h>
#include "dispatch_util.h"

int main( int argc, char ** argv )
{
    char * input_filename ;
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    int rankInWorld, totalProcesses, numberOfImages;
    double duration;
    int width, height;
    MPI_Init(&argc, &argv);
    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rankInWorld);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    if (rankInWorld == 0) {
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
      numberOfImages = image->n_images;
      width = image->width[0];
      height = image->height[0];
    }

    //Broadcast number of images, width and height to everybody.
    MPI_Bcast(&numberOfImages, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //fprintf(stderr, "Process n %d knows that there are %d images. \n", rankInWorld, numberOfImages);
    if (numberOfImages > totalProcesses) {
      fprintf(stderr, "Not enough processes.\n");
      return 1;
    }
    //Create communicators
    MPI_Comm imageCommunicator;
    int k = totalProcesses / numberOfImages;
    int r = totalProcesses - k * numberOfImages;
    int color;
    if (rankInWorld <= (r - 1) * (k + 1) + k) {
      color = rankInWorld / (k + 1);
    } else {
      color = (rankInWorld - r) / k;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, rankInWorld, &imageCommunicator);
    int rankInGroup;
    MPI_Comm_rank(imageCommunicator, &rankInGroup);
    //fprintf(stderr, "Process  %d has been assigned to group %d with local rank %d. \n",rankInWorld, color, rankInGroup);

    //Send width and height of image to the root of each group.
    int c;
    MPI_Status status;
    MPI_Request request;
    if (rankInWorld == 0) {
      for (c = 1; c < r; c++) {
          MPI_Isend(&image->width[c], 1, MPI_INT, c * (k + 1), 0, MPI_COMM_WORLD, &request);
          MPI_Isend(&image->height[c], 1, MPI_INT, c * (k + 1), 1, MPI_COMM_WORLD, &request);
          fprintf(
            stderr,
            "Sending Width: %d and height: %d for image %d to process %d. \n ",
            image->width[c],
            image->height[c],
            c,
            c * (k + 1));
      }
      for (c = r; c < numberOfImages; c++) {
        if (c == 0) continue;
        MPI_Isend(&image->width[c], 1, MPI_INT, c * k + r, 0, MPI_COMM_WORLD, &request);
        MPI_Isend(&image->height[c], 1, MPI_INT, c * k + r, 1, MPI_COMM_WORLD, &request);
        fprintf(
          stderr,
          "Sending Width: %d and height: %d for image %d to process %d.\n ",
          image->width[c],
          image->height[c],
          c,
          c * k + r);
      }
    }

    if (rankInGroup == 0 && rankInWorld != 0) {
      MPI_Recv(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&height, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      fprintf(
        stderr,
        "Process : %d is receiving width %d  and height %d of color %d.\n",
        rankInWorld,
        width,
        height,
        color);
    }

    if (false) {
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
    MPI_Comm_free(&imageCommunicator);
    MPI_Finalize();
    return 0 ;
}
