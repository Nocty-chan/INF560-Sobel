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
    /* I/O variables */
    char * input_filename ;
    char * output_filename ;
    // Loaded gif.
    animated_gif * image ;
    int numberOfImages;
    /* Variables for time measures */
    struct timeval t1, t2;
    double duration;
    /* General information for MPI */
    int rankInWorld, totalProcesses;
    /* Information for image Communicator - one communicator / images */
    MPI_Comm imageCommunicator;
    int color, rankInGroup, groupSize;
    /* Information for image processed by communicator */
    int width, height, size;
    pixel *picture;
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
      /* IMPORT Timer start */
      gettimeofday(&t1, NULL);
    }

    //Broadcast number of images to everybody.
    MPI_Bcast(&numberOfImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //fprintf(stderr, "Process %d knows that there are %d images. \n", rankInWorld, numberOfImages);
    if (numberOfImages > totalProcesses) {
      if (rankInWorld == 0) {
         fprintf(stderr, "Not enough processes, treating each image sequentially.\n");
      }
      int n;
      for (n = 0; n < numberOfImages; n++) {
         if (rankInWorld == 0) {
           width = image->width[n];
           height = image->height[n];
         }
         MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
         MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
         size = width * height;
         //fprintf(stderr, "Broadcasting image %d of size %d.\n", color, size);
         //Applying Gray Filter
         //Dispatch image to all processes.
        picture = (pixel *)malloc(size * sizeof(pixel));
         if (rankInWorld == 0) {
           copyImageIntoImage(image->p[n], picture, size);
         }
         broadcastImageToCommunicator(picture, size, rankInWorld, MPI_COMM_WORLD);
         applyGrayFilterDistributedInCommunicator(
           picture,
           width * height,
           MPI_COMM_WORLD);

         if (rankInWorld == 0) {
           fprintf(stderr, "Gray Filter successfully applied for image %d\n", n);
           apply_blur_filter_once(picture, width, height, 5, 20);
           fprintf(stderr, "Blur Filter successfully applied for image %d\n", n);
         }
         broadcastImageToCommunicator(picture, size, rankInWorld, MPI_COMM_WORLD);
         applySobelFilterDistributedInCommunicator(
           picture,
           n,
           width,
           height,
           MPI_COMM_WORLD);
         if (rankInWorld == 0) {
           fprintf(stderr, "Sobel filter successfully Applied for image %d\n", n);
           copyImageIntoImage(picture, image->p[n], size);
         }
      }
      if (rankInWorld == 0) {
       gettimeofday(&t2, NULL);
       duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
       printf( "Filtering done in %lf s \n", duration) ;
       /* EXPORT Timer start */
       gettimeofday(&t1, NULL);
       /* Store file from array of pixels to GIF file */
       fprintf(stderr, "Attempt at exporting.\n");
       if ( !store_pixels( output_filename, image ) ) {
         fprintf(stderr, "Attempt at exporting failed.\n");
         return 1 ; }
       /* EXPORT Timer stop */
       gettimeofday(&t2, NULL);
       duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
       printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;
       return 1;
     }
  }

    //Create communicators
    int k = totalProcesses / numberOfImages;
    int r = totalProcesses - k * numberOfImages;
    if (rankInWorld <= (r - 1) * (k + 1) + k) {
      color = rankInWorld / (k + 1);
    } else {
      color = (rankInWorld - r) / k;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, rankInWorld, &imageCommunicator);
    MPI_Comm_rank(imageCommunicator, &rankInGroup);
    MPI_Comm_size(imageCommunicator, &groupSize);
    //fprintf(stderr, "Process  %d has been assigned to group %d with local rank %d. \n",rankInWorld, color, rankInGroup);
    //Send image to the root of each group.
    int c;
    if (rankInWorld == 0) {
      width = image->width[0];
      height = image->height[0];
      size = image->width[0] * image->height[0];
      picture = (pixel *)malloc(size * sizeof(pixel));
      copyImageIntoImage(image->p[0], picture, size);
      for (c = 1; c < r; c++) {
        //fprintf(stderr, "Sending image %d of size %d.\n", c, image->width[c] * image->height[c]);
        sendImageToProcess(
          image->width[c],
          image->height[c],
          image->p[c],
          c * (k + 1));
      }
      for (c = r; c < numberOfImages; c++) {
        if (c == 0) continue;
        //fprintf(stderr, "Sending image %d of size %d.\n", c, image->width[c] * image->height[c]);
        sendImageToProcess(
          image->width[c],
          image->height[c],
          image->p[c],
          c * k + r);
      }
    }
    //Receive image from root.
    if (rankInGroup == 0 && rankInWorld != 0) {
      receiveWidthAndHeightFromProcess(&width, &height, 0);
      size = width * height;
    //  fprintf(stderr, "Receiving image %d of size %d.\n", color, size);
      picture = malloc(size * sizeof(pixel));
      receiveImageFromProcess(size, picture, 0);
    }
    // Processing each image
    applyFiltersDistributedInCommunicator(picture, color, width, height, imageCommunicator);
    
      //Send results back to root.
    if (rankInGroup == 0) {
      if (rankInWorld != 0) {
        //fprintf(stderr, "Sending picture %d of size %d\n",color, size);
        sendGreyImageToProcessWithTagAndSize(picture, 0, rankInWorld, width * height);
      }
    }

    if (rankInWorld == 0) {
      // Get result back from other processes.
      copyImageIntoImage(picture, image->p[0], size);
      receiveGreyImageFromAllProcessesWithSize(image, r, k , numberOfImages);
      /* FILTERS Timer stops */
      gettimeofday(&t2, NULL);
      duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
      printf( "Filtering done in %lf s \n", duration) ;
      /* EXPORT Timer start */
      gettimeofday(&t1, NULL);
      fprintf(stderr, "Attempt at storing pixels.\n");
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
