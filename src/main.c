/*
 * INF560
 *
 * Image Filtering Project
 */
#pragma once
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
    fprintf(stderr, "%d processes.\n", totalProcesses);
    if (rankInWorld == 0) {
      input_filename = argv[1] ;
      output_filename = argv[2] ;
      fprintf(stderr, "Treating file %s\n", input_filename);
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
         fprintf(stderr, "Not enough processes for % images, treating each image sequentially for %d processes.\n", numberOfImages, totalProcesses);
      }
      int n;
      for (n = 0; n < numberOfImages; n++) {
         if (rankInWorld == 0) {
           width = image->width[n];
           height = image->height[n];
           size = width * height;
           picture = (pixel *)malloc(size * sizeof(pixel));
           copyImageIntoImage(image->p[n], picture, size);
         }
         applyFiltersDistributedInCommunicator(picture, 0, width, height, MPI_COMM_WORLD);

         if (rankInWorld == 0) {
           //fprintf(stderr, "Sobel filter successfully Applied for image %d\n", n);
           copyImageIntoImage(picture, image->p[n], size);
         }
      }
  } else {
    //Create communicators. One communicator by image.
    //Processes are evenly distributed between images.
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
    //Send image to the root of each communicator.
    int c;
    if (rankInWorld == 0) {
      width = image->width[0];
      height = image->height[0];
      size = image->width[0] * image->height[0];
      picture = (pixel *)malloc(size * sizeof(pixel));
      copyImageIntoImage(image->p[0], picture, size);
      sendImagesToRootsOfImageCommunicator(image, k, r, numberOfImages);
    }
    if (rankInGroup == 0 && rankInWorld != 0) {
      picture = receiveImageFromRoot(&width, &height, &size);
    }

    // Processing image
    applyFiltersDistributedInCommunicator(picture, color, width, height, imageCommunicator);

    //Send results back to root.
    gatherAllImagesToRoot(picture, rankInGroup, size, image, r, k, numberOfImages);
    MPI_Comm_free(&imageCommunicator);
  }
   MPI_Barrier(MPI_COMM_WORLD);
  if (rankInWorld == 0) {
    /* FILTERS Timer stops */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "Filtering done in %lf s from process %d \n", duration, rankInWorld) ;
    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);
    //fprintf(stderr, "Attempt at storing pixels.\n");
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
