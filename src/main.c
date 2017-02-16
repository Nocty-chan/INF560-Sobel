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
    int width, height;
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
    }

    //Broadcast number of images, width and height to everybody.
    MPI_Bcast(&numberOfImages, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //fprintf(stderr, "Process n %d knows that there are %d images. \n", rankInWorld, numberOfImages);
    if (numberOfImages > totalProcesses) {
      fprintf(stderr, "Not enough processes.\n");
      return 1;
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
      picture = image->p[0];
      for (c = 1; c < r; c++) {
        sendImageToProcess(
          image->width[c],
          image->height[c],
          image->p[c],
          c * (k + 1));
      }
      for (c = r; c < numberOfImages; c++) {
        if (c == 0) continue;
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
      /*fprintf(
        stderr,
        "Process : %d is receiving width %d  and height %d of color %d.\n",
        rankInWorld,
        width,
        height,
        color);*/
      int size = width * height;
      picture = malloc(size * sizeof(pixel));
      receiveImageFromProcess(size, picture, 0);
    }

    /* Root of group applies two first filters */
    if (rankInGroup == 0) {
      /* Convert the pixels into grayscale */
      apply_gray_filter_once(picture, width * height) ;
      /* Apply blur filter with convergence value */
      apply_blur_filter_once(picture, width, height, 5, 20 ) ;
    }

    /* Dispatch height and width to all processes of the group */
    MPI_Bcast(&width, 1, MPI_INT, 0, imageCommunicator);
    MPI_Bcast(&height, 1, MPI_INT, 0, imageCommunicator);

    /* Dispatch image to the group */
    int size = width * height;
    if (rankInGroup > 0) {
      picture = malloc(size * sizeof(pixel));
    }
    broadcastImageToCommunicator(picture, size, rankInGroup, imageCommunicator);

    // Determine chunksizes and partially apply Sobel filter
    int chunksize = size / groupSize;
    int remainingChunk = size - groupSize * chunksize;
    pixel *processedChunk;
    if (rankInGroup < remainingChunk) {
      processedChunk = malloc((chunksize + 1) * sizeof(pixel));
      processedChunk = applySobelFilterFromTo(
        picture,
        width,
        height,
        rankInGroup * (chunksize + 1),
        (rankInGroup + 1) * (chunksize + 1)
      );
    } else {
      processedChunk = malloc(chunksize * sizeof(pixel));
      processedChunk = applySobelFilterFromTo(
        picture,
        width,
        height,
        rankInGroup * chunksize + remainingChunk,
        (rankInGroup + 1) * chunksize + remainingChunk
      );
    }
    //Convert processedChunk to int array.
    int *gray;
    int sizeOfChunk;
    if (rankInGroup < remainingChunk) {
      sizeOfChunk = chunksize + 1;
    } else {
      sizeOfChunk = chunksize;
    }
    gray = malloc(sizeOfChunk * sizeof(int));
    int i;
    for (i = 0; i < sizeOfChunk; i++) {
      gray[i] = processedChunk[i].g;
      if (gray[i] < 0 || gray[i] > 255) {
        gray[i] = 0;
      }
    }

    //Gather processedChunk to root.
     int *totalGray = malloc (size * sizeof(int));
     int *recvCounts = malloc (groupSize * sizeof(int));
     int *displs = malloc(groupSize * sizeof(int));
     for (i = 0; i < remainingChunk; i++) {
       recvCounts[i] = chunksize + 1;
       displs[i] = (chunksize + 1) * i;
     }
     for (i = remainingChunk; i < groupSize; i++) {
       recvCounts[i] = chunksize;
       displs[i] = chunksize * i + remainingChunk * (chunksize + 1);
     }

      MPI_Gatherv(
        gray,
        sizeOfChunk,
        MPI_INT,
        totalGray,
        recvCounts,
        displs,
        MPI_INT,
        0,
        imageCommunicator);

    //Put total gray into picture.
    if (rankInGroup == 0) {
      int i;
      for (i = 0; i < size; i++) {
        picture[i].r = totalGray[i];
        picture[i].g = totalGray[i];
        picture[i].b = totalGray[i];
      }
    }

      //Send results back to root.
    if (rankInGroup == 0) {
      if (rankInWorld != 0) {
        sendGreyImageToProcessWithTagAndSize(picture, 0, rankInWorld, width * height);
      }
    }

    if (rankInWorld == 0) {
      // Get result back from other processes.
      int c;
      for (c = 1; c < r; c++) {
        receiveGreyImageFromProcessWithTagAndSize(
          image->p[c],
          c * (k + 1),
          c * (k + 1),
          image->width[c] * image->height[c]);
      }
      for (c = r; c < numberOfImages; c++) {
        if (c == 0) continue;
        receiveGreyImageFromProcessWithTagAndSize(
          image->p[c],
          c * k + r,
          c * k + r,
          image->width[c] * image->height[c]);
      }

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
