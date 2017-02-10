#include <mpi.h>
#include <unistd.h>
#include <stdio.h>

// Fill in R,G, B components arrays from pixel image.
inline void pixelToArray(pixel *image, int *red, int *green, int *blue, int size) {
  int j;
  red = malloc(size * sizeof(int));
  blue = malloc(size * sizeof(int));
  green = malloc(size * sizeof(int));
  for (j = 0; j < size; j++) {
    red[j] = image[j].r;
    blue[j] = image[j].b;
    green[j] = image[j].g;
  }
}

//Send image information to a fiven process.
inline void sendImageToProcess(int width, int height, int *red, int *green, int *blue, int dest) {
  MPI_Isend(&width, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, NULL);
  MPI_Isend(&height, 1, MPI_INT, dest, 1, MPI_COMM_WORLD, NULL);
  MPI_Isend(red, width * height, MPI_INT, dest, 2, MPI_COMM_WORLD, NULL);
  MPI_Isend(blue, width * height, MPI_INT, dest, 3, MPI_COMM_WORLD, NULL);
  MPI_Isend(green, width * height, MPI_INT, dest, 4, MPI_COMM_WORLD, NULL);
}

//Receives image from root process.
inline void receiveImage(int *width, int *height, pixel *image) {
  int W, H;
  MPI_Recv(&W, 1, MPI_INT, 0, 0 , MPI_COMM_WORLD, NULL);
  MPI_Recv(&H, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL);
  int size = W * H;
  int *red = malloc(size * sizeof(int));
  int *blue = malloc(size * sizeof(int));
  int *green = malloc(size * sizeof(int));
  image = malloc(size * sizeof(pixel));
  MPI_Recv(red, size, MPI_INT, 0, 2 , MPI_COMM_WORLD, NULL);
  MPI_Recv(blue, size, MPI_INT, 0, 3, MPI_COMM_WORLD, NULL);
  MPI_Recv(green, size, MPI_INT, 0, 4, MPI_COMM_WORLD, NULL);
  int i;
  for (i = 0; i < size; i++) {
    pixel p = {red[i], green[i], blue[i]};
    image[i] = p;
  }
  *width = W;
  *height = H;
}

// Dispatch Images of a gif from root process to other processes and returns the received image.
inline pixel *dispatchImages(int processRank, int totalProcess, animated_gif *image) {
  if (processRank == 0) {
  int numberOfImages = image->n_images;
  int numberOfProcessesPerImage = totalProcess / numberOfImages;
  int remainingProcesses = totalProcess - numberOfProcessesPerImage * numberOfImages;
  fprintf(stderr, "Number of Processes Per Image is : %d. \n", numberOfProcessesPerImage);
  fprintf(stderr, "Remaining processes : %d. \n", remainingProcesses);

  int i,j;
  int *red, *green, *blue;
  int processCount = 0;
  // Sending images to processes
  for (i = 0; i < remainingProcesses; i++) {
  int size = image->width[i] * image->height[i];

    /* Copying image pixels into three arrays */
    pixelToArray(image->p[i], red, green, blue, size);
    /* Sending information to processes */
    for (j = 0; j < numberOfProcessesPerImage + 1; j++) {
     if (processCount == 0) {processCount ++; continue;}
      fprintf(stderr, "Sending image number %d to process %d.\n", i, processCount);
      sendImageToProcess(image->width[i], image->height[i], red, green, blue, processCount);
    }
  }
  for (i = remainingProcesses; i < numberOfImages; i++) {
  int size = image->width[i] * image->height[i];
    /* Copying image pixels into three arrays */
  pixelToArray(image->p[i], red, green, blue, size);
    /* Sending information to processes */
    for (j = 0; j < numberOfProcessesPerImage; j++) {
     if (processCount == 0) {processCount ++; continue;}
      fprintf(stderr, "Sending image number %d to process %d.\n", i, processCount);
      sendImageToProcess(image->width[i], image->height[i], red, green, blue, processCount);
    }
  }
  return image->p[0];
  } else {
    pixel *result;
    int width, height;
    receiveImage(&width, &height, result);
    fprintf(stderr, "Process number : %d received width %d and height %d. \n", processRank, width, height);
    return result;
  }
}
