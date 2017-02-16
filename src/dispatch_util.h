#include <mpi.h>
#include <unistd.h>
#include <stdio.h>

// Fill in R,G, B components arrays from pixel image.
inline void pixelToArray(pixel *image, int *red, int *green, int *blue, int size) {
  int j;
  for (j = 0; j < size; j++) {
    red[j] = image[j].r;
    blue[j] = image[j].b;
    green[j] = image[j].g;
  }
}

//Send image information to a given process.
inline void sendImageToProcess(int width, int height, pixel *image, int dest) {
  MPI_Request request;
  int size = width * height;
  int *red = malloc(size * sizeof (int));
  int *blue = malloc(size * sizeof (int));
  int *green = malloc(size * sizeof (int));
  pixelToArray(image, red, green, blue, size);
  MPI_Isend(&width, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &request);
  MPI_Isend(&height, 1, MPI_INT, dest, 1, MPI_COMM_WORLD, &request);
  MPI_Isend(red, width * height, MPI_INT, dest, 2, MPI_COMM_WORLD, &request);
  MPI_Isend(blue, width * height, MPI_INT, dest, 3, MPI_COMM_WORLD, &request);
  MPI_Isend(green, width * height, MPI_INT, dest, 4, MPI_COMM_WORLD, &request);
}

inline void sendGreyImageToProcessWithTagAndSize(pixel *image, int dest, int tag, int size) {
  MPI_Request request;
  int *grey = malloc(size * sizeof(int));
  int i;
  for (i = 0; i < size; i++) {
    grey[i] = image[i].r;
  }
  MPI_Isend(grey, size, MPI_INT, dest, tag, MPI_COMM_WORLD, &request);
}

//Receives image from root process.
inline void receiveWidthAndHeightFromProcess(int *width, int *height, int src) {
  MPI_Status status;
  MPI_Recv(width, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(height, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
}
inline void receiveImageFromProcess(int size, pixel *picture, int src) {
  MPI_Status status;
  int *red, *blue, *green;
  red = malloc(size * sizeof (int));
  blue = malloc(size * sizeof (int));
  green = malloc(size * sizeof (int));
  MPI_Recv(red, size, MPI_INT, src, 2, MPI_COMM_WORLD, &status);
  MPI_Recv(blue, size, MPI_INT, src, 3, MPI_COMM_WORLD, &status);
  MPI_Recv(green, size, MPI_INT, src, 4, MPI_COMM_WORLD, &status);
  int i;
  for (i = 0; i < size; i++) {
    pixel p = {red[i], green[i], blue[i]};
    //fprintf(stderr, "Receiving, Image: %d , %d, Red: %d, Green: %d, Blue: %d\n", color, i, red[i],green[i], blue[i]);
    picture[i] = p;
  }
  free(red);
  free(blue);
  free(green);
}

inline void receiveGreyImageFromProcessWithTagAndSize(pixel *image, int src, int tag, int size) {
  MPI_Status status;
  int *grey = malloc(size * sizeof (int));
  MPI_Recv(grey, size, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
  int i;
  for (i = 0; i < size; i++) {
    image[i].r = grey[i];
    image[i].g = grey[i];
    image[i].b = grey[i];
  }
  free(grey);
}

inline void receiveGreyImageFromAllProcessesWithSize(animated_gif *image, int r, int k, int numberOfImages) {
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
}
inline void broadcastImageToCommunicator(pixel *picture, int size, int rankInGroup, MPI_Comm imageCommunicator) {
  int *red = malloc(size * sizeof (int));
  int *blue = malloc(size * sizeof (int));
  int *green = malloc(size * sizeof (int));
  if (rankInGroup == 0) {
    pixelToArray(picture, red, green, blue, size);
  }
  MPI_Bcast(red, size, MPI_INT, 0, imageCommunicator);
  MPI_Bcast(blue, size, MPI_INT, 0, imageCommunicator);
  MPI_Bcast(green, size, MPI_INT, 0, imageCommunicator);
  if (rankInGroup > 0) {
    int i;
    for (i = 1; i < size; i++) {
      pixel p = {red[i], green[i], blue[i]};
      //fprintf(stderr, "Receiving, Image: %d , %d, Red: %d, Green: %d, Blue: %d\n", color, i, red[i],green[i], blue[i]);
      picture[i] = p;
    }
  }
  free (red);
  free (blue);
  free (green);
}
