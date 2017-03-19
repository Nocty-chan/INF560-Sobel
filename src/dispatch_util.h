#ifndef DISPATCH
#define DISPATCH

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)
/* ALL METHODS FOR COMMUNICATING WITH OTHER PROCESSES */

//Copies image into another one.
/* Arguments:
pixel *srcImage (input): image to be copied.
pixel *dstImage (output): destination image.
int size (input): size of image
*/
inline void copyImageIntoImage(pixel *srcImage, pixel *dstImage, int size) {
  int i;
  for (i = 0; i < size; i++) {
    dstImage[i].r = srcImage[i].r;
    dstImage[i].g = srcImage[i].g;
    dstImage[i].b = srcImage[i].b;
  }
}

// Fill in R, G, B components arrays from pixel image.
/* Arguments:
 pixel *image (input): image to be decomposed.
 int *red (output): red component of image.
 int *green (output): green component of image.
 int *blue (output): blue component of image.
 int size (input): size of image.
*/
inline void pixelToArray(pixel *image, int *red, int *green, int *blue, int size) {
  int j;
  for (j = 0; j < size; j++) {
    red[j] = image[j].r;
    blue[j] = image[j].b;
    green[j] = image[j].g;
  }
}

//Fill in picture from grey array.
/* Arguments:
 pixel *picture (output): output image.
 int *totalGray (input): grey image.
 int size (input): size of image
 */
inline void greyToPixel(pixel *picture, int *totalGray, int size) {
  int i;
  for (i = 0; i < size; i++) {
    //if (totalGray[i] >= 0 && totalGray[i] <= 255) {
      picture[i].r = totalGray[i];
      picture[i].g = totalGray[i];
      picture[i].b = totalGray[i];
  //  }
  }
}

//Send image information to a given process.
/* Arguments:
int width (input): width of image.
int height(input): height of image.
pixel *image (input): image to be sent.
int dest (input): ID of destination process.
*/
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

//Send grey Image to a process knowing the size.
/* Arguments:
pixel *image (input): grey image to be sent.
int dest (input): ID of destination process.
int tag (input): tag of communication.
int size (input): size of image.
*/

inline void sendGreyImageToProcessWithTagAndSize(pixel *image, int dest, int tag, int size) {
  MPI_Request request;
  int *grey = malloc(size * sizeof(int));
  int i;
  for (i = 0; i < size; i++) {
    grey[i] = image[i].r;
  }
  MPI_Isend(grey, size, MPI_INT, dest, tag, MPI_COMM_WORLD, &request);
}

//Receives width and height from source process.
/* Arguments:
  int *width (output): received width.
  int *height (output): received height.
  int src (input): ID of source process.
*/
inline void receiveWidthAndHeightFromProcess(int *width, int *height, int src) {
  MPI_Status status;
  MPI_Recv(width, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(height, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
}

//Receive image of a certain size from source process.
/*
* Arguments:
  int size: size of image (input)
  pixel *picture (output): received image
  int src (input): ID of the source process
*/
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

/* Receives a grey image from source process
* Arguments:
* pixel *image(output): received image
* int src (input): ID of source process
* int tag (input): tag of the communication
* int size (input): size of image
*/
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

/* Receives grey images of a gif from processes
* arguments:
* animated_gif *image (output): received images gathered in a gif.
* int r (input): number of processes that have size k + 1;
* int k (input): number of processes that process one image
* int numberOfImages (input): number of images that need to be gathered
*/
inline void receiveGreyImageFromAllProcessesWithSize(animated_gif *image, int r, int k, int numberOfImages) {
  int c;
  for (c = 1; c < r; c++) {
    //fprintf(stderr, "Receiving image %d of size %d.\n", c, image->width[c] * image->height[c]);
    receiveGreyImageFromProcessWithTagAndSize(
      image->p[c],
      c * (k + 1),
      c * (k + 1),
      image->width[c] * image->height[c]);

  }

  for (c = r; c < numberOfImages; c++) {
    if (c == 0) continue;
    //fprintf(stderr, "Receiving image %d of size %d.\n", c, image->width[c] * image->height[c]);
    receiveGreyImageFromProcessWithTagAndSize(
      image->p[c],
      c * k + r,
      c * k + r,
      image->width[c] * image->height[c]);
  }
}

/* Broadcasts an image within a communicator
* Arguments:
pixel *picture (input/output): if root, contains the image to be broadcasted, if not root, received image.
int size: size of image
int rankInGroup: rank of the calling process in the communicator
MPI_Comm imageCommunicator: communicator in which image has to be broadcasted
*/
inline void broadcastImageToCommunicator(pixel *picture, int size, int rankInGroup, MPI_Comm imageCommunicator) {
  int *red = malloc(size * sizeof (int));
  int *blue = malloc(size * sizeof (int));
  int *green = malloc(size * sizeof (int));
  if (rankInGroup == 0) {
    pixelToArray(picture, red, green, blue, size);
    int i;
  /*  for (i = 0; i < size; i++) {
      fprintf(stderr, "Red %d: %d\n", i, red[i]);
      fprintf(stderr, "Blue %d: %d\n", i, blue[i]);
      fprintf(stderr, "Green %d: %d\n", i, green[i]);
    }*/
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

/* Gathers small parts of a grey image into an entire grey image at root of the communicator
* Arguments:
  int *grayResult (output): significant at root, result grey image.
  int *graySend (input): small part of the gray image to be gathered at root.
  int chunkSize (input): size of the small image part.
  int remainingSize (input): number of processes which hold a part with a larger size.
  MPI_Comm immageCommunicator (input): communicator
*/
inline void gatherGrayImageWithChunkSizeAndRemainingSizeInCommunicator(
    int *grayResult,
    int *graySend,
    int chunkSize,
    int remainingSize,
    MPI_Comm imageCommunicator) {

  int i, groupSize, rankInGroup, actualSize;
  MPI_Comm_size(imageCommunicator, &groupSize);
  MPI_Comm_rank(imageCommunicator, &rankInGroup);
  if (rankInGroup < remainingSize) {
    actualSize = chunkSize + 1;
  } else {
    actualSize = chunkSize;
  }
  int *recvCounts = malloc (groupSize * sizeof(int));
  int *displs = malloc (groupSize * sizeof(int));
  if (rankInGroup == 0) {
    for (i = 0; i < remainingSize; i++) {
      recvCounts[i] = chunkSize + 1;
      displs[i] = (chunkSize + 1) * i;
      //fprintf(stderr, "Index i: %d, recvCounts : %d, displs: %d\n",i, recvCounts[i], displs[i]);
    }
    for (i = remainingSize; i < groupSize; i++) {
      recvCounts[i] = chunkSize;
      displs[i] = chunkSize * (i - remainingSize) + remainingSize * (chunkSize + 1);
    //  fprintf(stderr, "Index i: %d, recvCounts : %d, displs: %d\n", i, recvCounts[i], displs[i]);
    }
  }
  /*fprintf(stderr, "displacements and counts\n");
  fprintf(stderr, "graySend : %p.\n", graySend);
  fprintf(stderr, "grayResult : %p.\n", grayResult);
  fprintf(stderr, "recvCounts : %p.\n", recvCounts);
  fprintf(stderr, "displs : %p.\n", displs);*/
  MPI_Gatherv(
     graySend,
     actualSize,
     MPI_INT,
     grayResult,
     recvCounts,
     displs,
     MPI_INT,
     0,
     imageCommunicator);
 //fprintf(stderr, "Gathered \n");
 if (rankInGroup == 0) {
   free(displs);
   free(recvCounts);
 }
 
}

/* Scatter images of the gif to the roots of the communicator.
* Arguments:
* animated_gif *image (input): gif to be scattered.
* int k (input): number of processes assigned to one image.
* int r (input): number of images that are assigned to k + 1 processes.
* int numberOfImages (input): number of images in a gif.
*/
inline void sendImagesToRootsOfImageCommunicator(animated_gif *image, int k, int r, int numberOfImages) {
  int c;
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

/* Receives an image from root
* Arguments:
* int *width (output): width of the received image.
* int *height (output): height of the received image.
* int *size (output): size of the received image.
* Returns pixel* which is the image received by the process.
*/
inline pixel *receiveImageFromRoot(int *width, int *height, int *size) {
  receiveWidthAndHeightFromProcess(width, height, 0);
  *size = (*width) * (*height);
//  fprintf(stderr, "Receiving image %d of size %d.\n", color, size);
  pixel *picture = (pixel *)malloc((*size) * sizeof(pixel));
  receiveImageFromProcess(*size, picture, 0);
  return picture;
}

/* Gathers all images to root in a single gif.
* Arguments:
* pixel *picture(input): image to be sent to root.
* int rankInGroup(input): rank of the process in an image communicator.
* int size (input): size of the image to be sent.
* animated_gif *image (output): gif that will gather all images.
* int r (input): number of images that have k + 1 processes.
* int k (input): number of processed for one image.
* int numberOfImages(input): number of images in a gif.
*/
inline void gatherAllImagesToRoot(pixel *picture, int rankInGroup, int size, animated_gif *image, int r, int k, int numberOfImages) {
  int rankInWorld;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankInWorld);
  if (rankInGroup == 0 && rankInWorld != 0) {
    sendGreyImageToProcessWithTagAndSize(picture, 0, rankInWorld, size);
  }
  if (rankInWorld == 0) {
    copyImageIntoImage(picture, image->p[0], size);
    receiveGreyImageFromAllProcessesWithSize(image, r, k, numberOfImages);
  }
}

/* Transposes an image
* Arguments:
* pixel *image(input/output): transposed image.
* int size: size of image.
* int width: width of image.
* int height: height of image.
*/
inline void transposeImage(pixel *image, int size, int width, int height) {
  pixel *copyImage = (pixel *)malloc(size * sizeof(pixel));
  copyImageIntoImage(image, copyImage, size);
  int j,k;
  for (j = 0; j < height; j++) {
    for (k = 0; k < width; k++) {
      image[CONV(k,j,height)].r = copyImage[CONV(j,k,width)].r;
      image[CONV(k,j,height)].g = copyImage[CONV(j,k,width)].g;
      image[CONV(k,j,height)].b = copyImage[CONV(j,k,width)].b;
    }
  }
  free(copyImage);
}

/* Transposes array of int*/

inline void transposePixelArray(int *pixels, int size, int width, int height) {
	int *copyImage = (int *)malloc(size * sizeof(int));
	int i;
	for (i = 0; i < size; i++) {
		copyImage[i] = pixels[i];
	}
	int j, k;
	for (j = 0; j < height; j++) {
		for (k = 0; k < width; k++) {
			pixels[k * height + j] = copyImage[j * width + k];
		}
	}
	free(copyImage);
}

/* Copy red component of image into int array */

inline void copyRedComponent(pixel* image, int width, int height, int* pixels) {
	//fprintf(stderr, " Copying red component \n");
	int j, k;
	for (j = 0; j < height; j++) {
		for (k = 0; k < width; k++) {
			pixels[CONV(j, k, width)] = 
				(image[CONV(j, k, width)].r + 
				image[CONV(j, k, width)].g + 
				image[CONV(j, k, width)].b) /3;
		}
	}
        //fprintf(stderr, " Copied red component. \n");
}
#endif
