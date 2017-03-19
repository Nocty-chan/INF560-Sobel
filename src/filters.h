#include "load_util.h"
#include "mpi.h"

/* ALL METHODS FOR APLLYING FILTERS ON IMAGES */

/* Applies filters on all images of the gif. */
void apply_blur_filter( animated_gif * image, int size, int threshold );
void apply_gray_filter( animated_gif * image );
void apply_sobel_filter( animated_gif * image );

/* Applies filter on one image */
void apply_gray_filter_once(pixel *oneImage, int size);
void apply_blur_filter_once(pixel* oneImage, int width, int height, int blurSize, int threshold);
void apply_sobel_filter_once(pixel *oneImage, int width, int height);

/* Applies filter on part of one image */
int *applySobelFilterFromTo(int *oneImage, int width, int height, int beginIndex, int endIndex);
pixel *applyGrayFilterFromTo(pixel *oneImage, int beginIndex, int endIndex);
int *oneBlurIterationFromTo(int *oneImage, int beginColumn, int endColumn, int width, int height, int size);

/* Applies filter from one process */
pixel *applyGrayFilterOnOneProcess(pixel *picture, int size, MPI_Comm imageCommunicator);
int *applySobelFilterOnOneProcess(int *picture, int width, int height, MPI_Comm imageCommunicator);
int *oneBlurIterationOnOneProcess(int *picture, int width, int height, MPI_Comm imageCommunicator);

/* Applies filter on one image using processes of a communicator */
void applyGrayFilterDistributedInCommunicator(pixel *picture, int size, MPI_Comm imageCommunicator);
void applySobelFilterDistributedInCommunicator(int *picture, int color, int width, int height, MPI_Comm imageCommunicator);
void applyBlurFilterDistributedInCommunicator(int *picture, int width, int height, int blurSize, int threshold, MPI_Comm imageCommunicator);

/* Applies one iteration of the blur filter in a distributed way in the communicator. Returns 0 if one more iteration is needed.  */
int OneBlurIterationDistributedInCommunicator(int *picture, int width, int height, int blurSize, int threshold, MPI_Comm imageCommunicator);

/* Applies filters on one image using processes of a communicator */
void applyFiltersDistributedInCommunicator(pixel *picture, int color, int width, int height, MPI_Comm imageCommunicator);
