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
pixel *applySobelFilterFromTo(pixel *oneImage, int width, int height, int beginIndex, int endIndex);
pixel *applyGrayFilterFromTo(pixel *oneImage, int beginIndex, int endIndex);
pixel *oneBlurIterationFromTo(pixel *oneImage, int beginColumn, int endColumn, int width, int height, int size);

/* Applies filter from one process */
pixel *applyGrayFilterOnOneProcess(pixel *picture, int size, MPI_Comm imageCommunicator);
pixel *applySobelFilterOnOneProcess(pixel *picture, int width, int height, MPI_Comm imageCommunicator);
pixel *oneBlurIterationOnOneProcess(pixel *picture, int width, int height, MPI_Comm imageCommunicator);

/* Applies filter on one image using processes of a communicator */
void applyGrayFilterDistributedInCommunicator(pixel *picture, int size, MPI_Comm imageCommunicator);
void applySobelFilterDistributedInCommunicator(pixel *picture, int color, int width, int height, MPI_Comm imageCommunicator);
void applyBlurFilterDistributedInCommunicator(pixel *picture, int width, int height, int blurSize, MPI_Comm imageCommunicator);
/* Applies filters on one image using processes of a communicator */
void applyFiltersDistributedInCommunicator(pixel *picture, int color, int width, int height, MPI_Comm imageCommunicator);
