#include "filters.h"
#include "dispatch_util.h"
#include <math.h>
#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

void applyFiltersDistributedInCommunicator(pixel *picture, int color, int width, int height, MPI_Comm imageCommunicator) {
  int size, rankInGroup;
  MPI_Bcast(&width, 1, MPI_INT, 0, imageCommunicator);
  MPI_Bcast(&height, 1, MPI_INT, 0, imageCommunicator);
  size = width * height;
  MPI_Comm_rank(imageCommunicator, &rankInGroup);

  //Dispatch image to all processes.
  if (rankInGroup > 0) {
    picture = (pixel *)malloc(size * sizeof(pixel));
  }
  broadcastImageToCommunicator(picture, size, rankInGroup, imageCommunicator);
  //Applying Gray Filter
  applyGrayFilterDistributedInCommunicator(picture, size, imageCommunicator);
  if (rankInGroup == 0) {
    fprintf(stderr, "Gray Filter successfully applied.\n");
    apply_blur_filter_once(picture, width, height, 5, 20);
    //fprintf(stderr, "Processed image %d on process %d \n", color, rankInWorld);
  }
  fprintf(stderr, "Sobel filter\n");
  //Apply Sobel filter.
  broadcastImageToCommunicator(picture, size, rankInGroup, imageCommunicator);
  fprintf(stderr, "Broadcast\n");
  applySobelFilterDistributedInCommunicator(
    picture,
    color,
    width,
    height,
    imageCommunicator);
}

void applySobelFilterDistributedInCommunicator(pixel *picture, int color, int width, int height, MPI_Comm imageCommunicator) {
  int groupSize, rankInGroup, size;
  MPI_Comm_rank(imageCommunicator, &rankInGroup);
  MPI_Comm_size(imageCommunicator, &groupSize);
  size = width * height;
  //Determine chunksizes for Sobel Filter
  int chunksize = size / groupSize;
  int remainingChunk = size - groupSize * chunksize;
  int sizeOfChunk;
  if (rankInGroup < remainingChunk) {
    sizeOfChunk = chunksize + 1;
  } else {
    sizeOfChunk= chunksize;
  }
  //Apply sobel filter
  int i;
  pixel *sobelChunk = applySobelFilterOnOneProcess(picture, width, height, imageCommunicator);

  //Convert processedChunk into int array.
  int *grayArray = (int *)malloc(sizeOfChunk * sizeof(int));

  for (i = 0; i < sizeOfChunk; i++) {
    grayArray[i] = sobelChunk[i].g;
  }

  free(sobelChunk);

  //Gather processedChunk to root.
   int *totalGray = (int *)malloc (size * sizeof(int));
   gatherGrayImageWithChunkSizeAndRemainingSizeInCommunicator(
     totalGray,
     grayArray,
     chunksize,
     remainingChunk,
     imageCommunicator);
  //Put total gray into picture.
  if (rankInGroup == 0) {
    greyToPixel(picture, totalGray, size);
  }

  free(grayArray);
  free(totalGray);
}
void applyGrayFilterDistributedInCommunicator(pixel *picture, int size, MPI_Comm imageCommunicator) {
  int groupSize, rankInGroup;
  MPI_Comm_rank(imageCommunicator, &rankInGroup);
  MPI_Comm_size(imageCommunicator, &groupSize);
  //Determine chunksizes for Gray Filter
  int chunksizeForGrayFilter = size / groupSize;
  int remainingChunkForGrayFilter = size - groupSize * chunksizeForGrayFilter;
  int sizeOfChunkForGrayFilter;
  if (rankInGroup < remainingChunkForGrayFilter) {
    sizeOfChunkForGrayFilter = chunksizeForGrayFilter + 1;
  } else {
    sizeOfChunkForGrayFilter = chunksizeForGrayFilter;
  }
  /*fprintf(stderr, "Chunksize %d, remaining %d\n", chunksizeForGrayFilter, remainingChunkForGrayFilter);
  fprintf(stderr, "On process %d of group %d of size %d, sizeOfChunkForGrayFilter is %d out of %d.\n", rankInGroup, color, groupSize, sizeOfChunkForGrayFilter, size);*/
  //Apply gray filter
  pixel *grayChunk = applyGrayFilterOnOneProcess(picture, size, imageCommunicator);
//  fprintf(stderr, "Applied gray filter on process %d of group %d\n", rankInGroup, color);
  //Convert processedChunk into int array.
  int *grayArray = (int *)malloc(sizeOfChunkForGrayFilter * sizeof(int));
  int i;
  for (i = 0; i < sizeOfChunkForGrayFilter; i++) {
    grayArray[i] = grayChunk[i].g;
  }
  //fprintf(stderr, "Copied into array of int on process %d of group %d\n", rankInGroup, color);
  free(grayChunk);
  //Gather processedChunk to root.
   int *totalGray = (int *)malloc (size * sizeof(int));

   gatherGrayImageWithChunkSizeAndRemainingSizeInCommunicator(
     totalGray,
     grayArray,
     chunksizeForGrayFilter,
     remainingChunkForGrayFilter,
     imageCommunicator);
  //fprintf(stderr, "Gathered array of int on process %d of group %d\n", rankInGroup, color);
  //Put total gray into picture.
  if (rankInGroup == 0) {
    greyToPixel(picture, totalGray, size);
  }

  free(grayArray);
  free(totalGray);
}

pixel *applyGrayFilterOnOneProcess(pixel *picture, int size, MPI_Comm imageCommunicator) {
  int rankInGroup, groupSize;
  MPI_Comm_rank(imageCommunicator, &rankInGroup);
  MPI_Comm_size(imageCommunicator, &groupSize);
  int chunksizeForGrayFilter = size / groupSize;
  int remainingChunkForGrayFilter = size - groupSize * chunksizeForGrayFilter;
  int sizeOfChunkForGrayFilter;
  if (rankInGroup < remainingChunkForGrayFilter) {
    sizeOfChunkForGrayFilter = chunksizeForGrayFilter + 1;
  } else {
    sizeOfChunkForGrayFilter = chunksizeForGrayFilter;
  }
  if (rankInGroup < remainingChunkForGrayFilter) {
      return applyGrayFilterFromTo(
      picture,
      rankInGroup * (chunksizeForGrayFilter + 1),
      (rankInGroup + 1) * (chunksizeForGrayFilter + 1)
    );
  } else {
      return applyGrayFilterFromTo(
      picture,
      rankInGroup * chunksizeForGrayFilter + remainingChunkForGrayFilter,
      (rankInGroup + 1) * chunksizeForGrayFilter + remainingChunkForGrayFilter
    );
  }
}

pixel *applySobelFilterOnOneProcess(pixel *picture, int width, int height, MPI_Comm imageCommunicator) {
  int rankInGroup, groupSize, size;
  MPI_Comm_rank(imageCommunicator, &rankInGroup);
  MPI_Comm_size(imageCommunicator, &groupSize);
  size = width * height;
  int chunksize = size / groupSize;
  int remainingChunk = size - groupSize * chunksize;
  int sizeOfChunk;
  if (rankInGroup < remainingChunk) {
    sizeOfChunk = chunksize + 1;
  } else {
    sizeOfChunk = chunksize;
  }
//  pixel *sobelChunk = (pixel *)malloc(sizeOfChunk * sizeof(pixel));
//  pixel *sobel;
  if (rankInGroup < remainingChunk) {
    return applySobelFilterFromTo(
      picture,
      width,
      height,
      rankInGroup * (chunksize + 1),
      (rankInGroup + 1) * (chunksize + 1)
    );
  } else {
    return applySobelFilterFromTo(
      picture,
      width,
      height,
      rankInGroup * chunksize + remainingChunk,
      (rankInGroup + 1) * chunksize + remainingChunk
    );
  }
}

pixel *applyGrayFilterFromTo(pixel *oneImage, int beginIndex, int endIndex) {
  int j;
  pixel *gray = (pixel *)malloc((endIndex - beginIndex) * sizeof(pixel));
  for ( j = beginIndex ; j < endIndex ; j++ )
  {
      int moy ;
      // moy = p[i][j].r/4 + ( p[i][j].g * 3/4 ) ;
      moy = (oneImage[j].r + oneImage[j].g + oneImage[j].b)/3 ;
      if ( moy < 0 ) moy = 0 ;
      if ( moy > 255 ) moy = 255 ;

      gray[j - beginIndex].r = moy ;
      gray[j - beginIndex].g = moy ;
      gray[j - beginIndex].b = moy ;
  }
  return gray;
}

//pixel *applyBlurFilterFromTo(pixel* oneImage, int width, int height, int beginIndex, int endIndex, int blurSize, int threshold) {}

pixel *applySobelFilterFromTo(pixel *image, int width, int height, int beginIndex, int endIndex) {
  int i, j, k;
  pixel *sobel = (pixel *)malloc((endIndex - beginIndex)* sizeof(pixel)) ;
  for(i = beginIndex; i < endIndex; i++) {
    j = i / width;
    k = i % width;
    if (CONV(j,k, width) != i) {
      fprintf(stderr, "%d - %d \n",CONV(j,k,width), i);
    }
    if (j >= 1 && j < height - 1 && k >= 1 && k < width-1) {
      int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
      int pixel_blue_so, pixel_blue_s, pixel_blue_se;
      int pixel_blue_o , pixel_blue  , pixel_blue_e ;
      float deltaX_blue ;
      float deltaY_blue ;
      float val_blue;

      pixel_blue_no = image[CONV(j-1,k-1,width)].b ;
      pixel_blue_n  = image[CONV(j-1,k  ,width)].b ;
      pixel_blue_ne = image[CONV(j-1,k+1,width)].b ;
      pixel_blue_so = image[CONV(j+1,k-1,width)].b ;
      pixel_blue_s  = image[CONV(j+1,k  ,width)].b ;
      pixel_blue_se = image[CONV(j+1,k+1,width)].b ;
      pixel_blue_o  = image[CONV(j  ,k-1,width)].b ;
      pixel_blue    = image[CONV(j  ,k  ,width)].b ;
      pixel_blue_e  = image[CONV(j  ,k+1,width)].b ;

      deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
      deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
      val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;
      if ( val_blue > 50 )
      {
          sobel[CONV(j  ,k  ,width)-beginIndex].r = 255 ;
          sobel[CONV(j  ,k  ,width)-beginIndex].g = 255 ;
          sobel[CONV(j  ,k  ,width)-beginIndex].b = 255 ;
      } else
      {
          sobel[CONV(j  ,k  ,width)-beginIndex].r = 0 ;
          sobel[CONV(j  ,k  ,width)-beginIndex].g = 0 ;
          sobel[CONV(j  ,k  ,width)-beginIndex].b = 0 ;
      }
    } else {
      sobel[CONV(j  ,k  ,width)-beginIndex].r = image[CONV(j  ,k  ,width)].b ;
      sobel[CONV(j  ,k  ,width)-beginIndex].g = image[CONV(j  ,k  ,width)].b ;
      sobel[CONV(j  ,k  ,width)-beginIndex].b = image[CONV(j  ,k  ,width)].b ;
    }
  }

  return sobel;
}

void apply_gray_filter_once(pixel *image, int size) {
 copyImageIntoImage(applyGrayFilterFromTo(image, 0, size), image, size);
}

void apply_blur_filter_once(pixel *image, int width, int height , int size, int threshold) {
  int end = 0 ;
  int n_iter = 0 ;
  int j, k;
  /* Allocate array of new pixels */
  pixel *new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;
  copyImageIntoImage(image, new, width * height);
  /* Perform at least one blur iteration */
  do
  {
      end = 1 ;
      n_iter++ ;

      /* Apply blur on top part of image (10%) */
      for(j=size; j<height/10-size; j++)
      {
          for(k=size; k<width-size; k++)
          {
              int stencil_j, stencil_k ;
              int t_r = 0 ;
              int t_g = 0 ;
              int t_b = 0 ;

              for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
              {
                  for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                  {
                      t_r += image[CONV(j+stencil_j,k+stencil_k,width)].r ;
                      t_g += image[CONV(j+stencil_j,k+stencil_k,width)].g ;
                      t_b += image[CONV(j+stencil_j,k+stencil_k,width)].b ;
                  }
              }

              new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
              new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
              new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
          }
      }

      /* Apply blur on the bottom part of the image (10%) */
      for(j=height*0.9+size; j<height-size; j++)
      {
          for(k=size; k<width-size; k++)
          {
              int stencil_j, stencil_k ;
              int t_r = 0 ;
              int t_g = 0 ;
              int t_b = 0 ;

              for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
              {
                  for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                  {
                      t_r += image[CONV(j+stencil_j,k+stencil_k,width)].r ;
                      t_g += image[CONV(j+stencil_j,k+stencil_k,width)].g ;
                      t_b += image[CONV(j+stencil_j,k+stencil_k,width)].b ;
                  }
              }

              new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
              new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
              new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
          }
      }

      for(j=1; j<height-1; j++)
      {
          for(k=1; k<width-1; k++)
          {

              float diff_r ;
              float diff_g ;
              float diff_b ;

              diff_r = (new[CONV(j  ,k  ,width)].r - image[CONV(j  ,k  ,width)].r) ;
              diff_g = (new[CONV(j  ,k  ,width)].g - image[CONV(j  ,k  ,width)].g) ;
              diff_b = (new[CONV(j  ,k  ,width)].b - image[CONV(j  ,k  ,width)].b) ;

              if ( diff_r > threshold || -diff_r > threshold
                      ||
                       diff_g > threshold || -diff_g > threshold
                       ||
                        diff_b > threshold || -diff_b > threshold
                 ) {
                  end = 0 ;
              }

              image[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
              image[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
              image[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
          }
      }
  }
  while ( threshold > 0 && !end ) ;
  // printf( "Nb iter for image %d\n", n_iter ) ;
  free (new) ;
}

void apply_sobel_filter_once(pixel *image, int width, int height) {
  int j,k;
  pixel *sobel = applySobelFilterFromTo(image, width, height, 0, width * height);
  for(j=1; j<height-1; j++)
  {
      for(k=1; k<width-1; k++)
      {
          image[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
          image[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
          image[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
      }
  }
  free(sobel);
}

void apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
      apply_gray_filter_once(p[i], image->width[i] * image->height[i]);
    }
}

void apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i;
    pixel ** p ;
    /* Get the pixels of all images */
    p = image->p ;
    /* Process all images */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        apply_blur_filter_once(p[i], image->height[i], image->width[i], size, threshold);
    }
}

void
apply_sobel_filter( animated_gif * image )
{
    int i;
    pixel ** p ;
    p = image->p ;
    for ( i = 0 ; i < image->n_images ; i++ )
    {
      apply_sobel_filter_once(p[i], image->width[i], image->height[i]);
    }
}
