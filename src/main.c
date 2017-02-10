/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <gif_lib.h>
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
    double duration ;
    int totalProcess, processRank;
    pixel *processedImage;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    //printf("Rank is %d.\n", processRank);

    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }
    if(processRank == 0) {
    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if (image == NULL || totalProcess < image->n_images) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "GIF loaded from file %s with %d image(s) in %lf s\n",
            input_filename, image->n_images, duration ) ;
    }

    // Distributing images to each group of processes.
    gettimeofday(&t1, NULL);
    processedImage = dispatchImages(processRank, totalProcess, image);
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf("%d image(s) dispatched to %d processes in %lf s.\n", image->n_images, totalProcess, duration);

    if(false) {

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
    MPI_Finalize();
    return 0 ;
}
