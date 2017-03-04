#include "load_util.h"

void apply_blur_filter( animated_gif * image, int size, int threshold );
void apply_gray_filter( animated_gif * image );
void apply_sobel_filter( animated_gif * image );

void apply_gray_filter_once(pixel *oneImage, int size);
void apply_blur_filter_once(pixel* oneImage, int width, int height, int blurSize, int threshold);
void apply_sobel_filter_once(pixel *oneImage, int width, int height);

pixel *applySobelFilterFromTo(pixel *oneImage, int width, int height, int beginIndex, int endIndex);
pixel *applyGrayFilterFromTo(pixel *oneImage, int beginIndex, int endIndex);
