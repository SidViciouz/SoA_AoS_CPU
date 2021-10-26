//
//  Context_SoA_AoS.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#ifndef __CONTEXT_SOA_AOS__
#define __CONTEXT_SOA_AOS__

#define KERNEL1 "Source/Kernel/SoA_GS.cl"
#define KERNEL2 "Source/Kernel/Aos_GS.cl"

#define KERNELNAME1 "soa_gs_kernel"
#define KERNELNAME2 "aos_gs_kernel"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <FreeImage/FreeImage.h>

#include "Util/my_OpenCL_util_2_2.h"

typedef struct _Pixel_Channels {
    BYTE R, G, B, A;
} Pixel_Channels;

typedef struct _Pixel_Planes {
    BYTE *R_plane, *G_plane, *B_plane, *A_plane;
} Pixel_Planes;

typedef struct _Context {
    FREE_IMAGE_FORMAT image_format;
    unsigned int image_width, image_height, image_pitch;
    size_t image_data_bytes;

    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } input;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } output;

    Pixel_Channels *AoS_image_input, *AoS_image_output;
    Pixel_Planes SoA_image_input, SoA_image_output;

    cl_command_queue my_queue;

    cl_kernel my_kernel;

    cl_mem input_buffer_object;
    cl_mem output_buffer_object;

    size_t global_work_size[2];
    size_t local_work_size[2];

    cl_event event_for_timing;
} Context;

extern Context context;

////////////////////// Image_IO.cpp /////////////////////////////////////////
void read_input_image_from_file32(const char* filename);
void prepare_output_image(void);
void write_output_image_to_file32(const char* filename);
void prepare_SoA_input_and_output(void);
void prepare_AoS_input_and_output(void);
void convert_SoA_output_to_output_image_data(void);
void convert_AoS_output_to_output_image_data(void);


///////////// My_Image_Filtering_Codes.cpp ///////////////////////////////////
void convert_to_greyscale_image_SoA_CPU(void);
void convert_to_greyscale_image_AoS_CPU(void);
void convert_to_sobel_image_SoA_CPU(void);
void convert_to_sobel_image_AoS_CPU(void);

int initialize_OpenCL(void);
int set_local_work_size_and_kernel_arguments(void);
int run_OpenCL_kernel(void);

#endif // __CONTEXT_SOA_AOS__