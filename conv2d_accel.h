#ifndef CONV2D_ACCEL_H
#define CONV2D_ACCEL_H

#include <ap_fixed.h>

// --- Data Type Definitions ---

// DTYPE: Our fixed-point data type
// <16, 8> means 16 total bits, 8 bits for the integer part, and 8 bits for the fractional part.
// This is much simpler and faster to implement in hardware than standard 'float'.
typedef ap_fixed<16, 8> DTYPE;

// --- Size Definitions (Keeping them small for initial testing) ---
#define IFM_CHANNELS 1      // Input Feature Map Channels (e.g., 1 for grayscale image)
#define OFM_CHANNELS 1      // Output Feature Map Channels
#define IFM_SIZE 8          // Input Feature Map size (e.g., 8x8 pixels)
#define KERNEL_SIZE 3       // Kernel/Weight size (e.g., 3x3 filter)
#define STRIDE 1            // How many pixels the kernel moves at a time
#define PADDING 0           // We will use 'VALID' convolution (no padding)

// OFM_SIZE calculation: (Input_Size - Kernel_Size) / Stride + 1
#define OFM_SIZE ((IFM_SIZE - KERNEL_SIZE) / STRIDE + 1) // 6x6 output

// --- Top Function Signature ---

// This function will be synthesized into the FPGA hardware block.
void conv2d_accelerator(
    const DTYPE input_feature_map[IFM_CHANNELS][IFM_SIZE][IFM_SIZE],
    const DTYPE weights[OFM_CHANNELS][IFM_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    const DTYPE biases[OFM_CHANNELS],
    DTYPE output_feature_map[OFM_CHANNELS][OFM_SIZE][OFM_SIZE]
);

#endif // CONV2D_ACCEL_H