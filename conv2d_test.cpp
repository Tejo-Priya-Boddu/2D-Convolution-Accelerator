#include <iostream>
#include <cstdlib>
#include "conv2d_accel.h"

// Reference function for comparison (it's the same logic, just for testing)
void conv2d_reference(
    const DTYPE input_feature_map[IFM_CHANNELS][IFM_SIZE][IFM_SIZE],
    const DTYPE weights[OFM_CHANNELS][IFM_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    const DTYPE biases[OFM_CHANNELS],
    DTYPE output_feature_map[OFM_CHANNELS][OFM_SIZE][OFM_SIZE]
) {
    // (Contains the exact same 6-nested loop logic as conv2d_accelerator)
    // We copy the logic here just for a formal comparison function.
    for (int k = 0; k < OFM_CHANNELS; k++) {
        for (int r = 0; r < OFM_SIZE; r++) {
            for (int c = 0; c < OFM_SIZE; c++) {
                DTYPE sum = biases[k];
                for (int m = 0; m < IFM_CHANNELS; m++) {
                    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                            int in_r = r * STRIDE + kr;
                            int in_c = c * STRIDE + kc;
                            sum += input_feature_map[m][in_r][in_c] * weights[k][m][kr][kc];
                        }
                    }
                }
                output_feature_map[k][r][c] = sum;
            }
        }
    }
}


int main() {
    // 1. Initialize data arrays
    DTYPE input_fm[IFM_CHANNELS][IFM_SIZE][IFM_SIZE];
    DTYPE weights[OFM_CHANNELS][IFM_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    DTYPE biases[OFM_CHANNELS];

    // Outputs:
    DTYPE output_hls[OFM_CHANNELS][OFM_SIZE][OFM_SIZE];      // Output from the HLS kernel
    DTYPE output_ref[OFM_CHANNELS][OFM_SIZE][OFM_SIZE];      // Output from the golden reference

    // 2. Populate input and weights with simple, known values for testing
    std::cout << "Initializing data..." << std::endl;
    for (int i = 0; i < IFM_SIZE; i++) {
        for (int j = 0; j < IFM_SIZE; j++) {
            input_fm[0][i][j] = DTYPE((float)(i * IFM_SIZE + j)); // Simple incremental values
        }
    }
    // Set all weights to a simple constant for easy calculation check
    for (int k = 0; k < OFM_CHANNELS; k++) {
        biases[k] = DTYPE(1.0); // Bias set to 1.0
        for (int m = 0; m < IFM_CHANNELS; m++) {
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                    weights[k][m][kr][kc] = DTYPE(0.5); // Weight set to 0.5
                }
            }
        }
    }

    // 3. Run the HLS Kernel and the Reference function
    conv2d_accelerator(input_fm, weights, biases, output_hls);
    conv2d_reference(input_fm, weights, biases, output_ref);

    // 4. Compare results (The key verification step!)
    int error_count = 0;
    for (int k = 0; k < OFM_CHANNELS; k++) {
        for (int r = 0; r < OFM_SIZE; r++) {
            for (int c = 0; c < OFM_SIZE; c++) {
                if (output_hls[k][r][c] != output_ref[k][r][c]) {
                    error_count++;
                }
            }
        }
    }

    if (error_count == 0) {
        std::cout << "\nTEST PASSED! C-Simulation (CSIM) is functionally correct." << std::endl;
        return 0;
    } else {
        std::cout << "\nTEST FAILED! Errors found: " << error_count << std::endl;
        return 1;
    }
}