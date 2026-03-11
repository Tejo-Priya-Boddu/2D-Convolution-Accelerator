#include "conv2d_accel.h"

void conv2d_accelerator(
    const DTYPE input_feature_map[IFM_CHANNELS][IFM_SIZE][IFM_SIZE],
    const DTYPE weights[OFM_CHANNELS][IFM_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    const DTYPE biases[OFM_CHANNELS],
    DTYPE output_feature_map[OFM_CHANNELS][OFM_SIZE][OFM_SIZE]
) {
    // ----------------------------------------------------------------------------------
    // GLOBAL MEMORY OPTIMIZATIONS
    // ----------------------------------------------------------------------------------

    // 1. Array Partitioning on Weights: Splits the weight memory to allow 
    //    all 9 parallel MAC units to access weights simultaneously.
    #pragma HLS ARRAY_PARTITION variable=weights dim=4 complete

    // 2. Array Partitioning on Input Feature Map (IFM): Splits the main IFM memory
    //    to allow parallel reads for the current 3x3 window being processed.
    #pragma HLS ARRAY_PARTITION variable=input_feature_map dim=3 complete 
    
    // ----------------------------------------------------------------------------------
    // KERNEL LOOPS AND PIPELINING
    // ----------------------------------------------------------------------------------

    for (int k = 0; k < OFM_CHANNELS; k++) {
        for (int r = 0; r < OFM_SIZE; r++) {

            // 3. Pipelining: Target II=1 on the 'r' loop to maximize throughput 
            //    (start a new output row calculation every clock cycle).
            #pragma HLS PIPELINE II=1 

            for (int c = 0; c < OFM_SIZE; c++) {
                
                // Local Buffer for the current 3x3 input window
                DTYPE ifm_window[KERNEL_SIZE][KERNEL_SIZE];
                // 4. Partitioning Local Buffer: Guarantees 9 parallel read ports for the MACs.
                #pragma HLS ARRAY_PARTITION variable=ifm_window complete 
                
                // ----------------------------------------------------
                // Data Loading: Read the 3x3 window from the main IFM into the fast local buffer
                // This is where the previous bottleneck occurred.
                for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                    
                    // 5. NEW: Unroll 'kr' to parallelize the 3 row reads
                    #pragma HLS UNROLL 
                    for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                         // 6. NEW: Unroll 'kc' to parallelize the 3 column reads
                         #pragma HLS UNROLL
                         
                         int in_r = r * STRIDE + kr;
                         int in_c = c * STRIDE + kc;
                         // The read from the partitioned IFM should now be fully parallelized (9 reads)
                         ifm_window[kr][kc] = input_feature_map[0][in_r][in_c];
                    }
                }
                // ----------------------------------------------------

                DTYPE sum = biases[k];
                for (int m = 0; m < IFM_CHANNELS; m++) {
                    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                        
                        // 7. Unroll MAC: Create 3 parallel MAC operations
                        #pragma HLS UNROLL factor=3
                        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                            
                            // Core MAC Operation: Reads from the fast, partitioned local buffer
                            sum += ifm_window[kr][kc] * weights[k][m][kr][kc];
                            
                        } // end kc
                    } // end kr
                } // end m

                output_feature_map[k][r][c] = sum;
            } // end c
        } // end r
    } // end k
}