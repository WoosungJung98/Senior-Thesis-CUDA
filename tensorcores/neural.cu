#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABELS 10
#define BLOCK_SZ 16
#define NUM_STREAMS 4

typedef struct mnist_label_file_header_t_ {
  uint32_t magic_number;
  uint32_t num_labels;
} __attribute__((packed)) mnist_label_file_header_t;  

typedef struct mnist_image_file_header_t_ {
  uint32_t magic_number;
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_cols;
} __attribute__((packed)) mnist_image_file_header_t;

typedef struct mnist_batch_t_ {
  FP* pixels;
  FP* labels;
  uint32_t size;
} __attribute__((packed)) mnist_batch_t;

typedef struct mnist_dataset_t_ {
  mnist_batch_t* batches;
  uint32_t num_batches;
} mnist_dataset_t;

typedef struct layer_metadata_t_ {
  FP* weights; // malloced in device (contains bias)
  FP* avg_actv_deriv; // averaged activation derivatives
  FP* outputs;
  FP* exp_shifted;
  FP* activations; // malloced in device
  FP* avg_activation;
  uint32_t n; // number of rows in A
  uint32_t p; // number of cols in A, number of rows in B
} layer_metadata_t;

typedef struct activation_deriv_args_t_ {
  FP* d_layer_output_single;
  FP* d_exp_shifted_single;
  FP* d_softmax_derivative;
  FP* d_max_val;
  FP* d_sum;
  FP* d_relu_derivative;
} activation_deriv_args_t;

typedef struct avgderiv_params_t_ {
  int idx_in_batch;
  uint32_t batch_size;
  layer_metadata_t* layer_mtdt_arr;
  uint32_t* layer_dims_arr;
  uint32_t num_layers;
  cudaStream_t* streams;
  activation_deriv_args_t* actv_deriv_args_arr;
  FP** d_activation_deriv_arr;
} avgderiv_params_t;

typedef struct cat_xentropy_args_t_ {
  FP* d_max;
  FP* d_sum_arr;
  FP* xentropy_arr;
  FP* d_xentropy_arr;
} cat_xentropy_args_t;

__device__ static float atomicMax(float* address, float val) {
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void blocked_gpu_matrixmult(FP* A, FP* B, FP* C, FP* C_activations, int n, int m, int p, bool forward_pass_flag, bool skip_bias_flag) {
  // position within submatrix
  int C_col = threadIdx.x;
  int C_row = threadIdx.y;
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
  // A_row_abs = C_row_abs
  int A_col_abs = C_col;
  int B_row_abs = C_row;
  // B_col_abs = C_col_abs

  // total number of submatrixes
  int totalSub = (p + BLOCK_SZ - 1) / BLOCK_SZ;
  int idxSub;
  int k;
  FP Cval = 0;
 
  for(idxSub = 0; idxSub < totalSub; idxSub++) {
    // Must use fixed block size as dynamic allocation is not possible
    __shared__ FP A_shr[BLOCK_SZ][BLOCK_SZ];
    __shared__ FP B_shr[BLOCK_SZ][BLOCK_SZ];
    if(C_row_abs < n && A_col_abs < p) {
      A_shr[C_row][C_col] = A[C_row_abs * p + A_col_abs];
    }
    if(B_row_abs < p && C_col_abs < m) {
      if(forward_pass_flag) {
        if(B_row_abs == 0) {
          // Bias applied
          B_shr[C_row][C_col] = 1;
        }
        else {
          B_shr[C_row][C_col] = B[(B_row_abs - 1) * m + C_col_abs];
        }
      }
      else if(skip_bias_flag)
        B_shr[C_row][C_col] = B[B_row_abs * m + C_col_abs + 1];
      else
        B_shr[C_row][C_col] = B[B_row_abs * m + C_col_abs];
    }
    // Synchronize
    __syncthreads();
    if(C_row_abs < n && C_col_abs < m) {
      // Multiply A submatrix and B submatrix
      if(idxSub == (totalSub - 1) && p % BLOCK_SZ != 0) {
        for(k = 0; k < p % BLOCK_SZ; k++)
          Cval += A_shr[C_row][k] * B_shr[k][C_col];
      }
      else {
        for(k = 0; k < BLOCK_SZ; k++)
          Cval += A_shr[C_row][k] * B_shr[k][C_col];
      }
    }
    // Synchronize
    __syncthreads();
    A_col_abs += BLOCK_SZ;
    B_row_abs += BLOCK_SZ;
  }
  if(C_row_abs < n && C_col_abs < m) {
    C[C_row_abs * m + C_col_abs] = Cval;
    if(forward_pass_flag) {
      // ReLU activation applied
      C_activations[C_row_abs * m + C_col_abs] = (Cval > 0) ? Cval : 0;
    }
  }
}

FP randn(double mu, double sigma) {
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1) {
    call = !call;
    return (mu + sigma * (double) X2);
  }
 
  do {
    U1 = -1 + ((double) rand () / RAND_MAX) * 2;
    U2 = -1 + ((double) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (FP)(mu + sigma * (double) X1);
}

uint8_t* read_binary_file(const char* filename) {
  FILE *fileptr;
  uint8_t *buffer;
  long filelen;

  fileptr = fopen(filename, "rb");
  fseek(fileptr, 0, SEEK_END);
  filelen = ftell(fileptr);
  rewind(fileptr);

  buffer = (uint8_t *)malloc(filelen * sizeof(uint8_t));
  fread(buffer, filelen, 1, fileptr);
  fclose(fileptr);
  return buffer;
}

uint32_t big_to_little_endian(uint32_t num) {
  return (
    ((num & 0xFF000000) >> 24) |
    ((num & 0x00FF0000) >>  8) |
    ((num & 0x0000FF00) <<  8) |
    ((num & 0x000000FF) << 24)
  );
}

void images_header_to_little_endian(mnist_image_file_header_t* images_header) {
  images_header->magic_number = big_to_little_endian(images_header->magic_number);
  images_header->num_images = big_to_little_endian(images_header->num_images);
  images_header->num_rows = big_to_little_endian(images_header->num_rows);
  images_header->num_cols = big_to_little_endian(images_header->num_cols);
}

void labels_header_to_little_endian(mnist_label_file_header_t* labels_header) {
  labels_header->magic_number = big_to_little_endian(labels_header->magic_number);
  labels_header->num_labels = big_to_little_endian(labels_header->num_labels);

}

void init_dataset(mnist_dataset_t* dataset, mnist_image_file_header_t* images_header, uint8_t* images_raw, mnist_label_file_header_t* labels_header, uint8_t* labels_raw, uint32_t batch_size) {
  if(images_header->num_images % batch_size == 0)
    dataset->num_batches = images_header->num_images / batch_size;
  else
    dataset->num_batches = images_header->num_images / batch_size + 1;
  dataset->batches = (mnist_batch_t*)malloc(dataset->num_batches * sizeof(mnist_batch_t));
  uint64_t image_total_pixels = images_header->num_rows * images_header->num_cols;
  uint64_t image_dataset_total_bytes = sizeof(mnist_image_file_header_t) + image_total_pixels * images_header->num_images;
  uint64_t actual_batch_size;
  int batch_idx = 0;
  int pixel_idx, label_idx;
  int raw_label_idx = sizeof(mnist_label_file_header_t);
  uint64_t i, j, k;

  for(i=sizeof(mnist_image_file_header_t); i<image_dataset_total_bytes; i+=image_total_pixels * batch_size) {
    if(i + image_total_pixels * batch_size > image_dataset_total_bytes)
      actual_batch_size = (image_dataset_total_bytes - i) / image_total_pixels;
    else
      actual_batch_size = batch_size;
    dataset->batches[batch_idx].pixels = (FP*)malloc(actual_batch_size * image_total_pixels * sizeof(FP));
    dataset->batches[batch_idx].labels = (FP*)malloc(actual_batch_size * MNIST_LABELS * sizeof(FP));
    dataset->batches[batch_idx].size = actual_batch_size;
    pixel_idx = 0;
    for(j=i; j<i + image_total_pixels; j++) {
      for(k=j; k<i + image_total_pixels * actual_batch_size; k+=image_total_pixels) {
        dataset->batches[batch_idx].pixels[pixel_idx] = images_raw[k] / (FP)255;
        pixel_idx++;
      }
    }
    label_idx = 0;
    for(k=0; k<MNIST_LABELS; k++) {
      for(j=raw_label_idx; j<raw_label_idx + actual_batch_size; j++) {
        dataset->batches[batch_idx].labels[label_idx] = labels_raw[j] == k;
        label_idx++;
      }
    }
    raw_label_idx += actual_batch_size; 
    batch_idx++;
  }
}

void free_dataset(mnist_dataset_t* dataset) {
  int batch_idx;
  for(batch_idx=0; batch_idx<dataset->num_batches; batch_idx++) {
    free(dataset->batches[batch_idx].pixels);
    free(dataset->batches[batch_idx].labels);
  }
  free(dataset->batches);
}

layer_metadata_t* init_layer_metadata_arr(uint32_t* layer_dims_arr, uint32_t num_layers, uint32_t batch_size) {
  layer_metadata_t* layer_mtdt_arr = (layer_metadata_t*)malloc(num_layers * sizeof(layer_metadata_t));
  uint32_t prev_dim, curr_dim;
  int i, k;

  layer_mtdt_arr[0].weights = NULL;
  layer_mtdt_arr[0].outputs = NULL;
  layer_mtdt_arr[0].avg_actv_deriv = NULL;
  layer_mtdt_arr[0].exp_shifted = NULL;
  cudaMalloc((void**)&layer_mtdt_arr[0].avg_activation, layer_dims_arr[0] * sizeof(FP));
  cudaMalloc((void**)&layer_mtdt_arr[0].activations, layer_dims_arr[0] * batch_size * sizeof(FP));

  for(i=1; i<num_layers; i++) {
    prev_dim = layer_dims_arr[i-1];
    curr_dim = layer_dims_arr[i];
    cudaMalloc((void**)&layer_mtdt_arr[i].weights, (prev_dim + 1) * curr_dim * sizeof(FP));
    FP* init_weights = (FP*)malloc((prev_dim + 1) * curr_dim * sizeof(FP));
    for(k=0; k<(prev_dim + 1) * curr_dim; k++) {
      if(k % (prev_dim + 1) == 0) {
        init_weights[k] = 0;
        continue;
      }
      init_weights[k] = randn(0, sqrt(2/(double) prev_dim));
    }
    cudaMemcpy(layer_mtdt_arr[i].weights, init_weights, (prev_dim + 1) * curr_dim * sizeof(FP), cudaMemcpyHostToDevice);
    free(init_weights);
    cudaMalloc((void**)&layer_mtdt_arr[i].avg_actv_deriv, curr_dim * sizeof(FP));
    cudaMalloc((void**)&layer_mtdt_arr[i].avg_activation, curr_dim * sizeof(FP));
    FP* init_zeros = (FP*)malloc(curr_dim * sizeof(FP));
    for(k=0; k<curr_dim; k++) {
      init_zeros[k] = 0;
    }
    cudaMemcpy(layer_mtdt_arr[i].avg_actv_deriv, init_zeros, curr_dim * sizeof(FP), cudaMemcpyHostToDevice);
    free(init_zeros);
    cudaMalloc((void**)&layer_mtdt_arr[i].outputs, curr_dim * batch_size * sizeof(FP));
    cudaMalloc((void**)&layer_mtdt_arr[i].exp_shifted, curr_dim * batch_size * sizeof(FP));
    cudaMalloc((void**)&layer_mtdt_arr[i].activations, curr_dim * batch_size * sizeof(FP));
    layer_mtdt_arr[i].n = curr_dim;
    layer_mtdt_arr[i].p = prev_dim + 1;
  }
  
  return layer_mtdt_arr;
}

void free_layer_metadata_arr(layer_metadata_t* layer_mtdt_arr, uint32_t num_layers) {
  int i;
  cudaFree(layer_mtdt_arr[0].avg_activation);
  cudaFree(layer_mtdt_arr[0].activations);
  for(i=1; i<num_layers; i++) {
    cudaFree(layer_mtdt_arr[i].weights);
    cudaFree(layer_mtdt_arr[i].outputs);
    cudaFree(layer_mtdt_arr[i].exp_shifted);
    cudaFree(layer_mtdt_arr[i].activations);
    cudaFree(layer_mtdt_arr[i].avg_actv_deriv);
    cudaFree(layer_mtdt_arr[i].avg_activation);
  }
  free(layer_mtdt_arr);
}

__global__ void calc_max_layer_outputs(FP* C, int n, int m, FP* d_max) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m)
    atomicMax(d_max, C[C_row_abs * m + C_col_abs]);
}

__global__ void calc_exp_shifted(FP* C, int n, int m, FP* d_max, FP* layer_outputs) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m)
    C[C_row_abs * m + C_col_abs] = ::expf(layer_outputs[C_row_abs * m + C_col_abs] - *d_max);
}

__global__ void calc_sum_exp_shifted(FP* C, int n, int m, FP* d_sum_arr, FP* d_max) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m) {
    if(C_row_abs == 0 && C_col_abs == 0 && d_max != NULL) *d_max = -INFINITY;
    atomicAdd(&d_sum_arr[C_col_abs], C[C_row_abs * m + C_col_abs]);
  }
}

__global__ void softmax_output_layer(FP* activations, FP* exp_shifted, int n, int m, FP* d_sum_arr, FP* d_xentropy_arr) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m) {
    activations[C_row_abs * m + C_col_abs] = ::fdividef(exp_shifted[C_row_abs * m + C_col_abs], d_sum_arr[C_col_abs]);
    if(C_row_abs == 0) d_xentropy_arr[C_col_abs] = 0;
  }
}

__global__ void crossentropy_output_layer(FP* C_activations, int n, int m, FP* d_labels, FP* d_sum_arr, FP* d_xentropy_arr) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m) {
    if(C_row_abs == 0) d_sum_arr[C_col_abs] = 0;
    atomicAdd(&d_xentropy_arr[C_col_abs], -1 * d_labels[C_row_abs * m + C_col_abs] * ::logf(C_activations[C_row_abs * m + C_col_abs]));
  }
}

FP* calc_categorical_xentropy(FP* output_layer_outputs, FP* output_layer_exp_shifted, FP* output_layer_activations, uint32_t batch_size, FP* d_labels, cat_xentropy_args_t* args) {
  FP* d_max = args->d_max;
  FP* d_sum_arr = args->d_sum_arr;
  FP* xentropy_arr = args->xentropy_arr;
  FP* d_xentropy_arr = args->d_xentropy_arr;
  
  dim3 Grid;
  Grid.x = (batch_size + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (MNIST_LABELS + BLOCK_SZ - 1) / BLOCK_SZ; 
  dim3 Block(BLOCK_SZ, BLOCK_SZ);

  calc_max_layer_outputs<<<Grid,Block>>>(output_layer_outputs, MNIST_LABELS, batch_size, d_max);
  calc_exp_shifted<<<Grid,Block>>>(output_layer_exp_shifted, MNIST_LABELS, batch_size, d_max, output_layer_outputs);
  calc_sum_exp_shifted<<<Grid,Block>>>(output_layer_exp_shifted, MNIST_LABELS, batch_size, d_sum_arr, d_max);
  softmax_output_layer<<<Grid,Block>>>(output_layer_activations, output_layer_exp_shifted, MNIST_LABELS, batch_size, d_sum_arr, d_xentropy_arr);
  
  crossentropy_output_layer<<<Grid,Block>>>(output_layer_activations, MNIST_LABELS, batch_size, d_labels, d_sum_arr, d_xentropy_arr);

  // cudaMemcpy(xentropy_arr, d_xentropy_arr, batch_size * sizeof(FP), cudaMemcpyDeviceToHost);

  return xentropy_arr;
}

__global__ void extract_single_example(FP* C, int n, int m, FP* C_full, int batch_size, int idx_in_batch, FP* d_max_val, FP* d_sum) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m) {
    C[C_row_abs] = C_full[C_row_abs * batch_size + idx_in_batch];
    if(d_max_val != NULL) *d_max_val = -INFINITY;
    if(d_sum != NULL) *d_sum = 0;
  }
}

__global__ void extract_avg_example(FP* C, int n, int m, FP* C_full, int batch_size) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
  int j;
  FP avg;
 
  if(C_row_abs < n && C_col_abs < m) {
    avg = 0;
    for(j=0; j<batch_size; j++)
      avg += ::fdividef(C_full[C_row_abs * batch_size + j], batch_size);
    C[C_row_abs] = avg;
  }
}

__global__ void softmax_derivative_gpu(FP* C, FP* exp_shifted, int n, int m, FP* d_sum) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
  FP Cval;
 
  if(C_row_abs < n && C_col_abs < m) {
    Cval = ::fdividef(exp_shifted[C_row_abs], *d_sum);
    Cval = Cval * (1 - Cval);
    C[C_row_abs] = Cval;
  }
}

FP* calc_softmax_derivative(FP* d_layer_outputs, int idx_in_batch, uint32_t batch_size, uint32_t layer_dim, cudaStream_t* streams, activation_deriv_args_t* args_arr) {
  uint32_t curr_thread = idx_in_batch % NUM_STREAMS;
  int k;
  
  dim3 Grid;
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  Grid.x = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (layer_dim + BLOCK_SZ - 1) / BLOCK_SZ;

  if(curr_thread == 0) {
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      extract_single_example<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_layer_output_single, layer_dim, 1, d_layer_outputs, batch_size, idx_in_batch + k, args_arr[k].d_max_val, args_arr[k].d_sum);
    }
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      calc_max_layer_outputs<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_layer_output_single, layer_dim, 1, args_arr[k].d_max_val);
    }
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      calc_exp_shifted<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_exp_shifted_single, layer_dim, 1, args_arr[k].d_max_val, args_arr[k].d_layer_output_single);
    }
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      calc_sum_exp_shifted<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_exp_shifted_single, layer_dim, 1, args_arr[k].d_sum, NULL);
    }
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      softmax_derivative_gpu<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_softmax_derivative, args_arr[k].d_exp_shifted_single, layer_dim, 1, args_arr[k].d_sum);
    }
  }
  #pragma omp barrier

  return args_arr[curr_thread].d_softmax_derivative;
}

__global__ void calc_relu_derivative_gpu(FP* C, int n, int m, FP* d_layer_output_single) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m)
    C[C_row_abs] = (d_layer_output_single[C_row_abs] > 0) ? 1 : 0;
}

FP* calc_relu_derivative(FP* d_layer_outputs, int idx_in_batch, uint32_t batch_size, uint32_t layer_dim, cudaStream_t* streams, activation_deriv_args_t* args_arr) {
  uint32_t curr_thread = idx_in_batch % NUM_STREAMS;
  int k;

  dim3 Grid;
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  Grid.x = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (layer_dim + BLOCK_SZ - 1) / BLOCK_SZ;

  if(curr_thread == 0) {
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      extract_single_example<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_layer_output_single, layer_dim, 1, d_layer_outputs, batch_size, idx_in_batch + k, NULL, NULL);
    }
    for(k=0; k<NUM_STREAMS; k++) {
      if(idx_in_batch + k >= batch_size) break;
      calc_relu_derivative_gpu<<<Grid,Block,0,streams[k]>>>(args_arr[k].d_relu_derivative, layer_dim, 1, args_arr[k].d_layer_output_single);
    }
  }
  #pragma omp barrier

  return args_arr[curr_thread].d_relu_derivative;
}

__global__ void update_weights_gpu(FP* avg_derivative, FP* avg_activation, int batch_size, FP* weights, int n, int m, FP learning_rate) {
  // position within submatrix
  int C_col = threadIdx.x;
  int C_row = threadIdx.y;
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
  FP Cval;
 
  if(C_row_abs < n && C_col_abs < m) {
    // Must use fixed block size as dynamic allocation is not possible
    __shared__ FP actv_shr[BLOCK_SZ];
    __shared__ FP deriv_shr[BLOCK_SZ];
    if(C_row == 0) {
      actv_shr[C_col] = avg_activation[C_col_abs];
    }
    if(C_col == 0)
      deriv_shr[C_row] = avg_derivative[C_row_abs];
    
    // Synchronize
    __syncthreads();

    Cval = learning_rate * deriv_shr[C_row] * actv_shr[C_col];
    weights[C_row_abs * (m + 1) + C_col_abs + 1] -= Cval;
  }
}

void update_weights(FP* d_avg_derivative, FP* d_avg_activation, int batch_size, FP* d_weights, int n, int m, FP learning_rate) {
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  dim3 Grid;
  Grid.x = (m + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (n + BLOCK_SZ - 1) / BLOCK_SZ;
  
  update_weights_gpu<<<Grid,Block>>>(d_avg_derivative, d_avg_activation, batch_size, d_weights, n, m, learning_rate);
}

__global__ void update_bias_gpu(FP* avg_derivative, int batch_size, FP* weights, int n, int m, int weights_m, FP learning_rate) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
  FP Cval;
 
  if(C_row_abs < n && C_col_abs < m) {
    Cval = learning_rate * avg_derivative[C_row_abs];
    weights[C_row_abs * weights_m] -= Cval;
  }
}

void update_bias(FP* d_avg_derivative, int batch_size, FP* d_weights, int n, int m, int weights_m, FP learning_rate) {
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  dim3 Grid;
  Grid.x = (m + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (n + BLOCK_SZ - 1) / BLOCK_SZ;
  
  update_bias_gpu<<<Grid,Block>>>(d_avg_derivative, batch_size, d_weights, n, m, weights_m, learning_rate);
}

__global__ void calc_hadamard_product(FP* d_curr_derivative, int n, int m, FP* d_next_derivative, FP* d_activation_derivative) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m) {
    d_curr_derivative[C_col_abs] = d_next_derivative[C_col_abs] * d_activation_derivative[C_col_abs];
    d_activation_derivative[C_col_abs] = 0; // Reset avg_actv_deriv in layer_mtdt_arr to 0 vector
  }
}

__global__ void accumulate_actv_deriv(FP* avg_actv_deriv, int n, int m, FP* actv_deriv, int batch_size) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m)
    atomicAdd(&avg_actv_deriv[C_col_abs], ::fdividef(actv_deriv[C_col_abs], batch_size));
}

void calc_avg_actv_deriv(void* avgderiv_params) {
  avgderiv_params_t* params = (avgderiv_params_t*)avgderiv_params;
  int idx_in_batch = params->idx_in_batch;
  uint32_t batch_size = params->batch_size;
  layer_metadata_t* layer_mtdt_arr = params->layer_mtdt_arr;
  uint32_t* layer_dims_arr = params->layer_dims_arr;
  uint32_t num_layers = params->num_layers;
  cudaStream_t* streams = params->streams;
  activation_deriv_args_t* actv_deriv_args_arr = params->actv_deriv_args_arr;
  FP** d_activation_deriv_arr = params->d_activation_deriv_arr;

  uint32_t curr_thread = idx_in_batch % NUM_STREAMS;

  dim3 Grid;
  dim3 Block(BLOCK_SZ, BLOCK_SZ);

  uint32_t curr_dim;
  int i, k;
  for(i=num_layers - 2; i>=1; i--) {
    curr_dim = layer_dims_arr[i];

    if(i == num_layers - 2)
      d_activation_deriv_arr[curr_thread] = calc_softmax_derivative(layer_mtdt_arr[i].outputs, idx_in_batch, batch_size, curr_dim, streams, actv_deriv_args_arr);
    else
      d_activation_deriv_arr[curr_thread] = calc_relu_derivative(layer_mtdt_arr[i].outputs, idx_in_batch, batch_size, curr_dim, streams, actv_deriv_args_arr);

    if(curr_thread == 0) {
      Grid.x = (curr_dim + BLOCK_SZ - 1) / BLOCK_SZ;
      Grid.y = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
      for(k=0; k<NUM_STREAMS; k++) {
        if(idx_in_batch + k >= batch_size) break;
        accumulate_actv_deriv<<<Grid,Block,0,streams[k]>>>(layer_mtdt_arr[i].avg_actv_deriv, 1, curr_dim, d_activation_deriv_arr[k], batch_size);
      }
    }
    #pragma omp barrier
  }
}

void exec_calc_avg_actv_deriv(uint32_t curr_thread, void* avgderiv_params) {
  avgderiv_params_t* params = (avgderiv_params_t*)avgderiv_params;
  uint32_t batch_size = params->batch_size;
  int j;

  for(j=0; j<batch_size; j+=NUM_STREAMS) {
    params->idx_in_batch = j + curr_thread;
    calc_avg_actv_deriv(avgderiv_params);
  }
}

void minibatch_backprop(FP* d_curr_avg_deriv, FP* d_next_avg_deriv, FP* d_xentropy_derivative, uint32_t batch_size, layer_metadata_t* layer_mtdt_arr, uint32_t* layer_dims_arr, uint32_t num_layers, FP learning_rate) {
  // Extract the average cross entropy derivative
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  dim3 Grid;
  Grid.x = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (MNIST_LABELS + BLOCK_SZ - 1) / BLOCK_SZ;
  extract_avg_example<<<Grid,Block>>>(d_curr_avg_deriv, MNIST_LABELS, 1, d_xentropy_derivative, batch_size);

  uint32_t curr_dim, next_dim;
  int i;
  for(i=num_layers - 1; i>=2; i--) {
    curr_dim = layer_dims_arr[i];
    next_dim = layer_dims_arr[i - 1];
    
    Grid.x = (next_dim + BLOCK_SZ - 1) / BLOCK_SZ;
    Grid.y = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
    blocked_gpu_matrixmult<<<Grid,Block>>>(d_curr_avg_deriv, layer_mtdt_arr[i].weights, d_next_avg_deriv, NULL, 1, next_dim, curr_dim, false, true);
 
    Grid.x = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
    Grid.y = (next_dim + BLOCK_SZ - 1) / BLOCK_SZ;
    extract_avg_example<<<Grid,Block>>>(layer_mtdt_arr[i - 1].avg_activation, next_dim, 1, layer_mtdt_arr[i - 1].activations, batch_size);
    update_weights(d_curr_avg_deriv, layer_mtdt_arr[i - 1].avg_activation, batch_size, layer_mtdt_arr[i].weights, curr_dim, next_dim, learning_rate);
    update_bias(d_curr_avg_deriv, batch_size, layer_mtdt_arr[i].weights, curr_dim, 1, next_dim + 1, learning_rate);

    Grid.x = (next_dim + BLOCK_SZ - 1) / BLOCK_SZ;
    Grid.y = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
    calc_hadamard_product<<<Grid,Block>>>(d_curr_avg_deriv, 1, next_dim, d_next_avg_deriv, layer_mtdt_arr[i - 1].avg_actv_deriv);
  }

  Grid.x = (1 + BLOCK_SZ - 1) / BLOCK_SZ;
  Grid.y = (layer_dims_arr[0] + BLOCK_SZ - 1) / BLOCK_SZ;
  extract_avg_example<<<Grid,Block>>>(layer_mtdt_arr[0].avg_activation, layer_dims_arr[0], 1, layer_mtdt_arr[0].activations, batch_size);
  update_weights(d_curr_avg_deriv, layer_mtdt_arr[0].avg_activation, batch_size, layer_mtdt_arr[1].weights, layer_dims_arr[1], layer_dims_arr[0], learning_rate);
  update_bias(d_curr_avg_deriv, batch_size, layer_mtdt_arr[1].weights, layer_dims_arr[1], 1, layer_dims_arr[0] + 1, learning_rate);
}

FP calc_accuracy(FP* activations, FP* labels, uint32_t batch_size, uint32_t* correct_cnt_arr) {
  int num_correct = 0;
  int i, j;
  int actv_idx;
  FP max_val;
  int max_i;
  for(j=0; j<batch_size; j++) {
    actv_idx = j;
    max_val = -INFINITY;
    for(i=0; i<MNIST_LABELS; i++) {
      if(activations[actv_idx] > max_val) {
        max_val = activations[actv_idx];
        max_i = i;
      }
      actv_idx += batch_size;
    }
    if((int)round(labels[max_i * batch_size + j]) == 1) {
      num_correct++;
      if(correct_cnt_arr != NULL)
        correct_cnt_arr[max_i]++;
    }
  }
  return num_correct / (FP)batch_size;
}

__global__ void calc_xentropy_derivative(FP* C, int n, int m, FP* d_labels, FP* derivative) {
  // absolute position
  int C_col_abs = threadIdx.x + BLOCK_SZ * blockIdx.x;
  int C_row_abs = threadIdx.y + BLOCK_SZ * blockIdx.y;
 
  if(C_row_abs < n && C_col_abs < m)
    derivative[C_row_abs * m + C_col_abs] = C[C_row_abs * m + C_col_abs] - d_labels[C_row_abs * m + C_col_abs];
}

void minibatch_gradient_descent(mnist_dataset_t* train_dataset, layer_metadata_t* layer_mtdt_arr, uint32_t* layer_dims_arr,
  uint32_t num_layers, FP learning_rate, FP* d_labels, cat_xentropy_args_t* xentropy_args, FP* output_layer_activations,
  FP* d_xentropy_derivative, avgderiv_params_t* avgderiv_params_arr, cudaStream_t* streams,
  activation_deriv_args_t* actv_deriv_args_arr, FP** d_activation_deriv_arr, FP* d_curr_avg_deriv, FP* d_next_avg_deriv, uint32_t curr_thread) {
 
  avgderiv_params_arr[curr_thread].layer_mtdt_arr = layer_mtdt_arr;
  avgderiv_params_arr[curr_thread].layer_dims_arr = layer_dims_arr;
  avgderiv_params_arr[curr_thread].num_layers = num_layers;
  avgderiv_params_arr[curr_thread].streams = streams;
  avgderiv_params_arr[curr_thread].actv_deriv_args_arr = actv_deriv_args_arr;
  avgderiv_params_arr[curr_thread].d_activation_deriv_arr = d_activation_deriv_arr;

  int i, j;
  uint32_t actual_batch_size;
  dim3 Grid;
  dim3 Block(BLOCK_SZ, BLOCK_SZ);
  FP* xentropy_arr;
  FP batch_train_accuracy;
  
  // Mini Batch Gradient Descent
  for(i=0; i<train_dataset->num_batches; i++) {
    actual_batch_size = train_dataset->batches[i].size; 

    // PERFORMED BY ONE THREAD
    if(curr_thread == 0) {
      cudaMemcpy(layer_mtdt_arr[0].activations, train_dataset->batches[i].pixels, layer_dims_arr[0] * actual_batch_size * sizeof(FP), cudaMemcpyHostToDevice);
      for(j=1; j<num_layers; j++) {
        Grid.x = (actual_batch_size + BLOCK_SZ - 1) / BLOCK_SZ;
        Grid.y = (layer_mtdt_arr[j].n + BLOCK_SZ - 1) / BLOCK_SZ;
        // Calculate outputs and ReLU activations for next layer
        blocked_gpu_matrixmult<<<Grid,Block>>>(layer_mtdt_arr[j].weights, layer_mtdt_arr[j - 1].activations, layer_mtdt_arr[j].outputs, layer_mtdt_arr[j].activations, layer_mtdt_arr[j].n, actual_batch_size, layer_mtdt_arr[j].p, true, false);
      }
      
      // Apply softmax to last layer and calculate categorical cross entropy loss
      cudaMemcpy(d_labels, train_dataset->batches[i].labels, MNIST_LABELS * actual_batch_size * sizeof(FP), cudaMemcpyHostToDevice);
      xentropy_arr = calc_categorical_xentropy(layer_mtdt_arr[num_layers - 1].outputs, layer_mtdt_arr[num_layers - 1].exp_shifted, layer_mtdt_arr[num_layers - 1].activations, actual_batch_size, d_labels, xentropy_args);
      
      // Calculate training accuracy for batch
      // cudaMemcpy(output_layer_activations, layer_mtdt_arr[num_layers - 1].activations, MNIST_LABELS * actual_batch_size * sizeof(FP), cudaMemcpyDeviceToHost);
      // batch_train_accuracy = calc_accuracy(output_layer_activations, train_dataset->batches[i].labels, actual_batch_size, NULL);
      // printf("Batch %d Train Accuracy: %.2f\n", i + 1, batch_train_accuracy);
      
      // Calculate cross entropy derivative (compare with labels)
      Grid.x = (actual_batch_size + BLOCK_SZ - 1) / BLOCK_SZ;
      Grid.y = (MNIST_LABELS + BLOCK_SZ - 1) / BLOCK_SZ;
      calc_xentropy_derivative<<<Grid,Block>>>(layer_mtdt_arr[num_layers - 1].activations, MNIST_LABELS, actual_batch_size, d_labels, d_xentropy_derivative);
    }
    #pragma omp barrier

    // Find average of activation derivatives for each example in batch
    avgderiv_params_arr[curr_thread].batch_size = actual_batch_size;
    exec_calc_avg_actv_deriv(curr_thread, (void*)&avgderiv_params_arr[curr_thread]);
    #pragma omp barrier

    // PERFORMED BY ONE THREAD
    if(curr_thread == 0) {
      // Backpropagation
      minibatch_backprop(d_curr_avg_deriv, d_next_avg_deriv, d_xentropy_derivative, actual_batch_size, layer_mtdt_arr,
                         layer_dims_arr, num_layers, learning_rate);
      // PRINT START
      /*printf("\n\nBatch %d Cross Entropy\n\n", i);
      for(j=0; j<actual_batch_size; j++) {
        printf("%.2f ", xentropy_arr[j]);
      }
      printf("\n");*/
      // PRINT END
    }
    #pragma omp barrier
  }
}

void train_mnist(mnist_dataset_t* train_dataset, layer_metadata_t* layer_mtdt_arr, uint32_t* layer_dims_arr, uint32_t num_layers, FP learning_rate) {
  int i, k;
  FP* output_layer_activations = (FP*)malloc(MNIST_LABELS * train_dataset->batches[0].size * sizeof(FP));
  avgderiv_params_t* avgderiv_params_arr = (avgderiv_params_t*)malloc(sizeof(avgderiv_params_t) * NUM_STREAMS);
  cudaStream_t streams[NUM_STREAMS];
  for(i=0; i<NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }
  FP* d_labels;
  cudaMalloc((void**)&d_labels, MNIST_LABELS * train_dataset->batches[0].size * sizeof(FP));
  FP* d_xentropy_derivative;
  cudaMalloc((void**)&d_xentropy_derivative, MNIST_LABELS * train_dataset->batches[0].size * sizeof(FP));

  uint32_t max_dim = 0;
  for(i=num_layers - 1; i>=2; i--) {
    if(layer_dims_arr[i - 1] > max_dim)
      max_dim = layer_dims_arr[i - 1];
  }
  FP* d_curr_avg_deriv;
  FP* d_next_avg_deriv;
  cudaMalloc((void**)&d_curr_avg_deriv, max_dim * sizeof(FP));
  cudaMalloc((void**)&d_next_avg_deriv, max_dim * sizeof(FP));

  activation_deriv_args_t* actv_deriv_args_arr = (activation_deriv_args_t*)malloc(NUM_STREAMS * sizeof(activation_deriv_args_t));
  for(i=0; i<NUM_STREAMS; i++) {
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_layer_output_single, max_dim * sizeof(FP));
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_exp_shifted_single, max_dim * sizeof(FP));
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_softmax_derivative, max_dim * sizeof(FP));
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_max_val, sizeof(FP));
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_sum, sizeof(FP));
    cudaMalloc((void**)&actv_deriv_args_arr[i].d_relu_derivative, max_dim * sizeof(FP));
  }

  cat_xentropy_args_t xentropy_args;
  cudaMalloc((void**)&xentropy_args.d_max, sizeof(FP));
  cudaMalloc((void**)&xentropy_args.d_sum_arr, train_dataset->batches[0].size * sizeof(FP));
  cudaMallocHost((void**)&xentropy_args.xentropy_arr, train_dataset->batches[0].size * sizeof(FP));
  cudaMalloc((void**)&xentropy_args.d_xentropy_arr, train_dataset->batches[0].size * sizeof(FP));
  FP max_init_val = -INFINITY;
  FP* sum_arr = (FP*)malloc(train_dataset->batches[0].size * sizeof(FP));
  for(i=0; i<train_dataset->batches[0].size; i++)
    sum_arr[i] = 0;
  cudaMemcpy(xentropy_args.d_max, &max_init_val, sizeof(FP), cudaMemcpyHostToDevice);
  cudaMemcpy(xentropy_args.d_sum_arr, sum_arr, train_dataset->batches[0].size * sizeof(FP), cudaMemcpyHostToDevice);
  free(sum_arr);

  FP** d_activation_deriv_arr = (FP**)malloc(NUM_STREAMS * sizeof(FP*));
  
  // MINI BATCH GRADIENT DESCENT
  omp_set_num_threads(NUM_STREAMS);
  #pragma omp parallel shared(train_dataset, layer_mtdt_arr, layer_dims_arr, num_layers, learning_rate, d_labels, xentropy_args, output_layer_activations, d_xentropy_derivative, avgderiv_params_arr, streams, actv_deriv_args_arr, d_activation_deriv_arr, d_curr_avg_deriv, d_next_avg_deriv) default(none)
  {
    int tid = omp_get_thread_num();
    minibatch_gradient_descent(train_dataset, layer_mtdt_arr, layer_dims_arr, num_layers, learning_rate, d_labels,
                                &xentropy_args, output_layer_activations, d_xentropy_derivative, avgderiv_params_arr,
                                streams, actv_deriv_args_arr, d_activation_deriv_arr, d_curr_avg_deriv, d_next_avg_deriv, tid);
  }

  for(i=0; i<NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  free(output_layer_activations);
  free(avgderiv_params_arr);
  cudaFree(d_labels);
  cudaFree(d_xentropy_derivative);

  cudaFree(d_curr_avg_deriv);
  cudaFree(d_next_avg_deriv);

  for(i=0; i<NUM_STREAMS; i++) {
    cudaFree(actv_deriv_args_arr[i].d_layer_output_single);
    cudaFree(actv_deriv_args_arr[i].d_exp_shifted_single);
    cudaFree(actv_deriv_args_arr[i].d_softmax_derivative);
    cudaFree(actv_deriv_args_arr[i].d_max_val);
    cudaFree(actv_deriv_args_arr[i].d_sum);
    cudaFree(actv_deriv_args_arr[i].d_relu_derivative);
  }
  free(actv_deriv_args_arr);

  cudaFree(xentropy_args.d_max);
  cudaFree(xentropy_args.d_sum_arr);
  cudaFreeHost(xentropy_args.xentropy_arr);
  cudaFree(xentropy_args.d_xentropy_arr);

  free(d_activation_deriv_arr);
}

void test_mnist(mnist_dataset_t* test_dataset, layer_metadata_t* layer_mtdt_arr, uint32_t* layer_dims_arr, uint32_t num_layers) {
  dim3 Grid;
  dim3 Block(BLOCK_SZ, BLOCK_SZ); 
  int i, j;
  uint32_t actual_batch_size;
  FP* xentropy_arr;
  FP* output_layer_activations = (FP*)malloc(MNIST_LABELS * test_dataset->batches[0].size * sizeof(FP));
  FP batch_test_accuracy;
  FP overall_test_accuracy = 0;
  FP* d_labels;
  cudaMalloc((void**)&d_labels, MNIST_LABELS * test_dataset->batches[0].size * sizeof(FP));

  cat_xentropy_args_t xentropy_args;
  cudaMalloc((void**)&xentropy_args.d_max, sizeof(FP));
  cudaMalloc((void**)&xentropy_args.d_sum_arr, test_dataset->batches[0].size * sizeof(FP));
  cudaMallocHost((void**)&xentropy_args.xentropy_arr, test_dataset->batches[0].size * sizeof(FP));
  cudaMalloc((void**)&xentropy_args.d_xentropy_arr, test_dataset->batches[0].size * sizeof(FP));
  FP max_init_val = -INFINITY;
  FP* sum_arr = (FP*)malloc(test_dataset->batches[0].size * sizeof(FP));
  for(i=0; i<test_dataset->batches[0].size; i++)
    sum_arr[i] = 0;
  cudaMemcpy(xentropy_args.d_max, &max_init_val, sizeof(FP), cudaMemcpyHostToDevice);
  cudaMemcpy(xentropy_args.d_sum_arr, sum_arr, test_dataset->batches[0].size * sizeof(FP), cudaMemcpyHostToDevice);
  free(sum_arr);

  uint32_t* correct_cnt_arr = (uint32_t*)malloc(MNIST_LABELS * sizeof(uint32_t));
  for(i=0; i<MNIST_LABELS; i++)
    correct_cnt_arr[i] = 0;
 
  for(i=0; i<test_dataset->num_batches; i++) {
    actual_batch_size = test_dataset->batches[i].size; 
    cudaMemcpy(layer_mtdt_arr[0].activations, test_dataset->batches[i].pixels, layer_dims_arr[0] * actual_batch_size * sizeof(FP), cudaMemcpyHostToDevice);
    for(j=1; j<num_layers; j++) {
      Grid.x = (actual_batch_size + BLOCK_SZ - 1) / BLOCK_SZ;
      Grid.y = (layer_mtdt_arr[j].n + BLOCK_SZ - 1) / BLOCK_SZ;
      // Calculate outputs and ReLU activations for next layer
      blocked_gpu_matrixmult<<<Grid,Block>>>(layer_mtdt_arr[j].weights, layer_mtdt_arr[j - 1].activations, layer_mtdt_arr[j].outputs, layer_mtdt_arr[j].activations, layer_mtdt_arr[j].n, actual_batch_size, layer_mtdt_arr[j].p, true, false);
    }

    // Apply softmax to the last layer and calculate categorical cross entropy loss
    cudaMemcpy(d_labels, test_dataset->batches[i].labels, MNIST_LABELS * actual_batch_size * sizeof(FP), cudaMemcpyHostToDevice);
    xentropy_arr = calc_categorical_xentropy(layer_mtdt_arr[num_layers - 1].outputs, layer_mtdt_arr[num_layers - 1].exp_shifted, layer_mtdt_arr[num_layers - 1].activations, actual_batch_size, d_labels, &xentropy_args);
    
    // Calculate test accuracy for batch
    cudaMemcpy(output_layer_activations, layer_mtdt_arr[num_layers - 1].activations, MNIST_LABELS * actual_batch_size * sizeof(FP), cudaMemcpyDeviceToHost);
    batch_test_accuracy = calc_accuracy(output_layer_activations, test_dataset->batches[i].labels, actual_batch_size, correct_cnt_arr);
    printf("Batch %d Test Accuracy: %.2f\n", i + 1, batch_test_accuracy);
    overall_test_accuracy += batch_test_accuracy;
    
    // PRINT START
    /*printf("\n\nBatch %d Cross Entropy\n\n", i);
    for(j=0; j<actual_batch_size; j++) {
      printf("%.2f ", xentropy_arr[j]);
    }
    printf("\n");*/
    // PRINT END
  }

  overall_test_accuracy /= (FP)test_dataset->num_batches;
  printf("Overall Test Accuracy: %.2f\n", overall_test_accuracy);
  printf("Correct Count:\n");
  for(i=0; i<MNIST_LABELS; i++) {
    printf("%d: %d\n", i, correct_cnt_arr[i]);
  }

  free(correct_cnt_arr);
  free(output_layer_activations);
  cudaFree(d_labels);
  cudaFree(xentropy_args.d_max);
  cudaFree(xentropy_args.d_sum_arr);
  cudaFreeHost(xentropy_args.xentropy_arr);
  cudaFree(xentropy_args.d_xentropy_arr);
}

int main(int argc, char *argv[]) {
  int i; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use 
  
  uint32_t* layer_dims_arr;

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  // --------------------SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
     printf("Device count = %d\n",gpucount);
  }
   
  if (argc < 5) {
    printf("Usage: neural <epochs> <learning rate> <batch size> <layer dim 1> <layer dim 2> ... \n");
    exit (-1);
  }
  if (argc < 6) {
    printf("Must specify input and output layer dims\n");
    exit(-1);
  }
  if (argc < 7) {
    printf("Must specify at least one hidden layer\n");
    exit(-1);
  }

  if(BLOCK_SZ * BLOCK_SZ > 1024) {
    printf("Error, too many threads in block\n");
    exit(-1);
  }
  
  int epochs = atoi(argv[1]);
  FP learning_rate = (FP)atof(argv[2]);
  uint32_t batch_size = (uint32_t)atoi(argv[3]);
  uint32_t num_layers = (uint32_t)(argc - 4);
  
  // Training Dataset Initialization 
  // uint8_t* train_images_raw = read_binary_file("../data/mnist/train-images-idx3-ubyte");
  // uint8_t* train_labels_raw = read_binary_file("../data/mnist/train-labels-idx1-ubyte");
  uint8_t* train_images_raw = read_binary_file("../data/cifar/train_images_cifar");
  uint8_t* train_labels_raw = read_binary_file("../data/cifar/train_labels_cifar");
  
  mnist_image_file_header_t train_images_header = *((mnist_image_file_header_t*)train_images_raw);
  images_header_to_little_endian(&train_images_header); 
  if(train_images_header.magic_number != MNIST_IMAGE_MAGIC) {
    printf("Train image file magic (checksum) doesn't match.\n");
    exit(-1);
  }

  mnist_label_file_header_t train_labels_header = *((mnist_label_file_header_t*)train_labels_raw);
  labels_header_to_little_endian(&train_labels_header);
  if(train_labels_header.magic_number != MNIST_LABEL_MAGIC) {
    printf("Train label file magic (checksum) doesn't match.\n");
    exit(-1);
  }

  if(train_images_header.num_images != train_labels_header.num_labels) {
    printf("Number of images has to match number of labels!\n");
    exit(-1);
  }

  mnist_dataset_t train_dataset;
  init_dataset(&train_dataset, &train_images_header, train_images_raw, &train_labels_header, train_labels_raw, batch_size);
  free(train_images_raw);
  free(train_labels_raw);

  // Test Dataset Initialization
  // uint8_t* test_images_raw = read_binary_file("../data/mnist/t10k-images-idx3-ubyte");
  // uint8_t* test_labels_raw = read_binary_file("../data/mnist/t10k-labels-idx1-ubyte");
  uint8_t* test_images_raw = read_binary_file("../data/cifar/test_images_cifar");
  uint8_t* test_labels_raw = read_binary_file("../data/cifar/test_labels_cifar");
  
  mnist_image_file_header_t test_images_header = *((mnist_image_file_header_t*)test_images_raw);
  images_header_to_little_endian(&test_images_header); 
  if(test_images_header.magic_number != MNIST_IMAGE_MAGIC) {
    printf("Train image file magic (checksum) doesn't match.\n");
    exit(-1);
  }

  mnist_label_file_header_t test_labels_header = *((mnist_label_file_header_t*)test_labels_raw);
  labels_header_to_little_endian(&test_labels_header);
  if(test_labels_header.magic_number != MNIST_LABEL_MAGIC) {
    printf("Train label file magic (checksum) doesn't match.\n");
    exit(-1);
  }

  if(test_images_header.num_images != test_labels_header.num_labels) {
    printf("Number of images has to match number of labels!\n");
    exit(-1);
  }

  mnist_dataset_t test_dataset;
  init_dataset(&test_dataset, &test_images_header, test_images_raw, &test_labels_header, test_labels_raw, batch_size);
  free(test_images_raw);
  free(test_labels_raw);
  
  layer_dims_arr = (uint32_t*)malloc(num_layers * sizeof(uint32_t));
  for(i=0; i<num_layers; i++) {
    layer_dims_arr[i] = (uint32_t)atoi(argv[i + 4]);
  }
  if(layer_dims_arr[0] != train_images_header.num_rows * train_images_header.num_cols) {
    printf("Number of nodes in first layer must match pixel count of each image in dataset.\n");
    exit(-1);
  }
  
  cudaSetDevice(gpunum);
  printf("Using device %d\n", gpunum);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  layer_metadata_t* layer_mtdt_arr = init_layer_metadata_arr(layer_dims_arr, num_layers, batch_size);
  
  // TRAIN
  for(i=0; i<epochs; i++) {
    printf("\n\n******EPOCH %d******\n\n", i+1);
    train_mnist(&train_dataset, layer_mtdt_arr, layer_dims_arr, num_layers, learning_rate);
  }

  // TEST
  printf("\n\n******TEST******\n\n");
  test_mnist(&test_dataset, layer_mtdt_arr, layer_dims_arr, num_layers);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);  

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  
  free(layer_dims_arr);
  free_layer_metadata_arr(layer_mtdt_arr, num_layers);
  free_dataset(&train_dataset);
  free_dataset(&test_dataset);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
