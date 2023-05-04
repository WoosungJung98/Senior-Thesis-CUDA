# Senior-Thesis-CUDA

<b>Thesis</b>: CUDA Optimizations for High Performance Machine Learning

<b>Abstract</b>: CUDA is a parallel computing platform created by NVIDIA. CUDA is widely used for any application that benefits from the computing power of a GPU such as machine learning, physics simulations, and 3D rendering. For machine learning, popular publicly available libraries such as Tensorflow and PyTorch use CUDA as a framework to train models on GPUs. Last semester, I implemented a general multi-layer neural network from scratch using CUDA and C for the purpose of classifying handwritten digits from the MNIST dataset. The goals for my senior project were twofold. First, I aimed to research and learn about CUDA optimization and profiling techniques that can improve training speed by a significant margin. Second, I aimed to match or exceed the training performance of Tensorflow. My research into CUDA optimization is especially interesting and important as training time is one of the largest bottlenecks for machine learning. When training a machine learning model, the tunable hyperparameters require running multiple iterations to optimize and cannot be calculated theoretically. By reducing training time, researchers can iterate faster and ultimately create more accurate and performant models. First, I used built-in linux tools to measure CPU and GPU utilization and resolved an apparent CPU bottleneck by parallelizing code across multiple cores with OpenMP. Second, I profiled my implementation with NVIDIA's built-in profiler to determine GPU specific performance bottlenecks. Upon finding excessive memory allocation and memory copy calls, I rewrote code to cut down the number of calls thus reducing the frequency of forced GPU synchronization. Lastly, I exploited CUDA's ability to execute kernels concurrently to reduce the amount of idling resources and fully utilize the GPU's available compute capacity regardless of the workload's size or complexity. I verified my optimized implementation across multiple GPUs on Yale's HPC cluster and multiple datasets other than MNIST.