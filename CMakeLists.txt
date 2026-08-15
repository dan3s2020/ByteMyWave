cmake_minimum_required(VERSION 3.24)

project(TensorWaveProof LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Initial target machines are Turing/Ampere (e.g. RTX 2080 Ti, RTX 3050 Ti,
# RTX 3060). Contributors can override this at configure time, for example:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86)
endif()

add_executable(tensorwave_stream_proof
    src/tensorwave_stream_proof.cu
)

target_link_libraries(tensorwave_stream_proof PRIVATE
    CUDA::cudart
    CUDA::cublas
)

add_executable(tensorwave_real_weight_proof
    src/tensorwave_real_weight_proof.cu
)

target_link_libraries(tensorwave_real_weight_proof PRIVATE
    CUDA::cudart
    CUDA::cublas
)

if(MSVC)
    target_compile_options(tensorwave_stream_proof PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
    target_compile_options(tensorwave_real_weight_proof PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else()
    target_compile_options(tensorwave_stream_proof PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
    target_compile_options(tensorwave_real_weight_proof PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()
