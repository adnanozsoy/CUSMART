cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(cudasmart LANGUAGES C CXX CUDA)

option(BUILD_TESTS "Build test program" OFF)

#set(CMAKE_C_FLAGS          "-Wall")
#set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -O3")

set(CMAKE_CUDA_FLAGS            "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_CUDA_FLAGS_DEBUG      "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELEASE    "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY          ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/bin)

set(MAIN_SOURCES
    src/util/argtable3.c
    src/main.c
    src/exactmatch.c
)

set(CUDA_WRAPPER_SOURCES
    src/wrappers/ac_wrapper.cu
    src/wrappers/ag_wrapper.cu
    src/wrappers/aoso2_wrapper.cu
    src/wrappers/bfbs_wrapper.cu
    src/wrappers/bfb_wrapper.cu
    src/wrappers/bfs_wrapper.cu
    src/wrappers/bf_wrapper.cu
    src/wrappers/bfst_wrapper.cu
    src/wrappers/blim_wrapper.cu
    src/wrappers/bmh_sbndm_wrapper.cu
    src/wrappers/bm_wrapper.cu
    src/wrappers/bndmq2_wrapper.cu
    src/wrappers/bndmq4_wrapper.cu
    src/wrappers/bndm_wrapper.cu
    src/wrappers/bom_wrapper.cu
    src/wrappers/br_wrapper.cu
    src/wrappers/bsdm_wrapper.cu
    src/wrappers/bww_wrapper.cu
    src/wrappers/bxs_wrapper.cu
    src/wrappers/col_wrapper.cu
    src/wrappers/dfah_wrapper.cu
    src/wrappers/dfa_wrapper.cu
    src/wrappers/dfdm_wrapper.cu
    src/wrappers/ebom_wrapper.cu
    src/wrappers/faoso2_wrapper.cu
    src/wrappers/fbom_wrapper.cu
    src/wrappers/fdm_wrapper.cu
    src/wrappers/ffs_wrapper.cu
    src/wrappers/fjs_wrapper.cu
    src/wrappers/fndm_wrapper.cu
    src/wrappers/fsbndmq20_wrapper.cu
    src/wrappers/fsbndm_wrapper.cu
    src/wrappers/fs_wrapper.cu
    src/wrappers/gg_wrapper.cu
    src/wrappers/hash3_wrapper.cu
    src/wrappers/hor_wrapper.cu
    src/wrappers/ildm1_wrapper.cu
    src/wrappers/ildm2_wrapper.cu
    src/wrappers/kbndm_wrapper.cu
    src/wrappers/kmpskip_wrapper.cu
    src/wrappers/kmp_wrapper.cu
    src/wrappers/kr_wrapper.cu
    src/wrappers/ksa_wrapper.cu
    src/wrappers/lbndm_wrapper.cu
    src/wrappers/ldm_wrapper.cu
    src/wrappers/mp_wrapper.cu
    src/wrappers/ms_wrapper.cu
    src/wrappers/nsn_wrapper.cu
    src/wrappers/om_wrapper.cu
    src/wrappers/pbmh_wrapper.cu
    src/wrappers/qs_wrapper.cu
    src/wrappers/raita_wrapper.cu
    src/wrappers/rcolussi_wrapper.cu
    src/wrappers/rf_wrapper.cu
    src/wrappers/sabp_wrapper.cu
    src/wrappers/sa_wrapper.cu
    src/wrappers/sbndm2_wrapper.cu
    src/wrappers/sbndmq2_wrapper.cu
    src/wrappers/sbndm_bmh_wrapper.cu
    src/wrappers/sbndm_wrapper.cu
    src/wrappers/sebom_wrapper.cu
    src/wrappers/sfbom_wrapper.cu
    src/wrappers/simon_wrapper.cu
    src/wrappers/skip_wrapper.cu
    src/wrappers/smith_wrapper.cu
    src/wrappers/smoa_wrapper.cu
    src/wrappers/so_wrapper.cu
    src/wrappers/ssabs_wrapper.cu
    src/wrappers/svm_wrapper.cu
    src/wrappers/tbm_wrapper.cu
    src/wrappers/tndm_wrapper.cu
    src/wrappers/trf_wrapper.cu
    src/wrappers/tsw_wrapper.cu
    src/wrappers/ts_wrapper.cu
    src/wrappers/tunedbm_wrapper.cu
    src/wrappers/tvsbs_wrapper.cu
    src/wrappers/tw_wrapper.cu
    src/wrappers/wrapper_helpers.cu
    src/wrappers/ww_wrapper.cu
    src/wrappers/zt_wrapper.cu
)
set(CUDA_ALGO_SOURCES
    src/util/tictoc.c
    src/util/device.cu
    src/util/reduction.cu
    src/algos/ac.cu
    src/algos/ag.cu
    src/algos/aoso2.cu
    src/algos/bf.cu
    src/algos/bfb.cu
    src/algos/bfbs.cu
    src/algos/bfs.cu
    src/algos/blim.cu
    src/algos/bm.cu
    src/algos/bmh_sbndm.cu
    src/algos/bndm.cu
    src/algos/bndmq2.cu
    src/algos/bndmq4.cu
    src/algos/bom.cu
    src/algos/br.cu
    src/algos/bsdm.cu
    src/algos/bww.cu
    src/algos/bxs.cu
    src/algos/col.cu
    src/algos/dfa.cu
    src/algos/dfah.cu
    src/algos/dfdm.cu
    src/algos/ebom.cu
    src/algos/faoso2.cu
    src/algos/fbom.cu
    src/algos/fdm.cu
    src/algos/ffs.cu
    src/algos/fjs.cu
    src/algos/fndm.cu
    src/algos/fs.cu
    src/algos/fsbndm.cu
    src/algos/fsbndmq20.cu
    src/algos/gg.cu
    src/algos/hash3.cu
    src/algos/hor.cu
    src/algos/ildm1.cu
    src/algos/ildm2.cu
    src/algos/kbndm.cu
    src/algos/kmp.cu
    src/algos/kmpskip.cu
    src/algos/kr.cu
    src/algos/ksa.cu
    src/algos/lbndm.cu
    src/algos/ldm.cu
    src/algos/mp.cu
    src/algos/ms.cu
    src/algos/nsn.cu
    src/algos/om.cu
    src/algos/pbmh.cu
    src/algos/qs.cu
    src/algos/raita.cu
    src/algos/rcolussi.cu
    src/algos/rf.cu
    src/algos/sa.cu
    src/algos/sabp.cu
    src/algos/sbndm.cu
    src/algos/sbndm2.cu
    src/algos/sbndmq2.cu
    src/algos/sbndm_bmh.cu
    src/algos/sebom.cu
    src/algos/sfbom.cu
    src/algos/simon.cu
    src/algos/skip.cu
    src/algos/smith.cu
    src/algos/smoa.cu
    src/algos/so.cu
    src/algos/ssabs.cu
    src/algos/svm.cu
    src/algos/tbm.cu
    src/algos/tndm.cu
    src/algos/trf.cu
    src/algos/ts.cu
    src/algos/tsw.cu
    src/algos/tunedbm.cu
    src/algos/tvsbs.cu
    src/algos/tw.cu
    src/algos/ww.cu
    src/algos/zt.cu
)
set(SERIAL_SOURCES
    src/srlalgos/ac.c
    src/srlalgos/ag.c
    src/srlalgos/akc.c
    src/srlalgos/aoso2.c
    src/srlalgos/bf.c
    src/srlalgos/bfs.c
    src/srlalgos/blim.c
    src/srlalgos/bm.c
    src/srlalgos/bmh_sbndm.c
    src/srlalgos/bndm.c
    src/srlalgos/bndml.c
    src/srlalgos/bndmq2.c
    src/srlalgos/bndmq4.c
    src/srlalgos/bom.c
    src/srlalgos/br.c
    src/srlalgos/bsdm.c
    src/srlalgos/bww.c
    src/srlalgos/bxs.c
    src/srlalgos/col.c
    src/srlalgos/dfa.c
    src/srlalgos/dfdm.c
    src/srlalgos/ebom.c
    src/srlalgos/faoso2.c
    src/srlalgos/fbom.c
    src/srlalgos/fdm.c
    src/srlalgos/ffs.c
    src/srlalgos/fjs.c
    src/srlalgos/fndm.c
    src/srlalgos/fs.c
    src/srlalgos/fsbndm.c
    src/srlalgos/fsbndmq20.c
    src/srlalgos/gg.c
    src/srlalgos/hash3.c
    src/srlalgos/hor.c
    src/srlalgos/ildm1.c
    src/srlalgos/ildm2.c
    src/srlalgos/kbndm.c
    src/srlalgos/kmp.c
    src/srlalgos/kmpskip.c
    src/srlalgos/kr.c
    src/srlalgos/ksa.c
    src/srlalgos/lbndm.c
    src/srlalgos/ldm.c
    src/srlalgos/mp.c
    src/srlalgos/ms.c
    src/srlalgos/nsn.c
    src/srlalgos/om.c
    src/srlalgos/pbmh.c
    src/srlalgos/qs.c
    src/srlalgos/raita.c
    src/srlalgos/rcolussi.c
    src/srlalgos/rf.c
    src/srlalgos/sa.c
    src/srlalgos/sabp.c
    src/srlalgos/sbndm.c
    src/srlalgos/sbndm2.c
    src/srlalgos/sbndmq2.c
    src/srlalgos/sbndm_bmh.c
    src/srlalgos/sebom.c
    src/srlalgos/sfbom.c
    src/srlalgos/simon.c
    src/srlalgos/skip.c
    src/srlalgos/smith.c
    src/srlalgos/smoa.c
    src/srlalgos/so.c
    src/srlalgos/ssabs.c
    src/srlalgos/svm.c
    src/srlalgos/tbm.c
    src/srlalgos/tndm.c
    src/srlalgos/trf.c
    src/srlalgos/ts.c
    src/srlalgos/tsw.c
    src/srlalgos/tunedbm.c
    src/srlalgos/tvsbs.c
    src/srlalgos/tw.c
    src/srlalgos/ww.c
    src/srlalgos/zt.c
)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(cpu-serial STATIC ${SERIAL_SOURCES})

add_library(gpu-parallel STATIC ${CUDA_ALGO_SOURCES} ${CUDA_WRAPPER_SOURCES})
set_target_properties(gpu-parallel
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

# disable warnings (uncomment end of line part to enable them)
if (MSVC)
    #target_compile_options(gpu-parallel PRIVATE -w)#/w /W0)#(/W4 /WX)
else()
    target_compile_options(gpu-parallel PRIVATE -w)#(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(cusmart ${MAIN_SOURCES})
target_link_libraries(cusmart 
    PRIVATE cpu-serial 
    PRIVATE gpu-parallel
    )
#set(BUILD_TESTS 1)
if(BUILD_TESTS)
    set(TEST_SOURCES
        tests/main.cpp
        tests/cputests.cpp
        tests/gputests.cpp
    )
    add_executable(cusmarttest ${TEST_SOURCES})
    target_link_libraries(cusmarttest 
        PRIVATE cpu-serial 
        PRIVATE gpu-parallel
        )   
endif(BUILD_TESTS)
