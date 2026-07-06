cmake_minimum_required(VERSION 3.18)
project(vk_gemm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")

# --- 找 shader 編譯器 -------------------------------------------------------
find_program(GLSLC glslc)
find_program(GLSLANG glslangValidator)

function(compile_shader SRC OUT REQUIRED)
    if(GLSLC)
        set(CMD ${GLSLC} --target-env=vulkan1.3 ${CMAKE_SOURCE_DIR}/${SRC} -o ${CMAKE_BINARY_DIR}/${OUT})
    elseif(GLSLANG)
        set(CMD ${GLSLANG} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/${SRC} -o ${CMAKE_BINARY_DIR}/${OUT})
    else()
        message(FATAL_ERROR "找不到 glslc / glslangValidator")
    endif()
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${OUT}
        COMMAND ${CMD}
        DEPENDS ${CMAKE_SOURCE_DIR}/${SRC}
        COMMENT "SPIR-V: ${OUT}")
endfunction()

compile_shader(shaders/gemm_tiled.comp   gemm_tiled.spv   TRUE)
compile_shader(shaders/gemm_coopmat.comp gemm_coopmat.spv FALSE)

add_custom_target(shaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/gemm_tiled.spv ${CMAKE_BINARY_DIR}/gemm_coopmat.spv)

# --- host -------------------------------------------------------------------
add_executable(vk_gemm src/main.c)
add_dependencies(vk_gemm shaders)

# libvulkan:優先 Termux 前綴,退回系統 loader
find_library(VULKAN_LIB vulkan HINTS $ENV{PREFIX}/lib)
if(NOT VULKAN_LIB)
    if(EXISTS /system/lib64/libvulkan.so)
        set(VULKAN_LIB /system/lib64/libvulkan.so)
    else()
        message(FATAL_ERROR "找不到 libvulkan(pkg install vulkan-loader-android)")
    endif()
endif()

target_include_directories(vk_gemm PRIVATE $ENV{PREFIX}/include)
target_link_libraries(vk_gemm ${VULKAN_LIB} m)

# .spv 跟執行檔擺一起(程式用相對路徑讀)
add_custom_command(TARGET vk_gemm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/gemm_tiled.spv $<TARGET_FILE_DIR:vk_gemm>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/gemm_coopmat.spv $<TARGET_FILE_DIR:vk_gemm>/ || true)
