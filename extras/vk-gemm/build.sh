#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
#  一鍵編譯:兩個 shader → host 程式。產物放專案根目錄。Termux 原生(非 proot)。
#
#  依賴:
#    pkg install clang shaderc vulkan-headers vulkan-loader-android
#  (shaderc 提供 glslc;沒有的話會自動退回 glslangValidator,需 pkg install glslang)
#
#  coopmat shader 需要較新的 glslc/glslang 才編得動;編不動不會擋整個 build,
#  只是少了 coopmat 路徑,程式會自動走 FAST(tiled)路徑。
# ============================================================================
set -e
cd "$(dirname "$0")"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

# ---- 挑 shader 編譯器 ------------------------------------------------------
if command -v glslc >/dev/null 2>&1; then
    COMPILE_SPV() { glslc --target-env=vulkan1.3 "$1" -o "$2"; }
elif command -v glslangValidator >/dev/null 2>&1; then
    COMPILE_SPV() { glslangValidator -V --target-env vulkan1.3 "$1" -o "$2"; }
else
    echo "!! 找不到 glslc 或 glslangValidator,先 pkg install shaderc" >&2
    exit 1
fi

echo ">> 編 gemm_tiled(必要)..."
COMPILE_SPV shaders/gemm_tiled.comp gemm_tiled.spv

echo ">> 編 gemm_coopmat(選配,失敗不擋)..."
if COMPILE_SPV shaders/gemm_coopmat.comp gemm_coopmat.spv 2>/tmp/coopmat_err; then
    echo "   coopmat.spv OK"
else
    echo "   coopmat 編不動(glslc 太舊?),略過 → 程式會走 FAST 路徑"
    rm -f gemm_coopmat.spv
fi

# ---- 找 libvulkan ----------------------------------------------------------
VK_LIB="-lvulkan"
if [ ! -e "$PREFIX/lib/libvulkan.so" ]; then
    if [ -e /system/lib64/libvulkan.so ]; then
        echo ">> 用系統 loader:/system/lib64/libvulkan.so"
        VK_LIB="/system/lib64/libvulkan.so"
    else
        echo "!! 找不到 libvulkan,先 pkg install vulkan-loader-android" >&2
        exit 1
    fi
fi

# ---- 編 host ---------------------------------------------------------------
echo ">> 編 host..."
clang -O2 -std=c11 -Wall -Wextra \
    src/main.c \
    -I"$PREFIX/include" \
    $VK_LIB -lm \
    -o vk_gemm

echo ""
echo "完成。跑法:"
echo "    ./vk_gemm            # 預設 1024x1024x1024"
echo "    ./vk_gemm 2048       # 自訂邊長(會進位到 64 的倍數)"
echo ""
echo "先確認 GPU 起得來:vulkaninfo | grep -i devicename"
