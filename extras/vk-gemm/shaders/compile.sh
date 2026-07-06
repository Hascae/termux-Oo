#!/data/data/com.termux/files/usr/bin/bash
# 單獨編 shader → SPIR-V。build.sh 會呼叫這個邏輯,這支給你手動重編用。
set -e
cd "$(dirname "$0")/.."

if command -v glslc >/dev/null 2>&1; then
    C() { glslc --target-env=vulkan1.3 "$1" -o "$2"; }
elif command -v glslangValidator >/dev/null 2>&1; then
    C() { glslangValidator -V --target-env vulkan1.3 "$1" -o "$2"; }
else
    echo "找不到 glslc / glslangValidator(pkg install shaderc)" >&2; exit 1
fi

C shaders/gemm_tiled.comp   gemm_tiled.spv   && echo "OK gemm_tiled.spv"
if C shaders/gemm_coopmat.comp gemm_coopmat.spv 2>/dev/null; then
    echo "OK gemm_coopmat.spv"
else
    echo "coopmat 編不動,略過(程式走 FAST 路徑)"; rm -f gemm_coopmat.spv
fi
