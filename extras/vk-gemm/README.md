# vk_gemm — Termux 原生高效能 Vulkan compute GEMM

矩陣乘(C = A·B)當 workload,因為它是 **compute-bound**,才看得出 GPU 肌肉
(上一版 SAXPY 是頻寬受限,GPU 算力用不到)。重點是那套走最新特性、又能在你
手機上真的跑起來的 compute 骨架。

## 設計原則:最新 + 性能 + 穩定 → 靠特性偵測 + fallback 共存

三個要求本質上會打架(最潮的特性 = 你驅動大概還沒有)。唯一的工程解就是
**runtime 偵測,偵測到走最快的路,偵測不到降級,程式照跑**。fallback 本身就是
「穩定」的實作。

兩條運算路徑,啟動時自動選:

| 路徑 | 用的東西 | 什麼時候走 |
|---|---|---|
| **FAST** | register-blocked tiled SGEMM(fp32) | 預設。行動 GPU 主力,現在就跑得動 |
| **COOP** | cooperative matrix(fp16→fp32, 2024 KHR) | 偵測到硬體支援才走(見下) |

## 用到的現代特性(能偵測就開)

- **Buffer Device Address(bindless)** — 零 descriptor set,GPU 指標直接從 push
  constant 傳進 shader(`buffer_reference`)。Android 11+ 主流。**必要**,不支援直接報錯。
- **Timeline semaphore** — 現代同步原語,取代 fence。偵測不到退回 fence。
- **Timestamp query** — 量*真正的 GPU kernel 時間*,不是含排程延遲的牆鐘。
- **Pipeline cache** — 編好的 pipeline 存 `vk_gemm_pipeline.cache`,下次啟動更快。
- **Specialization constants** — 建 pipeline 時把 coopmat 尺寸 / subgroup 大小烘進 shader。
- **fp16 儲存 + cooperative matrix** — 偵測到才走 COOP,否則 FAST。
- **Vulkan 1.4 instance** — header/loader 支援時用,否則退 1.3。

## 講白:「技術幾年次」

不誇大,這對你判斷值不值得很重要:

- **核心 GEMM idiom(shared-mem tiling + register blocking)**:HPC 幾十年的老東西,
  但這就是 GEMM 快的正解,不會過時。
- **BDA / timeline / timestamp / pipeline cache / spec constant**:大約 **2020–2022**
  的 Vulkan,現在是「主流成熟」,你機子跑得動。
- **cooperative matrix(KHR)**:**2024** 的前沿,desktop(NV/AMD/Intel)有,
  **行動 GPU 目前幾乎都還沒有**。Turnip 2025 初才剛加 ray_query,coopmat 沒進;
  Qualcomm 只有 `VK_QCOM_cooperative_matrix_conversion` 提案;Mali G1-Ultra 才開始
  吹 FP16 matmul 硬體。**所以你今天大概率走 FAST,COOP 是替「以后」鋪的路** ——
  硬體到位那天它自己亮,程式不用改。

沒有更新的東西被我藏起來沒用了 —— 該上的都上了,只是有些現在活不了。

## 安裝(Termux 原生)

```bash
pkg update && pkg upgrade -y
pkg install clang shaderc x11-repo tur-repo
pkg install vulkan-tools vulkan-headers vulkan-loader-android
```

先驗證 GPU 起得來(**看不到 GPU 名字就別往下**):

```bash
vulkaninfo | grep -i devicename
# Adreno 要用 Turnip 的話:
#   pkg install mesa-vulkan-icd-freedreno-dri3
#   export VK_ICD_FILENAMES=$PREFIX/share/vulkan/icd.d/freedreno_icd.aarch64.json
```

## 編 & 跑

```bash
bash build.sh          # 編兩個 shader + host。已附預編 .spv,沒 glslc 也能跑
./vk_gemm              # 1024^3
./vk_gemm 2048         # 自訂邊長(進位到 64 倍數)
```

預期輸出(註解版):

```
== vk_gemm ==  M=N=K=1024
Instance API : 1.3
GPU          : Adreno (TM) 740 (Vulkan 1.3.xxx)
Path         : FAST (tiled SGEMM, fp32)        ← 走了哪條路
Features     : BDA=on timeline=on timestamp=on fp16=on coopmat_ext=no
--- 結果 ---
wall (submit+wait) best : X.XXX ms  ->  XXX.X GFLOP/s   ← 含排程延遲
GPU kernel (timestamp)  : X.XXX ms  ->  XXX.X GFLOP/s   ← 真 kernel 時間,看這個
verify (4096 pts, ...)  : PASS
```

**看數字**:`GPU kernel (timestamp)` 才是純算力;`wall` 含 submit/wait 的 CPU 延遲。
GFLOP/s = 2·M·N·K / 時間。小矩陣時 wall 會被延遲主宰,把邊長開大(2048/4096)才看得準。

## 調教(要榨峰值就動這裡)

我給的 tile 參數是**安全預設,不是你 GPU 的最佳解**。真要快得自己 sweep:

改 `shaders/gemm_tiled.comp` 頂端這幾個 `#define`(改完 `bash build.sh` 重編 shader,
host 不用重編):

```
BM/BN  每個 workgroup 的 C 區塊大小(64×64)。太大爆 shared/暫存器,太小重用率低
BK     K 方向厚度(16)。影響 shared 用量
TM/TN  每條 thread 算幾個輸出(4×4)。register blocking 的核心,對峰值影響最大
```

約束:`(BM/TM)*(BN/TN)` = workgroup thread 數,必須同步改 `NTHREAD` 和 main.c 的
`local_size` 對應(目前寫死 256)。shared 用量 = `(BM+BN)*BK*4` bytes,別超過裝置上限
(`vulkaninfo` 看 `maxComputeSharedMemorySize`)。

**對齊 subgroup**:Adreno wave 常 64/128、Mali 不一,理想上 workgroup 尺寸對齊
`subgroupSize`(`vulkaninfo` 可查)。這種東西沒有通解,只能在你的機子上實測掃參數 ——
這也是為什麼我沒幫你「調好」,因為離開你的裝置調的都是假的。

想再往上:double buffering(算當前 tile 時預取下一塊)、vectorized load(`vec4`)、
把 accumulator 也 tile。這些我沒放,是 correctness/複雜度的取捨,你要的話自己疊。

## 常見雷(Android 特有)

- **`vulkaninfo` 說 no ICD** → 驅動沒暴露。裝 Turnip 設 `VK_ICD_FILENAMES`;真拿不到就這台無 root 用不了 GPU。
- **一跑就 SIGSEGV(Android 11+)** → MTE(memory tagging)。停掉 MTE 再跑。
- **`bufferDeviceAddress` 報 FATAL** → 極少見,舊驅動。這程式的 bindless 前提就是 BDA,沒有就沒法跑這版。
- **COOP 路徑始終不亮** → 正常,你的行動 GPU 現在就是沒 coopmat。不是 bug。

## 給你的隔離 fork

- **零 Termux plugin 依賴**:純 `libvulkan` + libc + libm。
- **不裝 vulkan-loader-android 也行**:直接連 `/system/lib64/libvulkan.so`(build.sh 有 fallback)。
- **附了預編 `.spv`**:你的 fork 沒 glslc 也能直接跑;要改 shader 再自己編。
- **想更硬**:`xxd -i gemm_tiled.spv > tiled_spv.h`,把 `read_file` 換成讀陣列,做成單一 binary 無外部檔案依賴。
- **headers**:不吃 tur-repo 就自己 vendor 一份 Khronos `Vulkan-Headers`,build.sh 的 `-I` 指過去。
  注意:coopmat 的 C 程式用 `#ifdef VK_KHR_cooperative_matrix` 包著,headers 太舊沒這定義也照編,只是無 COOP 路徑。

## 檔案結構

```
vk-gemm-termux/
├── README.md
├── build.sh              # 編 shader + host
├── CMakeLists.txt        # CMake 替代
├── gemm_tiled.spv        # 預編(FAST 路徑)
├── gemm_coopmat.spv      # 預編(COOP 路徑)
├── src/
│   └── main.c            # harness:偵測/BDA/timeline/timestamp/cache/bench/verify
└── shaders/
    ├── gemm_tiled.comp   # 主力 kernel(要調效能改這個)
    ├── gemm_coopmat.comp # coopmat kernel(未來路徑)
    └── compile.sh        # 單獨編 shader 用
```
