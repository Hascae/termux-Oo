/* clock_gettime / CLOCK_MONOTONIC(嚴格 c11 需要) */
#define _POSIX_C_SOURCE 199309L

/* ============================================================================
 *  vk_gemm — Termux 原生、走最新特性的高效能 Vulkan compute GEMM
 * ----------------------------------------------------------------------------
 *  設計原則:最新 + 性能 + 穩定 三者靠「特性偵測 + 快慢路徑 fallback」共存。
 *
 *  用到的現代特性(能偵測到就開,偵測不到就降級,程式照跑):
 *    - Buffer Device Address(bindless):不用 descriptor set,直接傳 GPU 指標
 *    - Timeline semaphore:現代同步原語(取代 fence)
 *    - Timestamp query:量「真正的 GPU kernel 時間」,不是牆鐘
 *    - Pipeline cache:把編好的 pipeline 存檔,下次啟動更快
 *    - Specialization constants:pipeline 建立期把常數烘進 shader
 *    - fp16 儲存 + cooperative matrix(2024 KHR):偵測到才走,否則 fallback
 *    - Vulkan 1.4 instance(header/loader 支援時),否則退 1.3
 *
 *  兩條運算路徑:
 *    [FAST ] gemm_tiled.spv    — register-blocked tiled SGEMM(fp32)。行動 GPU 主力。
 *    [COOP ] gemm_coopmat.spv  — cooperative matrix(fp16→fp32)。硬體支援才亮。
 * ==========================================================================*/

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ---- 這些數字必須跟 gemm_tiled.comp 的 #define 一致 ---------------------- */
#define TILE_BM 64
#define TILE_BN 64

#define ITERS       50      /* 計時迭代次數(取最佳) */
#define WARMUP      10      /* 暖身迭代(不計時) */
#define VERIFY_PTS  4096    /* CPU 抽樣驗證的輸出點數 */

#define VK_CHECK(expr)                                                       \
    do {                                                                    \
        VkResult _r = (expr);                                               \
        if (_r != VK_SUCCESS) {                                             \
            fprintf(stderr, "[FATAL] %s:%d  %s -> VkResult %d\n",           \
                    __FILE__, __LINE__, #expr, (int)_r);                    \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* push constant:三個 BDA 指標 + 尺寸。scalar layout,offsets A@0 B@8 C@16 M@24 N@28 K@32 */
typedef struct {
    VkDeviceAddress A, B, C;
    uint32_t        M, N, K;
} GemmPush;

/* ---------------------------------------------------------------------------
 *  Vulkan 1.1/1.2 進入點動態載入
 *  Android 的 libvulkan.so 在低 API stub(如 API 24)只導出 1.0 符號,直接連結
 *  1.1/1.2 的核心函式會 "undefined symbol"。改用 vkGetInstanceProcAddr 在執行期
 *  載入(核心名優先,退回對應 KHR 擴充名):連結只依賴 1.0 的 vkGetInstanceProcAddr,
 *  又能在各裝置/驅動上取到正確實作。
 * ------------------------------------------------------------------------- */
static PFN_vkGetPhysicalDeviceProperties2 p_GetPhysicalDeviceProperties2;
static PFN_vkGetPhysicalDeviceFeatures2   p_GetPhysicalDeviceFeatures2;
static PFN_vkWaitSemaphores               p_WaitSemaphores;
static PFN_vkGetBufferDeviceAddress       p_GetBufferDeviceAddress;

static PFN_vkVoidFunction load_ipa(VkInstance inst, const char *core, const char *khr)
{
    PFN_vkVoidFunction f = vkGetInstanceProcAddr(inst, core);
    if (!f && khr) f = vkGetInstanceProcAddr(inst, khr);
    if (!f) { fprintf(stderr, "[FATAL] 載入不到進入點 %s\n", core); exit(1); }
    return f;
}

/* ---------------------------------------------------------------------------
 *  小工具
 * ------------------------------------------------------------------------- */
static char *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] 打不開: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fprintf(stderr, "[FATAL] 空檔: %s\n", path); exit(1); }
    char *buf = malloc((size_t)sz);
    if (!buf) { fprintf(stderr, "[FATAL] 記憶體不足,讀不下 %s (%ld bytes)\n", path, sz); exit(1); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); exit(1); }
    fclose(f); *out_size = (size_t)sz; return buf;
}

/* 讀檔但允許不存在(pipeline cache 用):回傳 NULL 表示沒有 */
static char *read_file_opt(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { *out_size = 0; return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); *out_size = 0; return NULL; }
    char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); *out_size = 0; return NULL; }
    fclose(f); *out_size = (size_t)sz; return buf;
}

/* IEEE float32 → float16 bits(測試資料用,值域小、不處理次正規/溢位細節) */
static uint16_t f32_to_f16(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;                 /* 太小 → 0 */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);     /* 太大 → inf */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* 回傳第一個符合 want 旗標的 memory type;找不到回傳 UINT32_MAX(不致命,給呼叫端 fallback) */
static uint32_t find_memory_type_opt(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
    uint32_t i = find_memory_type_opt(pd, bits, want);
    if (i == UINT32_MAX) { fprintf(stderr, "[FATAL] 找不到 memory type\n"); exit(1); }
    return i;
}

/* 建一個 host-visible + device-address 的 buffer,回傳映射指標與 GPU 位址 */
static void create_bda_buffer(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize bytes,
                              VkBufferUsageFlags extra,
                              VkBuffer *buf, VkDeviceMemory *mem,
                              void **mapped, VkDeviceAddress *addr)
{
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extra,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(dev, &bi, NULL, buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);

    VkMemoryAllocateFlagsInfo flags = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    /* 優先挑「device-local 且 host-visible」的記憶體:行動端統一記憶體常提供這種型別,
     * 它 cached/device-local(GPU 讀寫快)又能直接 map(免 staging),對 compute-bound 的
     * GEMM 才量得到真正的算力。挑不到才退回純 host-visible+coherent(可能 uncached,較慢)。 */
    uint32_t memType = find_memory_type_opt(pd, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType == UINT32_MAX)
        memType = find_memory_type(pd, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo ai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &flags,
        .allocationSize  = req.size,
        .memoryTypeIndex = memType,
    };
    VK_CHECK(vkAllocateMemory(dev, &ai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
    VK_CHECK(vkMapMemory(dev, *mem, 0, bytes, 0, mapped));

    VkBufferDeviceAddressInfo dai = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = *buf,
    };
    *addr = p_GetBufferDeviceAddress(dev, &dai);
}

/* 建一條 compute pipeline(可選 specialization) */
static VkPipeline make_pipeline(VkDevice dev, VkPipelineLayout layout, VkPipelineCache cache,
                                const char *spv_path, const VkSpecializationInfo *spec,
                                VkShaderModule *out_module)
{
    size_t sz; char *code = read_file(spv_path, &sz);
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sz, .pCode = (const uint32_t *)code,
    };
    VkShaderModule mod; VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &mod));
    free(code);

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = mod, .pName = "main",
            .pSpecializationInfo = spec,
        },
        .layout = layout,
    };
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, cache, 1, &cpci, NULL, &pipe));
    *out_module = mod;
    return pipe;
}

int main(int argc, char **argv)
{
    /* 尺寸:預設 1024,對齊到 tile 的倍數(同時滿足兩條路徑的整除要求) */
    uint32_t SIZE = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : 1024u;
    SIZE = ((SIZE + TILE_BM - 1) / TILE_BM) * TILE_BM;     /* 進位到 64 的倍數 */
    if (SIZE == 0) SIZE = TILE_BM;
    const uint32_t M = SIZE, N = SIZE, K = SIZE;
    printf("== vk_gemm ==  M=N=K=%u\n", SIZE);

    /* ------------------------------------------------------------------ *
     * 1) Instance:盡量用最新(1.4→1.3)                                   *
     * ------------------------------------------------------------------ */
    uint32_t api = VK_API_VERSION_1_3;
#ifdef VK_API_VERSION_1_4
    { uint32_t iv = 0;
      if (vkEnumerateInstanceVersion(&iv) == VK_SUCCESS && iv >= VK_API_VERSION_1_4)
          api = VK_API_VERSION_1_4; }
#endif
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vk_gemm", .apiVersion = api,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
    };
    VkInstance instance; VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
    printf("Instance API : %u.%u\n", VK_VERSION_MAJOR(api), VK_VERSION_MINOR(api));

    /* 動態載入 1.1/1.2 進入點(見檔案上方說明,避免在 API 24 stub 上連結失敗) */
    p_GetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2) load_ipa(instance, "vkGetPhysicalDeviceProperties2", "vkGetPhysicalDeviceProperties2KHR");
    p_GetPhysicalDeviceFeatures2   = (PFN_vkGetPhysicalDeviceFeatures2)   load_ipa(instance, "vkGetPhysicalDeviceFeatures2",   "vkGetPhysicalDeviceFeatures2KHR");
    p_WaitSemaphores               = (PFN_vkWaitSemaphores)               load_ipa(instance, "vkWaitSemaphores",               "vkWaitSemaphoresKHR");
    p_GetBufferDeviceAddress       = (PFN_vkGetBufferDeviceAddress)       load_ipa(instance, "vkGetBufferDeviceAddress",       "vkGetBufferDeviceAddressKHR");

    /* ------------------------------------------------------------------ *
     * 2) 選 physical device                                              *
     * ------------------------------------------------------------------ */
    uint32_t gc = 0; VK_CHECK(vkEnumeratePhysicalDevices(instance, &gc, NULL));
    if (!gc) { fprintf(stderr, "[FATAL] 沒有 Vulkan 裝置。先跑 vulkaninfo(見 README)。\n"); return 1; }
    VkPhysicalDevice *gpus = malloc(sizeof(*gpus) * gc);
    if (!gpus) { fprintf(stderr, "[FATAL] 記憶體不足\n"); return 1; }
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gc, gpus));
    VkPhysicalDevice pd = gpus[0];

    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    printf("GPU          : %s (Vulkan %u.%u.%u)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));

    /* subgroup 大小(coopmat 用) */
    VkPhysicalDeviceVulkan11Properties p11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
    VkPhysicalDeviceProperties2 props2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p11 };
    p_GetPhysicalDeviceProperties2(pd, &props2);
    uint32_t subgroup_size = p11.subgroupSize ? p11.subgroupSize : 32;
    float timestamp_period = props.limits.timestampPeriod;   /* ns / tick */

    /* ------------------------------------------------------------------ *
     * 3) 查支援哪些特性 / 擴充                                            *
     * ------------------------------------------------------------------ */
    /* 3a. 核心 1.2 features(BDA / timeline / fp16 / scalar / 16bit storage) */
    VkPhysicalDeviceVulkan11Features s11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceVulkan12Features s12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &s11 };
    VkPhysicalDeviceFeatures2 sup = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &s12 };
    p_GetPhysicalDeviceFeatures2(pd, &sup);

    if (!s12.bufferDeviceAddress) {
        fprintf(stderr, "[FATAL] 這張 GPU 不支援 bufferDeviceAddress(BDA),\n"
                        "        本程式的 bindless 路徑無法運作。\n");
        return 1;
    }
    /* scalar block layout 是必需的(BDA buffer_reference 與 push constant 都用 scalar 佈局)。
     * 它是「啟用前必須先偵測」的 feature,不查就開的話 vkCreateDevice 會神秘失敗,所以明確擋。 */
    if (!s12.scalarBlockLayout) {
        fprintf(stderr, "[FATAL] 這張 GPU 不支援 scalarBlockLayout,\n"
                        "        本程式的 scalar-layout push constant / BDA 佈局需要它。\n");
        return 1;
    }
    int have_timeline = s12.timelineSemaphore ? 1 : 0;

    /* 3b. 掃裝置擴充,找 cooperative matrix */
    int have_coopmat_ext = 0;
    { uint32_t ec = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &ec, NULL);
      VkExtensionProperties *ex = malloc(sizeof(*ex) * ec);
      vkEnumerateDeviceExtensionProperties(pd, NULL, &ec, ex);
      for (uint32_t i = 0; i < ec; i++)
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
          if (!strcmp(ex[i].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) have_coopmat_ext = 1;
#endif
      free(ex); }

    /* fp16 儲存 + 運算,是 coopmat 的前提 */
    int fp16_ok = (s12.shaderFloat16 && s11.storageBuffer16BitAccess) ? 1 : 0;

    /* ------------------------------------------------------------------ *
     * 4) 找 compute queue(要有 timestamp 支援優先)                       *
     * ------------------------------------------------------------------ */
    uint32_t qfc = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, NULL);
    VkQueueFamilyProperties *qfs = malloc(sizeof(*qfs) * qfc);
    if (!qfs) { fprintf(stderr, "[FATAL] 記憶體不足\n"); return 1; }
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, qfs);
    uint32_t qf = UINT32_MAX; int ts_ok = 0;
    for (uint32_t i = 0; i < qfc; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (qf == UINT32_MAX) qf = i;
            if (qfs[i].timestampValidBits > 0) { qf = i; ts_ok = 1; break; }
        }
    }
    if (qf == UINT32_MAX) { fprintf(stderr, "[FATAL] 沒有 compute queue\n"); return 1; }

    /* ------------------------------------------------------------------ *
     * 5) 建 logical device,啟用「支援到的」特性 + 擴充                    *
     * ------------------------------------------------------------------ */
    VkPhysicalDeviceVulkan11Features e11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .storageBuffer16BitAccess = (fp16_ok ? VK_TRUE : VK_FALSE),
    };
    VkPhysicalDeviceVulkan12Features e12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &e11,
        .bufferDeviceAddress = VK_TRUE,
        .timelineSemaphore   = (have_timeline ? VK_TRUE : VK_FALSE),
        .scalarBlockLayout   = VK_TRUE,
        .shaderFloat16       = (fp16_ok ? VK_TRUE : VK_FALSE),
    };
    void *feat_chain = &e12;
    const char *dev_exts[4]; uint32_t dev_ext_count = 0;

    int use_coopmat = 0;
    uint32_t cm_M = 0, cm_N = 0, cm_K = 0;

#ifdef VK_KHR_cooperative_matrix
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
    if (have_coopmat_ext && fp16_ok && file_exists("gemm_coopmat.spv")) {
        /* 查支援哪些 coopmat 尺寸組合,挑 fp16*fp16→fp32、subgroup scope 的 */
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (pfn) {
            uint32_t cc = 0; pfn(pd, &cc, NULL);
            if (cc) {
                VkCooperativeMatrixPropertiesKHR *cp = calloc(cc, sizeof(*cp));
                for (uint32_t i = 0; i < cc; i++) cp[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                pfn(pd, &cc, cp);
                for (uint32_t i = 0; i < cc; i++) {
                    if (cp[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                        cp[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                        cp[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                        cp[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                        cp[i].scope == VK_SCOPE_SUBGROUP_KHR) {
                        cm_M = cp[i].MSize; cm_N = cp[i].NSize; cm_K = cp[i].KSize;
                        /* 需要能被矩陣尺寸整除,才走這條 */
                        if (M % cm_M == 0 && N % cm_N == 0 && K % cm_K == 0) { use_coopmat = 1; break; }
                    }
                }
                free(cp);
            }
        }
        if (use_coopmat) {
            ecm.cooperativeMatrix = VK_TRUE;
            ecm.pNext = feat_chain; feat_chain = &ecm;
            dev_exts[dev_ext_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        }
    }
#endif

    VkPhysicalDeviceFeatures2 feats2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = feat_chain,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feats2,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = dev_ext_count, .ppEnabledExtensionNames = dev_exts,
    };
    VkDevice dev; VK_CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    printf("Path         : %s\n", use_coopmat ? "COOP (cooperative matrix)" : "FAST (tiled SGEMM, fp32)");
    printf("Features     : BDA=on timeline=%s timestamp=%s fp16=%s coopmat_ext=%s\n",
           have_timeline?"on":"off", ts_ok?"on":"off", fp16_ok?"on":"off", have_coopmat_ext?"yes":"no");
    if (use_coopmat) printf("Coopmat dims : %ux%ux%u  subgroupSize=%u\n", cm_M, cm_N, cm_K, subgroup_size);

    /* ------------------------------------------------------------------ *
     * 6) 配置 buffer(fp32 A/B/C;coopmat 額外要 fp16 A/B)                *
     * ------------------------------------------------------------------ */
    VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; void *pA, *pB, *pC;
    VkDeviceAddress aA, aB, aC;
    create_bda_buffer(pd, dev, (VkDeviceSize)M*K*sizeof(float), 0, &bA, &mA, &pA, &aA);
    create_bda_buffer(pd, dev, (VkDeviceSize)K*N*sizeof(float), 0, &bB, &mB, &pB, &aB);
    create_bda_buffer(pd, dev, (VkDeviceSize)M*N*sizeof(float), 0, &bC, &mC, &pC, &aC);

    float *A = pA, *B = pB, *C = pC;
    for (uint32_t i = 0; i < M*K; i++) A[i] = (float)((i * 13u) % 17u) * 0.03125f - 0.25f;
    for (uint32_t i = 0; i < K*N; i++) B[i] = (float)((i *  7u) % 11u) * 0.03125f - 0.15f;
    memset(C, 0, (size_t)M*N*sizeof(float));

    /* coopmat 的 fp16 輸入 buffer */
    VkBuffer bAh = VK_NULL_HANDLE, bBh = VK_NULL_HANDLE;
    VkDeviceMemory mAh = VK_NULL_HANDLE, mBh = VK_NULL_HANDLE;
    void *pAh = NULL, *pBh = NULL; VkDeviceAddress aAh = 0, aBh = 0;
    if (use_coopmat) {
        create_bda_buffer(pd, dev, (VkDeviceSize)M*K*sizeof(uint16_t), 0, &bAh, &mAh, &pAh, &aAh);
        create_bda_buffer(pd, dev, (VkDeviceSize)K*N*sizeof(uint16_t), 0, &bBh, &mBh, &pBh, &aBh);
        uint16_t *Ah = pAh, *Bh = pBh;
        for (uint32_t i = 0; i < M*K; i++) Ah[i] = f32_to_f16(A[i]);
        for (uint32_t i = 0; i < K*N; i++) Bh[i] = f32_to_f16(B[i]);
    }

    /* ------------------------------------------------------------------ *
     * 7) Pipeline layout(push constant only,零 descriptor set)          *
     * ------------------------------------------------------------------ */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(GemmPush),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
    };
    VkPipelineLayout layout; VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    /* ------------------------------------------------------------------ *
     * 8) Pipeline cache(讀舊的 → 建 pipeline → 存回)                     *
     * ------------------------------------------------------------------ */
    const char *cache_path = "vk_gemm_pipeline.cache";
    size_t cache_sz; char *cache_data = read_file_opt(cache_path, &cache_sz);
    VkPipelineCacheCreateInfo pcci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = cache_sz, .pInitialData = cache_data,
    };
    VkPipelineCache cache; VK_CHECK(vkCreatePipelineCache(dev, &pcci, NULL, &cache));
    free(cache_data);

    /* 建對應路徑的 pipeline */
    VkShaderModule mod;
    VkPipeline pipeline;
    if (use_coopmat) {
        int32_t spec_vals[4] = { (int32_t)cm_M, (int32_t)cm_N, (int32_t)cm_K, (int32_t)subgroup_size };
        VkSpecializationMapEntry ents[4] = {
            { .constantID = 0, .offset = 0,  .size = 4 },
            { .constantID = 1, .offset = 4,  .size = 4 },
            { .constantID = 2, .offset = 8,  .size = 4 },
            { .constantID = 3, .offset = 12, .size = 4 },
        };
        VkSpecializationInfo spec = {
            .mapEntryCount = 4, .pMapEntries = ents, .dataSize = sizeof(spec_vals), .pData = spec_vals,
        };
        pipeline = make_pipeline(dev, layout, cache, "gemm_coopmat.spv", &spec, &mod);
    } else {
        pipeline = make_pipeline(dev, layout, cache, "gemm_tiled.spv", NULL, &mod);
    }

    /* 存回 pipeline cache */
    { size_t n = 0; vkGetPipelineCacheData(dev, cache, &n, NULL);
      if (n) { void *d = malloc(n);
               if (vkGetPipelineCacheData(dev, cache, &n, d) == VK_SUCCESS) {
                   FILE *f = fopen(cache_path, "wb"); if (f) { fwrite(d, 1, n, f); fclose(f); } }
               free(d); } }

    /* ------------------------------------------------------------------ *
     * 9) Command buffer + timestamp query pool + timeline semaphore       *
     * ------------------------------------------------------------------ */
    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf,
    };
    VkCommandPool cmdpool; VK_CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cmdpool));
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

    VkQueryPool qpool = VK_NULL_HANDLE;
    if (ts_ok) {
        VkQueryPoolCreateInfo qpi = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2,
        };
        VK_CHECK(vkCreateQueryPool(dev, &qpi, NULL, &qpool));
    }

    /* timeline semaphore(有支援就用,沒有就 fence) */
    VkSemaphore timeline = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    if (have_timeline) {
        VkSemaphoreTypeCreateInfo sti = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0,
        };
        VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &sti };
        VK_CHECK(vkCreateSemaphore(dev, &sci, NULL, &timeline));
    } else {
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));
    }

    GemmPush push = { .M = M, .N = N, .K = K };
    if (use_coopmat) { push.A = aAh; push.B = aBh; push.C = aC; }
    else             { push.A = aA;  push.B = aB;  push.C = aC; }

    /* dispatch grid */
    uint32_t gx, gy, lz = 1;
    if (use_coopmat) { gx = N / cm_N; gy = M / cm_M; }
    else             { gx = (N + TILE_BN - 1) / TILE_BN; gy = (M + TILE_BM - 1) / TILE_BM; }
    (void)lz;

    /* 錄一次 command buffer(每輪重錄,才能重設 timestamp query) */
    uint64_t tl_val = 0;
    double best_ms = 1e30, best_gpu_ms = 1e30;

    for (int it = 0; it < WARMUP + ITERS; it++) {
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        if (qpool) vkCmdResetQueryPool(cmd, qpool, 0, 2);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        vkCmdDispatch(cmd, gx, gy, 1);
        if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);

        VK_CHECK(vkEndCommandBuffer(cmd));

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (have_timeline) {
            tl_val++;
            VkTimelineSemaphoreSubmitInfo tsi = {
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &tl_val,
            };
            VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline,
            };
            VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VkSemaphoreWaitInfo wi = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &tl_val,
            };
            VK_CHECK(p_WaitSemaphores(dev, &wi, UINT64_MAX));
        } else {
            VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
            };
            VK_CHECK(vkResetFences(dev, 1, &fence));
            VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
            VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double wall_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        double gpu_ms = -1.0;
        if (qpool) {
            uint64_t ts[2] = {0,0};
            if (vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                gpu_ms = (double)(ts[1] - ts[0]) * (double)timestamp_period / 1e6;
            }
        }

        if (it >= WARMUP) {
            if (wall_ms < best_ms)  best_ms = wall_ms;
            if (gpu_ms > 0 && gpu_ms < best_gpu_ms) best_gpu_ms = gpu_ms;
        }
    }

    /* ------------------------------------------------------------------ *
     * 10) 抽樣對拍(CPU 算 VERIFY_PTS 個輸出點,跟 GPU 比)               *
     * ------------------------------------------------------------------ */
    uint32_t bad = 0; uint32_t step = (M * N) / VERIFY_PTS; if (!step) step = 1;
    float tol = use_coopmat ? 0.2f : 1e-2f;   /* fp16 路徑容忍度放寬 */
    for (uint32_t idx = 0; idx < M * N; idx += step) {
        uint32_t r = idx / N, c = idx % N;
        float ref = 0.0f;
        for (uint32_t k = 0; k < K; k++) ref += A[r*K + k] * B[k*N + c];
        if (fabsf(C[r*N + c] - ref) > tol * (1.0f + fabsf(ref))) {
            if (bad < 5) fprintf(stderr, "  mismatch @(%u,%u): gpu=%.4f ref=%.4f\n", r, c, C[r*N+c], ref);
            bad++;
        }
    }

    /* ------------------------------------------------------------------ *
     * 11) 結果                                                            *
     * ------------------------------------------------------------------ */
    double flop = 2.0 * (double)M * N * K;
    printf("\n--- 結果 ---\n");
    printf("wall (submit+wait) best : %.3f ms  ->  %.1f GFLOP/s\n", best_ms, flop / (best_ms/1e3) / 1e9);
    if (best_gpu_ms < 1e29)
        printf("GPU kernel (timestamp)  : %.3f ms  ->  %.1f GFLOP/s\n", best_gpu_ms, flop / (best_gpu_ms/1e3) / 1e9);
    else
        printf("GPU kernel (timestamp)  : 不可用(此 queue 無 timestamp)\n");
    printf("verify (%u pts, tol=%.2g) : %s\n", VERIFY_PTS, tol, bad ? "FAIL" : "PASS");
    if (bad) printf("                          %u 點不符\n", bad);

    /* ------------------------------------------------------------------ *
     * 12) 收尾                                                            *
     * ------------------------------------------------------------------ */
    if (timeline) vkDestroySemaphore(dev, timeline, NULL);
    if (fence)    vkDestroyFence(dev, fence, NULL);
    if (qpool)    vkDestroyQueryPool(dev, qpool, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, mod, NULL);
    vkDestroyPipelineCache(dev, cache, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    if (bAh) { vkUnmapMemory(dev, mAh); vkDestroyBuffer(dev, bAh, NULL); vkFreeMemory(dev, mAh, NULL); }
    if (bBh) { vkUnmapMemory(dev, mBh); vkDestroyBuffer(dev, bBh, NULL); vkFreeMemory(dev, mBh, NULL); }
    vkUnmapMemory(dev, mA); vkDestroyBuffer(dev, bA, NULL); vkFreeMemory(dev, mA, NULL);
    vkUnmapMemory(dev, mB); vkDestroyBuffer(dev, bB, NULL); vkFreeMemory(dev, mB, NULL);
    vkUnmapMemory(dev, mC); vkDestroyBuffer(dev, bC, NULL); vkFreeMemory(dev, mC, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(instance, NULL);
    free(gpus); free(qfs);
    return bad ? 2 : 0;
}
