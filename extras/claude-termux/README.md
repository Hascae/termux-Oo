# claude-termux — 在 Termux 上跑 Claude Code（glibc 沙盒方案）

在这个 fork 里，Claude Code 的启动器已经**预置进 bootstrap**，装机后开箱即用，一条
`claude` 命令搞定。这份文档是给自己看的备忘。

## 为什么需要它

Claude Code **v2.1.113+** 把入口从 JavaScript（`cli.js`）换成了**原生 glibc 二进制**
（`bin/claude.exe`），**没有 JS 回退**。而 Termux 底层是 Android 的 **bionic libc**，
两者是 ABI 层面的硬不兼容 —— Android 内核直接拒绝执行 glibc 二进制。所以最新版
**没法直接装在 Termux 里**。

解决办法**不是**魔改 Termux 底层（做不到，也危险），而是**在它上面叠一层 glibc 环境**：
用 `proot-distro` 开一个 Debian 沙盒（自带 glibc），Claude Code 在里面就是"在正常
Linux 上跑"。全程不碰 bootstrap / exec / 签名。

## 它由什么组成

预置进 `$PREFIX/bin` 的两个脚本（源码在 `extras/claude-termux/`，构建时注入 bootstrap）：

| 命令 | 作用 |
|---|---|
| `claude` | 启动器。把当前目录共享进沙盒并在里面跑 claude；首次使用自动触发安装 |
| `claude-setup` | 一次性安装/修复：装 proot-distro → Debian → Node.js → `@anthropic-ai/claude-code` |

**重的东西全部装在 `$HOME`（沙盒里），不在 APK 里**，所以 APK 几乎不增肥。

## 用法

首次（会联网下载 Debian + Node + Claude，几百 MB，慢，别中途杀）：

```
claude
```

它会自动跑 `claude-setup`，装完后进入 Claude Code 的登录/引导。以后再敲 `claude`
**秒起**，不会再走下载。

在某个项目里用 —— `cd` 进去再跑，当前目录会被 `--bind` 共享进沙盒：

```
cd ~/myproject
claude
```

## 维护 / 排错

**修复或更新沙盒**（Claude Code 出问题、想升级时）：
```
claude-setup
```

**彻底重置沙盒**（推倒重来）：
```
proot-distro remove debian
claude-setup
```

**换发行版**（默认 debian，可换 ubuntu 等 proot-distro 支持的）：
```
CLAUDE_TERMUX_DISTRO=ubuntu claude-setup
```
（`claude` 启动时也读这个环境变量，要固定就写进 `~/.bashrc`）

**登录**：首次进 Claude Code 会让你用 Claude 账号登录（给 URL，手机浏览器打开授权）
或贴 API key，跟着它的提示走即可。

## 改脚本不用重构建

这两个脚本就在 `$PREFIX/bin/claude` 和 `$PREFIX/bin/claude-setup`，想调逻辑**直接在
手机上编辑它们即可，不用重装 APK**。改稳了之后，再把最终版同步回
`extras/claude-termux/` 提交，下次构建就 baking 进 bootstrap。

## 注意（诚实话）

- Claude Code **官方不支持 Android/Termux**，而且分发方式老变（这次 glibc 化就是例子）。
  哪天它又改了，这套可能要跟着调 —— 出问题先看是不是 Claude Code 本身又换了打包方式。
- proot 有系统调用开销，比裸机稍慢，但日常够用。
- 沙盒装完 `$HOME` 会占 ~几百 MB（glibc + Node + Claude 的固有体积），属正常。

## 验证过的状态

首次跑通时：`proot-distro` + Debian(glibc 2.41) + Node 24（NodeSource LTS）+
Claude Code **v2.1.205** 正常启动并登录，工作目录 `--bind` 到真实 Termux 家目录，
能操作真实文件。
