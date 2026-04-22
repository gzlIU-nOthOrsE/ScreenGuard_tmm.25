# Inference Release v1.0.0

## 1) Note / 说明

### English
This repo contains an **inference-focused version** of our code, only intended for testing.

Because this project is related to several commercial products, the released code is a **modified version** of the original internal codebase.  
In general, these modifications **do not affect testing**.

### 中文
本目录为本工作的**推理（Inference）测试版本**，仅用于测试。

由于本工作与若干公司的产品有关联，当前公开代码为在原始内部版本基础上的**修改版**。  
整体上，这些修改**基本不影响测试**。

---

## 2) How To Use / 使用方式

### English
Entry point:

```bash
python main.py --step <embed|extract|all> --wm_mode <64|240>
```

- `--step` (3 modes):
  - `embed`: only embed watermark
  - `extract`: only extract watermark
  - `all`: embed first, then extract
- `--wm_mode` (2 modes):
  - `64`
  - `240`

Examples:

```bash
# Run full pipeline in 64 mode
python main.py --step all --wm_mode 64

# Embed only in 240 mode
python main.py --step embed --wm_mode 240

# Extract only in 64 mode (from to_extract/)
python main.py --step extract --wm_mode 64
```

Input/output behavior:

- For `--step all`: extraction reads from `embedded_output/images`
- For `--step extract`: extraction reads from `to_extract`
- If you want to test with your own screenshots, place them in `to_extract/`

Attack testing:

- A JPEG attack example is already provided in comments in `extract.py`.
- You can add other attacks to test robustness.

Screenshot simulation:

- Screenshot simulation is available in `embed.py` (commented optional block).
- You can enable it to simulate screenshot (just like Windows screenshot tools or scrrenshot tools of WeChat).
- Or you can take real screenshots yourself and put them into `to_extract/` for extraction tests.

---

### 中文
入口命令：

```bash
python main.py --step <embed|extract|all> --wm_mode <64|240>
```

- `--step`（3种模式）：
  - `embed`：仅执行嵌入
  - `extract`：仅执行提取
  - `all`：先嵌入，再提取
- `--wm_mode`（2种长度模式）：
  - `64`
  - `240`

示例：

```bash
# 64模式完整流程
python main.py --step all --wm_mode 64

# 240模式仅嵌入
python main.py --step embed --wm_mode 240

# 64模式仅提取（从to_extract目录读取）
python main.py --step extract --wm_mode 64
```

输入输出规则：

- `--step all`：提取阶段读取 `embedded_output/images`
- `--step extract`：提取阶段读取 `to_extract`
- 若你希望测试自己截图的图像，可直接把图片放入 `to_extract/`

攻击测试：

- `extract.py` 中已保留 JPEG 攻击的注释示例。
- 需要测试鲁棒性时，可添加其它攻击。

截图模拟：

- `embed.py` 中保留了“截图模拟”可选逻辑（默认注释状态）。
- 可按需启用以模拟截图过程（就像Windows自带截图软件或微信截图工具）。
- 也可以自行截图后放入 `to_extract/`，直接走提取测试。

---

## 3) Citation / 引用

### English
If you find this code useful in your research or product evaluation, please cite our work:

### 中文
如果本代码对你的研究或产品评测有帮助，欢迎引用我们的工作：

```text
Liu G, Liang X, Hu X, et al. ScreenGuard: A Screen-targeted Watermarking Scheme Against Arbitrary Screenshot[J]. IEEE Transactions on Multimedia, 2025.

@article{liu2025screenguard,
  title={ScreenGuard: A Screen-targeted Watermarking Scheme Against Arbitrary Screenshot},
  author={Liu, Gaozhi and Liang, Xiujian and Hu, Xiaoxiao and Si, Yichao and Zhang, Xinpeng and Qian, Zhenxing},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

Thank you very much!  
非常感谢！
