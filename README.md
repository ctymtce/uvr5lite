# uvr5lite

`uvr5lite` 是 [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui)（简称 UVR5）的精简命令行版本。

它只保留一个模型：`UVR-MDX-NET-Inst_HQ_3.onnx`，用于把音频分离成两个文件：

- 人声：主要是说话声或人声演唱部分；
- 伴奏：去掉人声后的背景音乐、环境声或“杂音”部分。

本项目不使用 UVR5 的 GUI，也不依赖 PyTorch。推理只使用 NumPy、SciPy、SoundFile 和 ONNX Runtime，适合在脚本、批处理和自动化任务中调用。

---

## 1. 目录结构

```text
uvr5lite/
├── uvr5lite.py                         # 主程序，命令行入口
├── requirements.txt                    # Python 依赖
├── setup_uvr5lite.bat                  # Windows 一键创建虚拟环境并安装依赖
├── uvr5lite.bat                        # Windows 一键运行脚本
├── .gitignore
├── README.md
└── models/
    └── UVR-MDX-NET-Inst_HQ_3.onnx       # 唯一使用的分离模型
```

默认情况下，`uvr5lite.py` 会在自己所在目录的 `models` 子目录中查找模型。也可以使用 `--model` 参数手动指定模型路径。

---

## 2. 功能

- 单文件 CLI 人声/伴奏分离；
- 输入格式由 SoundFile/libmad 支持，常见格式包括 WAV、MP3、FLAC、OGG 等；
- 输出格式支持 WAV 和 MP3；
- 不提供输出路径时，自动生成带时间戳的文件名；
- 无论是否提供输出路径，最后都会在标准输出中打印人声文件的绝对路径，便于其他脚本调用；
- 只加载 `UVR-MDX-NET-Inst_HQ_3.onnx` 一个模型，体积和内存占用比完整 UVR5 小；
- 模型推理固定使用 CPU，避免 CUDA、cuDNN 等额外环境配置。

---

## 3. 模型信息

| 项目 | 值 |
| --- | --- |
| 模型文件 | `UVR-MDX-NET-Inst_HQ_3.onnx` |
| 来源 | Ultimate Vocal Remover GUI / UVR5 |
| 架构 | MDX-Net |
| 采样率 | 44100 Hz |
| STFT 点数 | 6144 |
| 帧移 | 1024 |
| 输入频率维度 | 3072 |
| 输入时间帧 | 256 |
| 补偿系数 | 1.022 |
| 模型主输出 | Instrumental |

这个模型的主输出是伴奏。`uvr5lite` 会先推理出伴奏，再用原始音频减去伴奏，得到人声。这与 UVR5 使用该模型时的默认逻辑一致。

> 模型文件来自你本地的 `D:\UVR5\models\MDX_Net_Models`。如果你计划公开发布本仓库，请先确认该模型的再分发许可。代码本身使用本仓库约定，但模型版权属于原 UVR5/模型作者。

---

## 4. 安装

### 4.1 Windows 一键安装

在 `uvr5lite` 目录中双击运行，或在终端执行：

```bat
setup_uvr5lite.bat
```

脚本会：

1. 创建当前目录下的 `.venv` 虚拟环境；
2. 升级 `pip`；
3. 根据 `requirements.txt` 安装依赖。

安装完成后可以直接运行：

```bat
uvr5lite.bat input.mp3 vocals.mp3 instrumental.mp3
```

### 4.2 手动安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bat
.venv\Scripts\activate
```

macOS / Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

### 4.3 依赖说明

`requirements.txt` 内容如下：

```text
numpy>=1.26,<3
scipy>=1.11,<2
soundfile>=0.12,<1
onnxruntime>=1.17,<2
```

本项目不依赖 PyTorch。STFT/逆 STFT 使用 NumPy 实现，推理使用 ONNX Runtime。

---

## 5. 使用

### 5.1 基础命令

同时指定人声和伴奏输出：

```bash
python uvr5lite.py input.mp3 vocals.mp3 instrumental.mp3
```

只指定输入文件：

```bash
python uvr5lite.py input.mp3
```

此时会自动生成：

```text
u5vocals_20260908_123456.wav
u5instrumental_20260908_123456.wav
```

默认文件会保存在输入音频所在的目录中。

### 5.2 显式选项

也可以使用 `--vocals` 和 `--instrumental`：

```bash
python uvr5lite.py input.wav --vocals out_vocal.wav --instrumental out_inst.mp3
```

### 5.3 Windows 批处理

```bat
uvr5lite.bat input.mp3 vocals.mp3 instrumental.mp3
```

如果 `.venv` 尚未创建，`uvr5lite.bat` 会先调用 `setup_uvr5lite.bat` 完成安装。

### 5.4 运行设备配置

`main()` 的第一个参数是 `device`，默认值为 `"cpu"`。需要 GPU 时，把 `uvr5lite.py` 末尾附近的默认值改成 `"cuda"` 即可：

```python
def main(device: str = "cuda", argv: list[str] | None = None) -> int:
```

或者直接修改 `if __name__ == "__main__":` 处的调用：

```python
raise SystemExit(main(device="cuda"))
```

如果当前 `onnxruntime` 没有 CUDA ExecutionProvider，程序会自动回退到 CPU，并在终端输出警告。

### 5.5 终端统计信息

非 `--quiet` 模式下，运行结束后会输出：

- 实际运行设备；
- 输入音频时长、采样率和声道数；
- 推理分块数和单块长度；
- 读取、推理、分离、写入和总耗时；
- 实时率；
- 两个输出文件的大小。

---

## 6. 命令行参数

| 参数 | 必选/可选 | 说明 |
| --- | --- | --- |
| `input_path` | 必选 | 输入音频文件路径 |
| `vocals` 位置参数 | 可选 | 人声输出路径，支持 `.wav`、`.mp3` |
| `instrumental` 位置参数 | 可选 | 伴奏输出路径，支持 `.wav`、`.mp3` |
| `--vocals` | 可选 | 显式指定人声输出路径 |
| `--instrumental` | 可选 | 显式指定伴奏输出路径 |
| `--model` | 可选 | 手动指定模型文件路径 |
| `--quiet` | 可选 | 只输出最终人声文件路径，不输出进度信息 |

示例：

```bash
python uvr5lite.py song.flac
python uvr5lite.py song.flac voice.wav
python uvr5lite.py song.flac voice.wav music.wav
python uvr5lite.py song.mp3 --vocals voice.mp3 --instrumental music.mp3
python uvr5lite.py song.wav --model D:\models\UVR-MDX-NET-Inst_HQ_3.onnx
```

---

## 7. 返回值

程序始终把最终生成的人声文件绝对路径打印到标准输出。例如：

```text
F:\audio\u5vocals_20260908_123456.wav
```

因此可以在批处理中捕获：

```bat
for /f "delims=" %%P in ('uvr5lite.bat input.mp3 --quiet') do set VOCAL_FILE=%%P
echo %VOCAL_FILE%
```

也可以在 PowerShell 中捕获：

```powershell
$vocal = python .\uvr5lite.py .\input.mp3 --quiet
Write-Output $vocal
```

---

## 8. 输出文件

- 如果输出扩展名是 `.wav`，写入 44100 Hz 的 PCM WAV；
- 如果输出扩展名是 `.mp3`，写入 44100 Hz 的 MPEG Layer III MP3；
- 如果未指定输出路径，默认生成 WAV 文件；
- 输出文件为立体声，长度与重采样后的输入音频一致。

示例默认文件名：

```text
u5vocals_20260908_123456.wav
u5instrumental_20260908_123456.wav
```

时间戳格式为 `YYYYMMDD_HHMMSS`。

---

## 9. 处理流程

本项目的推理流程与 UVR5 的 MDX-Net 路径保持一致：

1. 使用 SoundFile 读取输入音频；
2. 如果采样率不是 44100 Hz，使用 SciPy 重采样到 44100 Hz；
3. 单声道复制为双声道，多声道取前两个声道；
4. 将音频切成 `261120` 采样点的窗口；
5. 对每个窗口做 `n_fft=6144`、`hop_length=1024` 的 STFT；
6. 将左右声道的实部/虚部组成 `[1, 4, 3072, 256]` 输入张量；
7. 丢弃最低 3 个频点；
8. 调用 ONNX Runtime 推理；
9. 对模型输出做逆 STFT，得到伴奏窗口；
10. 通过 overlap-add 合并所有窗口；
11. 将伴奏乘以补偿系数 `1.022`；
12. 人声 = 原始音频 - 补偿后的伴奏；
   13. 写入人声和伴奏文件。

---

## 10. 性能和资源

- 每次窗口推理约覆盖 `261120 / 44100 ≈ 5.92` 秒音频；
- CPU 推理速度受机器影响，普通桌面 CPU 处理长音频可能需要数十秒到数分钟；
- 内存占用主要来自 ONNX Runtime 模型、NumPy 数组和当前音频窗口；
- `--quiet` 不会提高速度，但能减少控制台输出，便于脚本捕获路径。

---

## 11. 常见问题

### 11.1 找不到模型文件

```text
找不到模型文件：...\uvr5lite\models\UVR-MDX-NET-Inst_HQ_3.onnx
```

请确认模型文件位于 `uvr5lite\models` 目录中，或使用：

```bash
python uvr5lite.py input.wav --model "D:\path\to\UVR-MDX-NET-Inst_HQ_3.onnx"
```

### 11.2 安装依赖失败

先确认 Python 版本：

```bash
python --version
```

建议使用 Python 3.10 或更高版本。然后重新运行：

```bash
python -m pip install -r requirements.txt
```

如果网络较慢，可以配置国内 PyPI 镜像后再安装。

### 11.3 输出路径不支持

当前只支持 `.wav` 和 `.mp3`。如果需要其他格式，可以先输出 WAV，再使用 FFmpeg 转码。

### 11.4 输入音频不是 44100 Hz

程序会自动重采样到 44100 Hz，但会以模型目标采样率输出，不会保留原采样率。

---

## 12. 与完整 UVR5 的区别

| 项目 | 完整 UVR5 | uvr5lite |
| --- | --- | --- |
| 交互方式 | GUI | CLI |
| 模型数量 | 大量 VR/MDX/Demucs 模型 | 只保留 1 个模型 |
| 推理框架 | PyTorch / ONNX Runtime | 仅 ONNX Runtime |
| 依赖体积 | 很大 | 较小 |
| 功能 | 多模型、合奏、后处理等 | 单模型双输出 |
| 适合场景 | 图形界面手动操作 | 批处理、自动化、脚本集成 |

---

## 13. 致谢

- 模型和原始分离逻辑来自 [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui)；
- 本脚本中的 MDX STFT 参数和补偿系数参考了 UVR5 的实现。
