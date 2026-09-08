#!/usr/bin/env python3
"""uvr5lite 命令行人声/伴奏分离工具。

本脚本是 UVR5 的精简命令行版本，只使用 UVR-MDX-NET-Inst_HQ_3.onnx
这一个模型。模型的原始输出是 Instrumental，也就是伴奏；脚本会先用
ONNX Runtime 推理出伴奏，再通过“原始音频 - 伴奏”得到人声。

命令行示例：
    python uvr5lite.py input.mp3 vocals.mp3 instrumental.mp3
    python uvr5lite.py input.mp3
"""

from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# 模型相关常量
#
# 这些数值与 UVR5 中 UVR-MDX-NET-Inst_HQ_3 的官方推理配置保持一致。
# 不要把这里的常量当成普通超参数随意修改，否则模型输入形状或频率映射会错位。
# ---------------------------------------------------------------------------
MODEL_FILENAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
SAMPLE_RATE = 44100

# MDX-Net 使用的 STFT 参数。
N_FFT = 6144
HOP_LENGTH = 1024
DIM_F = 3072
DIM_T = 256

# UVR5 的默认 MDX 分段大小。每一段送入模型的时间帧数为 256。
SEGMENT_SIZE = 256

# 该模型的官方补偿系数，用于让伴奏输出与原始响度更接近。
COMPENSATE = 1.022

# 每次送入模型的实际采样点数。
CHUNK_SIZE = HOP_LENGTH * (SEGMENT_SIZE - 1)

# STFT 中心填充的半个窗长度。
TRIM = N_FFT // 2

# 去掉前后 trim 后，每个有效段落实际覆盖的采样点数。
GEN_SIZE = CHUNK_SIZE - 2 * TRIM

# UVR5 默认重叠策略下相邻两个推理窗口的步长。
STEP = CHUNK_SIZE - N_FFT


def build_hann_window() -> np.ndarray:
    """构造与 PyTorch periodic Hann window 等价的 NumPy 窗。

    PyTorch 的 ``torch.hann_window(N, periodic=True)`` 使用分母 N。
    NumPy 的 ``numpy.hanning`` 使用分母 N-1，因此不能直接替代。
    """
    n = np.arange(N_FFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / N_FFT)).astype(np.float32)


# 模块加载时只构造一次窗，避免每个音频块重复计算。
WINDOW = build_hann_window()
WINDOW_SQUARE = WINDOW * WINDOW


def resolve_model_path(explicit_model: str | None) -> Path:
    """解析模型文件路径。

    默认情况下，模型放在本脚本同级的 ``models`` 目录中。
    """
    if explicit_model:
        model_path = Path(explicit_model).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "models" / MODEL_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}\n"
            f"请确认 {MODEL_FILENAME} 已放在 models 目录，"
            f"或使用 --model 指定完整路径。"
        )
    return model_path


def load_onnx_session(model_path: Path, device: str = "cpu") -> tuple[ort.InferenceSession, str, str]:
    """创建 ONNX Runtime 推理会话。

    device 只接受 "cpu" 或 "cuda"。请求 CUDA 但当前环境没有 CUDA
    ExecutionProvider 时，会自动回退到 CPU，并返回实际使用的设备名。
    """
    requested_device = device.lower()
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError(f"不支持的 device 参数：{device}。可选值：cpu、cuda。")

    available_providers = ort.get_available_providers()
    if requested_device == "cuda" and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        used_device = "cuda"
    else:
        if requested_device == "cuda":
            print(
                "警告：当前 onnxruntime 不包含 CUDAExecutionProvider，已回退到 CPU。",
                file=sys.stderr,
            )
        providers = ["CPUExecutionProvider"]
        used_device = "cpu"

    session = ort.InferenceSession(str(model_path), providers=providers)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    # 模型输入应当为 [batch, 4, 3072, 256]。这里的检查只是提前暴露错误，
    # 实际推理时仍然使用从模型读取到的输入名称。
    if len(input_shape) != 4 or input_shape[1] != 4 or input_shape[2] != DIM_F:
        raise ValueError(
            f"模型输入形状异常：{input_shape}，"
            f"预期为 [batch, 4, {DIM_F}, {DIM_T}]。"
        )

    return session, input_name, used_device


def load_and_resample_audio(input_path: Path) -> np.ndarray:
    """读取输入音频并转换为模型需要的格式。

    返回形状为 ``[2, n_samples]`` 的 float32 立体声数组。
    如果原文件不是 44100 Hz，则使用 scipy 重采样到 44100 Hz。
    """
    try:
        info = sf.info(str(input_path))
        audio, source_rate = sf.read(
            str(input_path), dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise RuntimeError(f"读取音频失败：{input_path}") from exc

    if audio.ndim == 1:
        audio = audio[:, None]

    # 统一为最多两个声道：单声道复制为双声道，多声道取前两个声道。
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    if source_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, source_rate, axis=0)
        audio = audio.astype(np.float32, copy=False)

    # 后续处理按 [channels, samples] 存储，方便 STFT 和切片。
    return np.ascontiguousarray(audio.T, dtype=np.float32)


def stft_mdx(audio_chunk: np.ndarray) -> np.ndarray:
    """计算与 UVR5 MDX-Net 一致的 STFT 输入。

    参数 ``audio_chunk`` 形状为 ``[2, CHUNK_SIZE]``。

    返回形状为 ``[4, DIM_F, DIM_T]`` 的 float32 数组，通道顺序为：
    左声道实部、左声道虚部、右声道实部、右声道虚部。
    """
    # PyTorch 的 STFT 在 center=True 时使用 reflect 边界填充。
    padded = np.pad(
        audio_chunk,
        ((0, 0), (TRIM, TRIM)),
        mode="reflect",
    )

    num_frames = 1 + (padded.shape[1] - N_FFT) // HOP_LENGTH
    if num_frames != DIM_T:
        raise RuntimeError(
            f"STFT 帧数计算异常：得到 {num_frames}，预期 {DIM_T}。"
        )

    # 使用 stride trick 一次取出所有帧，避免 Python 层逐帧循环。
    shape = (2, num_frames, N_FFT)
    strides = (
        padded.strides[0],
        HOP_LENGTH * padded.strides[1],
        padded.strides[1],
    )
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides
    )

    windowed = frames * WINDOW[None, None, :]
    spectrum = np.fft.rfft(windowed, n=N_FFT, axis=2)
    spectrum = np.transpose(spectrum, (0, 2, 1))

    left = spectrum[0]
    right = spectrum[1]

    stacked = np.stack(
        [left.real, left.imag, right.real, right.imag], axis=0
    ).astype(np.float32, copy=False)

    # UVR5 的 STFT 只保留前 DIM_F 个频点，其余高频不进入模型。
    return stacked[:, :DIM_F, :]


def istft_mdx(spec: np.ndarray) -> np.ndarray:
    """计算与 UVR5 MDX-Net 一致的逆 STFT。

    参数 ``spec`` 形状为 ``[4, DIM_F, DIM_T]``。

    返回形状为 ``[2, CHUNK_SIZE]`` 的 float32 波形。
    """
    # 模型只输出前 DIM_F 个频点，逆变换前需要补回高频零点。
    freq_pad = N_FFT // 2 + 1 - DIM_F
    full_spec = np.pad(
        spec,
        ((0, 0), (0, freq_pad), (0, 0)),
        mode="constant",
    )

    left = full_spec[0] + 1j * full_spec[1]
    right = full_spec[2] + 1j * full_spec[3]

    def _inverse_channel(channel_spec: np.ndarray) -> np.ndarray:
        """对单个声道的复数频谱做 overlap-add 逆 STFT。"""
        num_frames = channel_spec.shape[1]
        output_len = N_FFT + HOP_LENGTH * (num_frames - 1)

        # irfft 在 axis=0 上返回 [N_FFT, num_frames]，
        # 转置后得到 [num_frames, N_FFT]，便于逐帧加窗。
        time_frames = np.fft.irfft(channel_spec, n=N_FFT, axis=0).T

        output = np.zeros(output_len, dtype=np.float32)
        window_energy = np.zeros(output_len, dtype=np.float32)

        for frame_index in range(num_frames):
            start = frame_index * HOP_LENGTH
            output[start : start + N_FFT] += (
                time_frames[frame_index] * WINDOW
            )
            window_energy[start : start + N_FFT] += WINDOW_SQUARE

        # 与 torch.istft 一样，除以窗能量实现 NOLA 重建。
        window_energy = np.maximum(window_energy, 1e-12)
        output /= window_energy

        # center=True 时 PyTorch 会在两端各去掉 TRIM 个采样点。
        return output[TRIM:-TRIM]

    left_wave = _inverse_channel(left)
    right_wave = _inverse_channel(right)

    return np.stack([left_wave, right_wave], axis=0).astype(
        np.float32, copy=False
    )


def infer_chunk(
    session: ort.InferenceSession, input_name: str, audio_chunk: np.ndarray
) -> np.ndarray:
    """对一个固定长度的音频块执行模型推理。

    参数 ``audio_chunk`` 形状为 ``[2, CHUNK_SIZE]``。

    返回模型估计出的伴奏波形，形状同样为 ``[2, CHUNK_SIZE]``。
    """
    input_spec = stft_mdx(audio_chunk)

    # UVR5 的 MDX-Net 路径会丢弃最低 3 个频点。
    input_spec[:, :3, :] = 0.0

    batch = np.ascontiguousarray(input_spec[None, ...], dtype=np.float32)
    predicted_spec = session.run(None, {input_name: batch})[0]

    # 去掉 batch 维度，送入逆 STFT。
    return istft_mdx(predicted_spec[0])


def separate_audio(
    session: ort.InferenceSession,
    input_name: str,
    mix: np.ndarray,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """对整段音频执行分离。

    参数 ``mix`` 形状为 ``[2, n_samples]``。

    返回值依次为：
    - vocals：人声，由原始音频减去伴奏得到；
    - instrumental：模型估计出的伴奏，已经乘以官方补偿系数。
    """
    sample_count = mix.shape[1]

    # 该公式与 UVR5 SeperateMDX.demix() 保持一致。
    pad_size = GEN_SIZE + TRIM - (sample_count % GEN_SIZE)
    mixture = np.concatenate(
        [
            np.zeros((2, TRIM), dtype=np.float32),
            mix,
            np.zeros((2, pad_size), dtype=np.float32),
        ],
        axis=1,
    )

    result = np.zeros_like(mixture)
    divider = np.zeros_like(mixture)

    total_chunks = (mixture.shape[1] + STEP - 1) // STEP
    inference_seconds = 0.0

    for chunk_index, start in enumerate(
        range(0, mixture.shape[1], STEP), start=1
    ):
        end = min(start + CHUNK_SIZE, mixture.shape[1])
        actual_size = end - start

        audio_chunk = mixture[:, start:end]
        if actual_size < CHUNK_SIZE:
            # 最后一个窗口不足固定长度时补零，保证 STFT 帧数始终正确。
            audio_chunk = np.concatenate(
                [
                    audio_chunk,
                    np.zeros(
                        (2, CHUNK_SIZE - actual_size), dtype=np.float32
                    ),
                ],
                axis=1,
            )

        chunk_start = time.perf_counter()
        predicted = infer_chunk(session, input_name, audio_chunk)
        inference_seconds += time.perf_counter() - chunk_start

        # 默认 MDX 重叠策略不使用额外窗函数，只做线性 overlap-add。
        result[:, start:end] += predicted[:, :actual_size]
        divider[:, start:end] += 1.0

        if progress_callback is not None:
            progress_callback(chunk_index, total_chunks)

    # 与 UVR5 相同，先去掉两侧 center padding，再截回原始长度。
    divider = np.maximum(divider, 1e-12)
    instrumental = result / divider
    instrumental = instrumental[:, TRIM:-TRIM]
    instrumental = instrumental[:, :sample_count]

    # 模型原始输出为伴奏，官方推理路径会乘补偿系数。
    instrumental = instrumental * COMPENSATE

    # UVR-MDX-NET-Inst_HQ_3 是 Instrumental 模型，人声通过相减得到。
    vocals = mix - instrumental

    return (
        vocals.astype(np.float32, copy=False),
        instrumental.astype(np.float32, copy=False),
        total_chunks,
        inference_seconds,
    )


def save_audio(output_path: Path, audio: np.ndarray) -> None:
    """将形状为 ``[2, n_samples]`` 的音频写入目标文件。

    当前支持 WAV 和 MP3；其他扩展名会明确报错，避免误写入不可读文件。
    """
    suffix = output_path.suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError(
            f"不支持的输出格式：{suffix}。当前只支持 .wav 和 .mp3。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile 写入 PCM 时会做裁剪，这里提前限制到合法范围。
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(output_path), audio.T, SAMPLE_RATE)


def default_output_path(input_path: Path, prefix: str) -> Path:
    """根据当前时间生成默认输出文件名。

    默认输出到输入文件所在目录，文件名形如 ``u5vocals_20260908_123456.wav``。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return input_path.parent / f"{prefix}{timestamp}.wav"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "UVR5 精简命令行版：使用 UVR-MDX-NET-Inst_HQ_3 分离人声和伴奏。"
        ),
        epilog=(
            "示例：\n"
            "  python uvr5lite.py input.mp3 vocals.mp3 instrumental.mp3\n"
            "  python uvr5lite.py input.mp3 --vocals out.wav\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_path",
        help="输入音频文件，必选。支持 WAV、MP3、FLAC 等 soundfile 可读格式。",
    )
    parser.add_argument(
        "vocals_positional",
        nargs="?",
        default=None,
        help="可选：人声输出文件路径，例如 vocals.mp3 或 vocals.wav。",
    )
    parser.add_argument(
        "instrumental_positional",
        nargs="?",
        default=None,
        help="可选：伴奏输出文件路径，例如 instrumental.mp3。",
    )
    parser.add_argument(
        "--vocals",
        dest="vocals_option",
        default=None,
        help="可选：用 --vocals 显式指定人声输出路径。",
    )
    parser.add_argument(
        "--instrumental",
        dest="instrumental_option",
        default=None,
        help="可选：用 --instrumental 显式指定伴奏输出路径。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "可选：手动指定 ONNX 模型路径。"
            "默认使用脚本同级 models 目录下的模型。"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="可选：只输出最终的人声文件路径，不输出进度信息。",
    )

    return parser


def main(device: str = "cpu", argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Windows 终端默认可能不是 UTF-8，这里统一输出编码，避免中文乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)


    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"输入文件不存在：{input_path}")

    # 同时兼容位置参数和 --vocals / --instrumental，显式选项优先。
    vocals_path_value = args.vocals_option or args.vocals_positional
    instrumental_path_value = (
        args.instrumental_option or args.instrumental_positional
    )

    vocals_path = (
        Path(vocals_path_value).expanduser().resolve()
        if vocals_path_value
        else default_output_path(input_path, "u5vocals_")
    )
    instrumental_path = (
        Path(instrumental_path_value).expanduser().resolve()
        if instrumental_path_value
        else default_output_path(input_path, "u5instrumental_")
    )

    model_path = resolve_model_path(args.model)
    total_start = time.perf_counter()
    session, input_name, used_device = load_onnx_session(model_path, device)

    def _log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    _log(f"输入文件：{input_path}")
    _log(f"模型文件：{model_path}")
    _log(f"人声输出：{vocals_path}")
    _log(f"伴奏输出：{instrumental_path}")

    _log("正在读取音频...")
    read_start = time.perf_counter()
    mix = load_and_resample_audio(input_path)
    read_seconds = time.perf_counter() - read_start

    def _progress(current: int, total: int) -> None:
        if not args.quiet:
            print(
                f"\r推理进度：{current}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

    _log("正在进行人声/伴奏分离...")
    separation_start = time.perf_counter()
    vocals, instrumental, chunk_count, inference_seconds = separate_audio(
        session, input_name, mix, progress_callback=_progress
    )
    separation_seconds = time.perf_counter() - separation_start

    if not args.quiet:
        print("", file=sys.stderr, flush=True)

    _log("正在写入输出文件...")
    write_start = time.perf_counter()
    save_audio(vocals_path, vocals)
    save_audio(instrumental_path, instrumental)
    write_seconds = time.perf_counter() - write_start

    total_seconds = time.perf_counter() - total_start
    audio_seconds = mix.shape[1] / SAMPLE_RATE

    if not args.quiet:
        _log("技术统计：")
        _log(f"  运行设备：{used_device}")
        _log(f"  输入时长：{audio_seconds:.2f} 秒")
        _log(f"  采样率/声道：{SAMPLE_RATE} Hz / {mix.shape[0]}")
        _log(f"  推理分块数：{chunk_count}")
        _log(f"  单块长度：{CHUNK_SIZE} samples / {CHUNK_SIZE / SAMPLE_RATE:.2f} 秒")
        _log(f"  读取耗时：{read_seconds:.3f} 秒")
        _log(f"  推理耗时：{inference_seconds:.3f} 秒")
        _log(f"  平均每块推理：{inference_seconds / max(chunk_count, 1):.3f} 秒")
        _log(f"  分离阶段耗时：{separation_seconds:.3f} 秒")
        _log(f"  写入耗时：{write_seconds:.3f} 秒")
        _log(f"  总耗时：{total_seconds:.3f} 秒")
        _log(f"  实时率：{total_seconds / max(audio_seconds, 1e-12):.2f}x")
        _log(
            f"  输出文件大小："
            f"{os.path.getsize(vocals_path) / (1024 * 1024):.2f} MB / "
            f"{os.path.getsize(instrumental_path) / (1024 * 1024):.2f} MB"
        )

    # 无论是否 quiet，最终都要向标准输出返回人声文件地址。
    print(str(vocals_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main('cpu'))
