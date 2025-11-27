import argparse
import os
import re
import string
import unicodedata
import warnings
import csv

import cn2an
import jiwer
import librosa
import numpy as np
import torch
import whisper
from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from opencc import OpenCC
import glob2

from baselines.dnsmos.dnsmos_computor import DNSMOSComputer

# ==== 环境 & 全局 ====
_cc_t2s = OpenCC("t2s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.simplefilter("ignore")

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


# ==== 文本清洗 ====
def sanitize_text(s: str) -> str:
    """
    清洗规则：
    1) NFKC 全半角统一
    2) 繁体 -> 简体
    3) 数字转中文
    4) 去所有空白
    5) 去掉所有 Unicode 标点(P*)和符号(S*)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))  # 1
    s = _cc_t2s.convert(s)                    # 2
    s = cn2an.transform(s, "an2cn")           # 3
    s = re.sub(r"\s+", "", s)                 # 4
    s = "".join(
        ch for ch in s
        if unicodedata.category(ch)[0] not in ("P", "S")
    )                                          # 5
    return s.strip()


# ==== 模型加载 ====
def load_evaluation_models(xvector_extractor="resemblyzer"):
    """加载评测所需模型（说话人提取器、ASR、DNSMOS）"""
    # 1. 说话人嵌入提取器
    if xvector_extractor == "wavlm":
        wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        wavlm_model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device)
        speaker_extractor = (wavlm_feature_extractor, wavlm_model)
    elif xvector_extractor == "resemblyzer":
        resemblyzer_encoder = VoiceEncoder()
        speaker_extractor = resemblyzer_encoder
    else:
        raise ValueError(f"不支持的说话人提取器: {xvector_extractor}")

    # 2. ASR模型（whisper turbo）
    asr_model = whisper.load_model("turbo", device=device)

    # 3. DNSMOS音质评估器
    mos_computer = DNSMOSComputer(
        "baselines/dnsmos/sig_bak_ovr.onnx",
        "baselines/dnsmos/model_v8.onnx",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return speaker_extractor, asr_model, mos_computer


# ==== 文本 index 读取 ====
def get_source_transcript(index_path, source_prefix):
    """从 index 文件中获取源语音的参考文本（格式：file.wav<空格>文本）"""
    # print(index_path)
    # if not os.path.exists(index_path):
    #     print("标签文件不存在")
    #     return ""
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            file_name, transcript = parts
            file_prefix = os.path.splitext(file_name)[0]
            if file_prefix == source_prefix:
                return transcript
    return ""


# ==== 说话人相似度 ====
def calculate_speaker_similarity(speaker_extractor, target_wav_path, converted_wav_path, extractor_type):
    """计算说话人相似度（SECS）"""
    if extractor_type == "resemblyzer":
        target_wav = preprocess_wav(target_wav_path)
        converted_wav = preprocess_wav(converted_wav_path)
        target_embed = speaker_extractor.embed_utterance(target_wav)
        converted_embed = speaker_extractor.embed_utterance(converted_wav)
        return float(np.inner(target_embed, converted_embed))

    elif extractor_type == "wavlm":
        feature_extractor, model = speaker_extractor
        target_wav, _ = librosa.load(target_wav_path, sr=16000)
        converted_wav, _ = librosa.load(converted_wav_path, sr=16000)

        target_inputs = feature_extractor(
            target_wav, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(device)
        converted_inputs = feature_extractor(
            converted_wav, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            target_emb = model(**target_inputs).embeddings
            converted_emb = model(**converted_inputs).embeddings

        target_emb = torch.nn.functional.normalize(target_emb, dim=-1)
        converted_emb = torch.nn.functional.normalize(converted_emb, dim=-1)
        return float(torch.nn.functional.cosine_similarity(target_emb, converted_emb, dim=-1).item())


# ==== ASR / CER ====
def calculate_asr_metrics(asr_model, audio_path, reference_text):
    """计算 CER（字符错误率）"""
    result = asr_model.transcribe(audio_path, language="zh")
    transcription = result["text"]
    transcription = sanitize_text(transcription)

    reference = reference_text.lower().translate(str.maketrans("", "", string.punctuation)) if reference_text else ""
    hypothesis = transcription.lower().translate(str.maketrans("", "", string.punctuation))

    cer = jiwer.cer(reference, hypothesis) if reference else None
    return cer, reference, hypothesis


# ==== DNSMOS ====
def calculate_dnsmos(mos_computer, audio_path):
    """计算 DNSMOS 指标（SIG/BAK/OVRL）"""
    audio, sr = librosa.load(audio_path, sr=16000)
    result = mos_computer.compute(audio, 16000, False)
    return float(result["SIG"]), float(result["BAK"]), float(result["OVRL"])


# ==== 主流程 ====
def main(args):
    # 1. 加载模型
    speaker_extractor, asr_model, mos_computer = load_evaluation_models(args.xvector_extractor)

    # 2. 用 glob2 建立源/目标语音查找表
    print("使用 glob2 建立源语音/说话人语音索引...")
    src_wavs = glob2.glob(os.path.join(args.source_root, "**", "*.wav"))
    spk_wavs = glob2.glob(os.path.join(args.speaker_root, "**", "*.wav"))

    # 使用 basename(去掉 .wav) 作为键（假设全局唯一）
    src_map = {os.path.splitext(os.path.basename(p))[0]: p for p in src_wavs}
    spk_map = {os.path.splitext(os.path.basename(p))[0]: p for p in spk_wavs}

    print(f"  源语音数量: {len(src_map)}")
    print(f"  说话人语音数量: {len(spk_map)}")

    # 3. 遍历 converted_root 下的各个 seen_* 目录
    if not os.path.isdir(args.converted_root):
        print(f"converted_root 不存在或不是目录: {args.converted_root}")
        return

    all_results = []

    subdirs = sorted(
        d for d in os.listdir(args.converted_root)
        if os.path.isdir(os.path.join(args.converted_root, d))
    )
    if not subdirs:
        print(f"在 {args.converted_root} 下未找到任何子目录（如 seen_SSBxxxx），请检查。")
        return

    print("开始遍历转换结果目录...")
    for subdir in tqdm(subdirs, desc="目录进度"):
        subdir_path = os.path.join(args.converted_root, subdir)

        # 找到该目录下所有转换后 wav
        conv_wavs = sorted(glob2.glob(os.path.join(subdir_path, "*.wav")))
        if not conv_wavs:
            continue

        for conv_path in conv_wavs:
            conv_name = os.path.basename(conv_path)
            stem = os.path.splitext(conv_name)[0]

            # 你已经明确保证：一定是 原_to_目标.wav
            if "_to_" not in stem:
                print(f"[警告] 文件名不符合 src_to_tgt 规则，跳过：{conv_name}")
                continue

            source_prefix, target_prefix = stem.split("_to_", 1)

            # 在 src_map / spk_map 中查找真正的源/目标 wav 路径
            source_wav_path = src_map.get(source_prefix, None)
            target_wav_path = spk_map.get(target_prefix, None)

            if source_wav_path is None:
                print(f"[警告] 找不到源语音 {source_prefix} 对应的 wav，跳过 {conv_name}")
                continue
            if target_wav_path is None:
                print(f"[警告] 找不到目标说话人 {target_prefix} 对应的 wav，跳过 {conv_name}")
                continue

            # 1. 参考文本（用于 CER）
            source_transcript = get_source_transcript(args.source_labels, source_prefix)
            if not source_transcript:
                print(f"[警告] index 中未找到 {source_prefix} 的文本，CER 将为 None")

            # 2. 说话人相似度
            try:
                secs = calculate_speaker_similarity(
                    speaker_extractor,
                    target_wav_path,
                    conv_path,
                    args.xvector_extractor
                )
            except Exception as e:
                print(f"[错误] 计算 {conv_name} 的 SECS 失败: {e}")
                secs = None

            # 3. CER
            try:
                cer, ref_text, hyp_text = calculate_asr_metrics(
                    asr_model, conv_path, source_transcript
                )
            except Exception as e:
                print(f"[错误] 计算 {conv_name} 的 CER 失败: {e}")
                cer, ref_text, hyp_text = None, None, None

            # 4. DNSMOS
            try:
                sig, bak, ovr = calculate_dnsmos(mos_computer, conv_path)
            except Exception as e:
                print(f"[错误] 计算 {conv_name} 的 DNSMOS 失败: {e}")
                sig, bak, ovr = None, None, None

            # 记录结果
            rel_conv = os.path.relpath(conv_path, args.converted_root)
            a_result = {
                "subdir": subdir,
                "converted_wav": rel_conv,
                "source_prefix": source_prefix,
                "target_prefix": target_prefix,
                "source_wav": source_wav_path,
                "target_wav": target_wav_path,
                "SECS": secs,
                "CER": cer,
                "ref_text":ref_text, 
                "hyp_text":hyp_text,
                "DNSMOS_SIG": sig,
                "DNSMOS_BAK": bak,
                "DNSMOS_OVRL": ovr,
            } #我在这里添加了一些键 ，但是我后面的代码没有改动，请你帮我改一下
            print(a_result)
            all_results.append(a_result)

    if not all_results:
        print("没有成功评测的条目，退出。")
        return

    # 4. 写 CSV & 计算平均值
    if os.path.dirname(args.out_csv):
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    fieldnames = [
        "subdir", "converted_wav",
        "source_prefix", "target_prefix",
        "source_wav", "target_wav",
        "SECS", "CER",
        "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL",
        "ref_text", "hyp_text",
    ]

    secs_list, cer_list, sig_list, bak_list, ovr_list = [], [], [], [], []

    def add_if_not_none(lst, v):
        if v is not None:
            lst.append(v)

    for r in all_results:
        add_if_not_none(secs_list, r["SECS"])
        add_if_not_none(cer_list, r["CER"])
        add_if_not_none(sig_list, r["DNSMOS_SIG"])
        add_if_not_none(bak_list, r["DNSMOS_BAK"])
        add_if_not_none(ovr_list, r["DNSMOS_OVRL"])

    avg_secs = sum(secs_list) / len(secs_list) if secs_list else 0.0
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
    avg_sig = sum(sig_list) / len(sig_list) if sig_list else 0.0
    avg_bak = sum(bak_list) / len(bak_list) if bak_list else 0.0
    avg_ovr = sum(ovr_list) / len(ovr_list) if ovr_list else 0.0

    # 写 CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
        # 最后一行写平均值
        writer.writerow({
            "subdir": "AVERAGE",
            "converted_wav": "",
            "source_prefix": "",
            "target_prefix": "",
            "source_wav": "",
            "target_wav": "",
            "SECS": avg_secs,
            "CER": avg_cer,
            "DNSMOS_SIG": avg_sig,
            "DNSMOS_BAK": avg_bak,
            "DNSMOS_OVRL": avg_ovr,
            "ref_text": "",
            "hyp_text": "",
        })

    print("\n评测完成！")
    print(f"共 {len(all_results)} 条样本，结果已写入：{args.out_csv}")
    print(f"平均 SECS = {avg_secs:.4f}")
    print(f"平均 CER  = {avg_cer:.4f}")
    print(f"平均 DNSMOS_SIG = {avg_sig:.4f}")
    print(f"       DNSMOS_BAK = {avg_bak:.4f}")
    print(f"       DNSMOS_OVRL = {avg_ovr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="适配 原_to_目标 命名的语音转换评测工具（CSV 输出）")

    parser.add_argument(
        "--converted_root",
        default="/root/autodl-tmp/test_result/openvoicev2",
        help="转换语音的根目录（下面有 seen_xxx 等子目录，每个目录下是若干 原_to_目标.wav）"
    )
    parser.add_argument(
        "--source_root",
        default="/root/autodl-tmp/test_set/speech",
        help="源语音根目录（用 glob2 递归收集所有 wav）"
    )
    parser.add_argument(
        "--speaker_root",
        default="/root/autodl-tmp/test_set/speaker",
        help="说话人语音根目录（用 glob2 递归收集所有 wav 作为 target pool）"
    )
    parser.add_argument(
        "--source_labels",
        default="/root/zh-vc-evaluate/test_labels.txt",
        help="源语音 index 文件路径（格式：file.wav<空格>文本）"
    )
    parser.add_argument(
        "--xvector_extractor",
        default="resemblyzer",
        choices=["resemblyzer", "wavlm"],
        help="说话人嵌入提取器类型"
    )
    parser.add_argument(
        "--out_csv",
        default="evaluation_results.csv",
        help="输出 CSV 文件路径"
    )

    args = parser.parse_args()
    main(args)
