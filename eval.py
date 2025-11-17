import warnings
import argparse
import torch
import os
import os.path as osp
import numpy as np
import librosa
import torchaudio
from tqdm import tqdm
import jiwer
import string

from resemblyzer import preprocess_wav, VoiceEncoder
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, Wav2Vec2Processor, HubertForCTC
from baselines.dnsmos.dnsmos_computor import DNSMOSComputer

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.simplefilter("ignore")

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# （可选）部分系统可能依赖大写环境变量，建议同时设置以兼容
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def load_evaluation_models(xvector_extractor="resemblyzer"):
    """加载评测所需模型（说话人提取器、ASR、DNSMOS）"""
    # 1. 说话人嵌入提取器
    if xvector_extractor == "wavlm":
        wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
        speaker_extractor = (wavlm_feature_extractor, wavlm_model)
    elif xvector_extractor == "resemblyzer":
        resemblyzer_encoder = VoiceEncoder()
        speaker_extractor = resemblyzer_encoder
    else:
        raise ValueError(f"不支持的说话人提取器: {xvector_extractor}")

    # 2. ASR模型（用于WER/CER计算）
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft",sampling_rate=16000)
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

    # 3. DNSMOS音质评估器
    mos_computer = DNSMOSComputer(
        "baselines/dnsmos/sig_bak_ovr.onnx",
        "baselines/dnsmos/model_v8.onnx",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return speaker_extractor, asr_processor, asr_model, mos_computer


def get_source_transcript(source_index_tsv, source_prefix):
    """从index.tsv中获取源语音的参考文本（适配你的空格分隔格式）"""
    if not os.path.exists(source_index_tsv):
        return ""
    with open(source_index_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按第一个空格分割（文件名和文本），文件名带.wav后缀
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            file_name, transcript = parts
            # 提取文件名前缀（去掉.wav）
            file_prefix = os.path.splitext(file_name)[0]
            if file_prefix == source_prefix:
                return transcript
    return ""


def calculate_speaker_similarity(speaker_extractor, target_wav_path, converted_wav_path, extractor_type):
    """计算说话人相似度（SECS）"""
    if extractor_type == "resemblyzer":
        target_wav = preprocess_wav(target_wav_path)
        converted_wav = preprocess_wav(converted_wav_path)
        target_embed = speaker_extractor.embed_utterance(target_wav)
        converted_embed = speaker_extractor.embed_utterance(converted_wav)
        return np.inner(target_embed, converted_embed)

    elif extractor_type == "wavlm":
        feature_extractor, model = speaker_extractor
        target_wav, _ = librosa.load(target_wav_path, sr=16000)
        converted_wav, _ = librosa.load(converted_wav_path, sr=16000)

        target_inputs = feature_extractor(target_wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        converted_inputs = feature_extractor(converted_wav, sampling_rate=16000, return_tensors="pt", padding=True).to(
            device)

        with torch.no_grad():
            target_emb = model(**target_inputs).embeddings
            converted_emb = model(**converted_inputs).embeddings

        target_emb = torch.nn.functional.normalize(target_emb, dim=-1)
        converted_emb = torch.nn.functional.normalize(converted_emb, dim=-1)
        return torch.nn.functional.cosine_similarity(target_emb, converted_emb, dim=-1).item()


def calculate_asr_metrics(asr_processor, asr_model, audio_path, reference_text):
    """计算WER和CER"""
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = asr_processor(audio, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.decode(predicted_ids[0])

    # 预处理文本（小写、去除标点）
    reference = reference_text.lower().translate(str.maketrans("", "", string.punctuation)) if reference_text else ""
    hypothesis = transcription.lower().translate(str.maketrans("", "", string.punctuation))

    wer = jiwer.wer(reference, hypothesis) if reference else None
    cer = jiwer.cer(reference, hypothesis) if reference else None
    return wer, cer, reference, hypothesis


def calculate_dnsmos(mos_computer, audio_path):
    """计算DNSMOS指标（SIG/BAK/OVRL）"""
    audio, sr = librosa.load(audio_path, sr=16000)
    result = mos_computer.compute(audio, 16000, False)
    return result["SIG"], result["BAK"], result["OVRL"]


def main(args):
    # 加载评估模型
    speaker_extractor, asr_processor, asr_model, mos_computer = load_evaluation_models(args.xvector_extractor)

    # 总结果存储
    all_results = []

    # 遍历所有转换任务文件夹（格式：原语音前缀_to_目标语音前缀）
    conversion_dirs = [
        d for d in os.listdir(args.converted_root)
        if os.path.isdir(os.path.join(args.converted_root, d)) and "_to_" in d
    ]

    if not conversion_dirs:
        print(f"未找到符合格式的转换任务文件夹（需包含 '_to_'），请检查路径：{args.converted_root}")
        return

    # 源语音的index.tsv路径（用于获取参考文本）
    source_index_tsv = args.source_labels  # 假设你的index.tsv在源语音根目录

    for conv_dir in tqdm(conversion_dirs, desc="评测进度"):
        # 解析文件夹名称：原语音前缀_to_目标语音前缀
        conv_dir_path = os.path.join(args.converted_root, conv_dir)
        try:
            source_prefix, target_prefix = conv_dir.split("_to_", 1)
        except ValueError:
            print(f"跳过无效文件夹名（格式应为 '原前缀_to_目标前缀'）：{conv_dir}")
            continue

        # 定义当前任务的三个文件路径
        source_wav_path = os.path.join(conv_dir_path, f"{conv_dir.split("_to_")[0]}.wav")  # 子文件夹内的源语音
        target_wav_path = os.path.join(conv_dir_path, f"{conv_dir.split("_to_")[1]}.wav")  # 子文件夹内的目标语音
        converted_wav_path = os.path.join(conv_dir_path, f"{conv_dir}.wav")  # 转换后的语音（与文件夹同名）

        # 检查文件是否存在
        missing_files = []
        if not os.path.exists(source_wav_path):
            missing_files.append(f"源语音.wav（路径：{source_wav_path}）")
        if not os.path.exists(target_wav_path):
            missing_files.append(f"目标语音.wav（路径：{target_wav_path}）")
        if not os.path.exists(converted_wav_path):
            missing_files.append(f"转换后语音（{conv_dir}.wav，路径：{converted_wav_path}）")

        if missing_files:
            print(f"跳过文件夹 {conv_dir}，缺少文件：{', '.join(missing_files)}")
            continue

        # 1. 获取源语音的参考文本（用于WER/CER计算）
        source_transcript = get_source_transcript(source_index_tsv, source_prefix)
        if not source_transcript:
            print(f"警告：未在index.tsv中找到 {source_prefix} 的参考文本，WER/CER将无法计算")

        # 2. 计算说话人相似度（转换后语音 vs 目标语音）
        try:
            secs = calculate_speaker_similarity(
                speaker_extractor,
                target_wav_path,
                converted_wav_path,
                args.xvector_extractor
            )
        except Exception as e:
            print(f"计算 {conv_dir} 的说话人相似度失败：{str(e)}")
            secs = None

        # 3. 计算WER和CER（转换后语音的识别结果 vs 源语音参考文本）
        try:
            wer, cer, ref_text, hyp_text = calculate_asr_metrics(
                asr_processor, asr_model, converted_wav_path, source_transcript
            )
        except Exception as e:
            print(f"计算 {conv_dir} 的WER/CER失败：{str(e)}")
            wer, cer = None, None

        # 4. 计算DNSMOS音质指标
        try:
            sig, bak, ovr = calculate_dnsmos(mos_computer, converted_wav_path)
        except Exception as e:
            print(f"计算 {conv_dir} 的DNSMOS失败：{str(e)}")
            sig, bak, ovr = None, None, None

        # 保存当前任务结果
        result = {
            "转换任务文件夹": conv_dir,
            "源语音前缀": source_prefix,
            "目标语音前缀": target_prefix,
            "说话人相似度（SECS）": secs,
            "词错误率（WER）": wer,
            "字符错误率（CER）": cer,
            "DNSMOS清晰度（SIG）": sig,
            "DNSMOS背景噪声（BAK）": bak,
            "DNSMOS整体质量（OVRL）": ovr
        }
        all_results.append(result)

        # 打印当前任务结果
        print(f"\n任务 {conv_dir} 评测完成：")
        print(f"说话人相似度：{secs:.4f}" if secs is not None else "说话人相似度：计算失败")
        if wer is not None and cer is not None:
            print(f"WER：{wer:.4f}，CER：{cer:.4f}")
        print(f"DNSMOS（SIG/BAK/OVRL）：{sig:.4f}/{bak:.4f}/{ovr:.4f}" if ovr is not None else "DNSMOS：计算失败")

        # 在当前任务文件夹下保存单独结果
        with open(os.path.join(conv_dir_path, "evaluation_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"转换任务：{conv_dir}\n")
            f.write(f"源语音前缀：{source_prefix}\n")
            f.write(f"目标语音前缀：{target_prefix}\n")
            f.write(f"说话人相似度（SECS）：{secs:.4f}\n" if secs is not None else "说话人相似度：计算失败\n")
            f.write(f"词错误率（WER）：{wer:.4f}\n" if wer is not None else "WER：计算失败\n")
            f.write(f"字符错误率（CER）：{cer:.4f}\n" if cer is not None else "CER：计算失败\n")
            f.write(f"DNSMOS清晰度（SIG）：{sig:.4f}\n" if sig is not None else "SIG：计算失败\n")
            f.write(f"DNSMOS背景噪声（BAK）：{bak:.4f}\n" if bak is not None else "BAK：计算失败\n")
            f.write(f"DNSMOS整体质量（OVRL）：{ovr:.4f}\n" if ovr is not None else "OVRL：计算失败\n")

    # 生成总结果文件
    total_result_path = os.path.join(args.converted_root, "总评测结果.txt")
    with open(total_result_path, "w", encoding="utf-8") as f:
        f.write("所有转换任务总评测结果\n")
        f.write("=" * 50 + "\n")
        for res in all_results:
            f.write(f"转换任务：{res['转换任务文件夹']}\n")
            f.write(f"源前缀 -> 目标前缀：{res['源语音前缀']} -> {res['目标语音前缀']}\n")
            f.write(f"说话人相似度：{res['说话人相似度（SECS）']:.4f}\n" if res[
                                                                             '说话人相似度（SECS）'] is not None else "说话人相似度：计算失败\n")
            f.write(f"WER：{res['词错误率（WER）']:.4f}，CER：{res['字符错误率（CER）']:.4f}\n" if res[
                                                                                                '词错误率（WER）'] is not None else "WER/CER：计算失败\n")
            f.write(
                f"DNSMOS（SIG/BAK/OVRL）：{res['DNSMOS清晰度（SIG）']:.4f}/{res['DNSMOS背景噪声（BAK）']:.4f}/{res['DNSMOS整体质量（OVRL）']:.4f}\n" if
                res['DNSMOS整体质量（OVRL）'] is not None else "DNSMOS：计算失败\n")
            f.write("-" * 50 + "\n")

    print(f"\n所有任务评测完成！总结果已保存至：{total_result_path}")
    print(f"每个任务的单独结果已保存至各自文件夹下的 'evaluation_result.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="适配自定义文件夹结构的语音转换评测工具")
    parser.add_argument("--converted_root", default='/root/autodl-tmp/output/consistency_VC',
                        help="转换语音的根目录（包含所有 '原前缀_to_目标前缀' 子文件夹）")
    parser.add_argument("--source_labels", default='AISHELL3_all.txt', help="源语音的根目录（包含 index.tsv 文件）")
    parser.add_argument("--xvector_extractor", default="resemblyzer",
                        choices=["resemblyzer", "wavlm"], help="说话人嵌入提取器类型")
    args = parser.parse_args()
    main(args)
