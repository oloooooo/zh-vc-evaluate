import argparse
import os
import re
import string
import unicodedata
import warnings

import cn2an
import jiwer
import librosa
import numpy as np
import torch
import whisper
import zhconv
from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from baselines.dnsmos.dnsmos_computor import DNSMOSComputer
from opencc import OpenCC

_cc_t2s = OpenCC("t2s")
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.simplefilter("ignore")

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# （可选）部分系统可能依赖大写环境变量，建议同时设置以兼容
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def sanitize_text(s: str) -> str:
    """
    清洗规则（顺序很重要）：
    1) NFKC 全半角统一
    2) 繁体 -> 简体
    3) 数字按语境转中文（年：逐位；其他：正常中文数词）
    4) 去所有空白
    5) 去掉所有 Unicode 标点(P*)和符号(S*)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))  # 1) 全半角统一
    s = _cc_t2s.convert(s)  # 2) 繁->简
    s = cn2an.transform(s, "an2cn")  # 3) 数字转中文（句子级）
    s = re.sub(r"\s+", "", s)  # 4) 去空白
    s = "".join(ch for ch in s  # 5) 去标点/符号
                if unicodedata.category(ch)[0] not in ("P", "S"))
    return s.strip()


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

    # 2. ASR模型（使用whisper，指定中文）
    asr_model = whisper.load_model("turbo", device=device)

    # 3. DNSMOS音质评估器
    mos_computer = DNSMOSComputer(
        "baselines/dnsmos/sig_bak_ovr.onnx",
        "baselines/dnsmos/model_v8.onnx",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return speaker_extractor, asr_model, mos_computer


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
def get_conversion_task_dirs(converted_root):
    """
    返回所有的任务文件夹名称。

    默认规则：
        文件夹名包含 "_to_"

    如果用户的结构不同，可以在这里完全修改筛选逻辑。

    返回：
        任务文件夹名列表（不含完整路径）
    """

    dirs = []
    for d in os.listdir(converted_root):
        full_path = os.path.join(converted_root, d)
        if os.path.isdir(full_path) and "_to_" in d:   # ← 默认规则
            dirs.append(d)

    return dirs

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


def calculate_asr_metrics(asr_model, audio_path, reference_text):
    """计算CER（只保留字符错误率）"""
    # 使用whisper进行语音识别，指定语言为中文
    result = asr_model.transcribe(audio_path, language="zh")
    transcription = result["text"]
    transcription = sanitize_text(transcription)

    # 预处理文本（小写、去除标点）
    reference = reference_text.lower().translate(str.maketrans("", "", string.punctuation)) if reference_text else ""
    hypothesis = transcription.lower().translate(str.maketrans("", "", string.punctuation))

    cer = jiwer.cer(reference, hypothesis) if reference else None
    return cer, reference, hypothesis


def calculate_dnsmos(mos_computer, audio_path):
    """计算DNSMOS指标（SIG/BAK/OVRL）"""
    audio, sr = librosa.load(audio_path, sr=16000)
    result = mos_computer.compute(audio, 16000, False)
    return result["SIG"], result["BAK"], result["OVRL"]


def main(args):
    # 加载评估模型
    speaker_extractor, asr_model, mos_computer = load_evaluation_models(args.xvector_extractor)
    # 总结果存储

    all_results = []
    conversion_dirs = get_conversion_task_dirs(args.converted_root)
    if not conversion_dirs:
        print(f"未找到符合格式的转换任务文件夹（需包含 '_to_'），请检查路径：{args.converted_root}")
        return

    # 源语音的index.tsv路径（用于获取参考文本）
    source_index_tsv = args.source_labels

    for conv_dir in tqdm(conversion_dirs, desc="评测进度"):
        ref_text = None
        hyp_text = None
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

        # 1. 获取源语音的参考文本（用于CER计算）
        source_transcript = get_source_transcript(source_index_tsv, source_prefix)
        if not source_transcript:
            print(f"警告：未在index.tsv中找到 {source_prefix} 的参考文本，CER将无法计算")

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

        # 3. 计算CER（转换后语音的识别结果 vs 源语音参考文本）
        try:
            cer, ref_text, hyp_text = calculate_asr_metrics(
                asr_model, converted_wav_path, source_transcript
            )
        except Exception as e:
            print(f"计算 {conv_dir} 的CER失败：{str(e)}")
            cer = None

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
            "字符错误率（CER）": cer,
            "DNSMOS清晰度（SIG）": sig,
            "DNSMOS背景噪声（BAK）": bak,
            "DNSMOS整体质量（OVRL）": ovr
        }
        all_results.append(result)

        # 打印当前任务结果
        print(f"\n任务 {conv_dir} 评测完成：")
        print(f"说话人相似度：{secs:.4f}" if secs is not None else "说话人相似度：计算失败")
        if cer is not None and ref_text is not None and hyp_text is not None:
            print(f"CER：{cer:.4f}, 真实标签为：{ref_text}, 识别结果为：{hyp_text}")
        print(f"DNSMOS（SIG/BAK/OVRL）：{sig:.4f}/{bak:.4f}/{ovr:.4f}" if ovr is not None else "DNSMOS：计算失败")

        # 在当前任务文件夹下保存单独结果
        with open(os.path.join(conv_dir_path, "evaluation_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"转换任务：{conv_dir}\n")
            f.write(f"源语音前缀：{source_prefix}\n")
            f.write(f"目标语音前缀：{target_prefix}\n")
            f.write(f"说话人相似度（SECS）：{secs:.4f}\n" if secs is not None else "说话人相似度：计算失败\n")
            f.write(f"字符错误率（CER）：{cer:.4f}\n" if cer is not None else "CER：计算失败\n")
            f.write(f"DNSMOS清晰度（SIG）：{sig:.4f}\n" if sig is not None else "SIG：计算失败\n")
            f.write(f"DNSMOS背景噪声（BAK）：{bak:.4f}\n" if bak is not None else "BAK：计算失败\n")
            f.write(f"DNSMOS整体质量（OVRL）：{ovr:.4f}\n" if ovr is not None else "OVRL：计算失败\n")

    # 生成总结果文件
    total_result_path = os.path.join(args.converted_root, "总评测结果.txt")
    with open(total_result_path, "w", encoding="utf-8") as f:
        f.write("所有转换任务总评测结果\n")
        f.write("=" * 50 + "\n")
        sec_list = []
        cer_list = []
        sig_list = []
        ovrl_list = []
        bak_list = []
        add_to_list = lambda lst, var: lst + [var] if var is not None else lst
        for res in all_results:
            sec = res['说话人相似度（SECS）']
            cer = res['字符错误率（CER）']
            SIG = res['DNSMOS清晰度（SIG）']
            OVRL = res['DNSMOS整体质量（OVRL）']
            BAK = res['DNSMOS背景噪声（BAK）']
            sec_list = add_to_list(sec_list, sec)
            cer_list = add_to_list(cer_list, cer)
            sig_list = add_to_list(sig_list, SIG)
            ovrl_list = add_to_list(ovrl_list, OVRL)
            bak_list = add_to_list(bak_list, BAK)
            f.write(f"转换任务：{res['转换任务文件夹']}\n")
            f.write(f"源前缀 -> 目标前缀：{res['源语音前缀']} -> {res['目标语音前缀']}\n")
            f.write(f"说话人相似度：{sec:.4f}\n" if res[
                                                       '说话人相似度（SECS）'] is not None else "说话人相似度：计算失败\n")
            f.write(f"CER：{cer :.4f}\n" if res[
                                               '字符错误率（CER）'] is not None else "CER：计算失败\n")
            f.write(
                f"DNSMOS（SIG/BAK/OVRL）：{SIG:.4f}/{BAK:.4f}/{OVRL:.4f}\n" if
                res['DNSMOS整体质量（OVRL）'] is not None else "DNSMOS：计算失败\n")
            f.write("-" * 50 + "\n")

        f.write(f"该模型的平均sec为: {sum(sec_list) / len(sec_list) if len(sec_list) > 0 else 0}\n"
                f"该模型的平均cer为: {sum(cer_list) / len(cer_list) if len(cer_list) > 0 else 0}\n"
                f"该模型的平均SIG为: {sum(sig_list) / len(sig_list) if len(sig_list) > 0 else 0}\n"
                f"该模型的平均OVRL为: {sum(ovrl_list) / len(ovrl_list) if len(ovrl_list) > 0 else 0}\n"
                f"该模型的平均BAK为: {sum(bak_list) / len(bak_list) if len(bak_list) > 0 else 0}")
    print(f"\n所有任务评测完成！总结果已保存至：{total_result_path}")
    print(f"每个任务的单独结果已保存至各自文件夹下的 'evaluation_result.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="适配自定义文件夹结构的语音转换评测工具")
    parser.add_argument("--converted_root", default='/root/autodl-tmp/output/myfreevc_1',
                        help="转换语音的根目录（包含所有 '原前缀_to_目标前缀' 子文件夹）")
    parser.add_argument("--source_labels", default='AISHELL3_all.txt', help="源语音的根目录（包含 index.tsv 文件）")
    parser.add_argument("--xvector_extractor", default="resemblyzer",
                        choices=["resemblyzer", "wavlm"], help="说话人嵌入提取器类型")
    args = parser.parse_args()
    main(args)
