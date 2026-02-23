import argparse
import os.path
import re
from math import ceil

import pysrt
import silero_vad
import whisper
import whisper_timestamped
from joblib import Memory
from pydantic import BaseModel
from pysrt import SubRipItem

from utils.cacheutil import default_cache_callback
from utils.logutil import get_logger

log = get_logger('transcribe')

memory = Memory(location='./cache', verbose=0)


@memory.cache(ignore=['whisper_model', 'audio'], cache_validation_callback=default_cache_callback)
def transcribe(whisper_model, audio, audio_language, _cache_ref_audio_path=None):
    return whisper_timestamped.transcribe(whisper_model, audio, language=audio_language)


@memory.cache(cache_validation_callback=default_cache_callback)
def get_speech_timestamps(audio, model):
    return silero_vad.get_speech_timestamps(
        audio,
        model,
        return_seconds=True,
        speech_pad_ms=20,
        min_speech_duration_ms=150,
        min_silence_duration_ms=40
    )


class WhisperWord(BaseModel):
    """
    表示whisper_timestamped识别出的单个单词
    """
    text: str  # 文本内容
    start: float  # 开始时间
    end: float  # 结束时间
    confidence: float  # 置信度


class TalkingSegment(BaseModel):
    """
    表示VAD识别出的说话片段
    """
    start: float  # 开始时间（秒）
    end: float  # 结束时间（秒）
    content: str = ''  # 片段内容


def break_talking_segments(segments: list[TalkingSegment], duration_threshold: float) -> list[TalkingSegment]:
    """
    将时长超过阈值的说话片段分割为多个时长不超过阈值的小片段。

    Args:
        segments: 原始说话片段列表
        duration_threshold: 最大允许时长（秒）

    Returns:
        分割后的说话片段列表
    """
    result = []
    for seg in segments:
        duration = seg.end - seg.start
        if duration <= duration_threshold:
            result.append(seg)
        else:
            # 计算需要分割成的段数（向上取整）
            num_parts = ceil(duration / duration_threshold)
            for i in range(num_parts):
                part_start = seg.start + i * duration_threshold
                part_end = min(seg.start + (i + 1) * duration_threshold, seg.end)
                # 创建新片段，content暂时沿用原始内容（实际使用中可能需要按时间切分）
                new_seg = TalkingSegment(
                    start=part_start,
                    end=part_end,
                    content=seg.content
                )
                result.append(new_seg)
    return result


def assign_words_to_talking_segments(words: list[WhisperWord], segments: list[TalkingSegment], bias: float = 0.1) -> \
        list[TalkingSegment]:
    """
    根据时间戳信息，将words中的每一项对应到segments中，以拼接的形式反映于segment.content字段上。
    :param words: 所有单词信息
    :param segments: 所有活动片段
    :param bias: 时间误差（秒）
    :return: 补充了content字段值的TalkingSegment数组
    """
    segments = segments[:]

    def word_relation_to_segment(word: WhisperWord, seg: TalkingSegment, bias: float):
        if word.end <= seg.start - bias:
            return "left"
        elif word.start >= seg.end + bias:
            return "right"
        elif seg.start - bias <= word.start and word.end <= seg.end + bias:
            return "inside"
        elif word.start < seg.start - bias and word.end <= seg.end + bias:
            return "partial_left"
        elif seg.start - bias <= word.start and word.end > seg.end + bias:
            return "partial_right"
        else:
            return "unknown"

    word_ptr = 0
    word_buf = []
    next_word_buf = []

    for seg in segments:
        while word_ptr < len(words):
            word = words[word_ptr]
            relation = word_relation_to_segment(word, seg, bias)
            if relation == 'inside':
                word_buf.append(word.text)
            elif relation == 'partial_left':
                word_buf.append(word.text)
            elif relation == 'partial_right':
                next_word_buf.append(word.text)
            elif relation == 'left':
                word_buf.append(word.text)
            elif relation == 'right':
                break
            word_ptr += 1
        seg.content = ' '.join(word_buf)
        word_buf = next_word_buf
        next_word_buf = []

    return segments


def postprocess_srts_move_extra_trailing_words(srts: list[SubRipItem]):
    """
    将字幕中处于标点符号后的末尾单词移动到下一行字幕的开头。注：此函数原地修改srts。
    :param srts: 所有字幕项
    :return: 移动后的字幕项
    """
    idx = 0

    while idx < len(srts):
        sub = srts[idx]
        sub.text = sub.text.strip()
        # 注意避免识别句尾的数字 eg. "1,234" 以及短句 eg. "and,"
        m = re.search(r"(\w+[,?.])\s*([A-Za-z]+)\s*$", sub.text)
        if idx < len(srts) - 1 and m:
            log.info(f'在句子 <{sub.text}> 中找到了多余的单词 <{m.group(2)}>')
            sub.text = sub.text.replace(m.group(0), m.group(1))
            extra_word = m.group(2)
            srts[idx + 1].text = extra_word + ' ' + srts[idx + 1].text
        idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对音频进行转录并输出SRT')
    parser.add_argument('input', help='输入音频文件')
    parser.add_argument('-lang', '--language', help='音频语言', default='autodetect')
    parser.add_argument('-o', '--output', help='输出路径', default=None)
    parser.add_argument('-oj', '--with-json-output', help='是否输出JSON', action='store_true')
    parser.add_argument('-otxt', '--with-txt-output', help='是否输出TXT', action='store_true')
    parser.add_argument('-m', '--model', choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                        help='Whisper模型', default='turbo')
    args = parser.parse_args()

    output_path = args.output

    if output_path is not None and not output_path.lower().endswith('.srt'):
        log.error('输出路径必须指向一个SRT文件')
        exit(1)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.input),
                                   os.path.splitext(os.path.basename(args.input))[0] + '.srt')

    log.info('加载音频活动检测（VAD）模型')
    vad_model = silero_vad.load_silero_vad()
    log.info('加载Whisper模型')
    whisper_model = whisper.load_model(args.model)
    log.info('加载音频')
    audio = whisper.load_audio(args.input)
    vad_audio = silero_vad.read_audio(args.input)

    audio_language = args.language
    if audio_language == 'autodetect':
        log.info('未明确指定音频语言，自动检测中...')
        audio_trimmed = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_trimmed, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)
        _, lang_probs = whisper_model.detect_language(mel)
        # lang_probs is a dict
        # noinspection PyUnresolvedReferences
        audio_language = max(lang_probs, key=lang_probs.get)
        log.info(f'检测出的音频语言为 {audio_language}')

    log.info('转录中...')
    whisper_result = transcribe(whisper_model, audio, audio_language, _cache_ref_audio_path=args.input)

    log.info('检测音频活动中...')
    vad_speech_timestamps = get_speech_timestamps(vad_audio, vad_model)

    log.info('合成SRT字幕...')
    whisper_words: list[WhisperWord] = [WhisperWord(**word) for seg in whisper_result['segments'] for word in
                                        seg['words']]
    talking_segments = [TalkingSegment(**seg) for seg in vad_speech_timestamps]
    talking_segments = break_talking_segments(talking_segments, duration_threshold=5)

    assigned_talking_segments = assign_words_to_talking_segments(whisper_words, talking_segments, 0.1)

    srts = [
        SubRipItem(start=int(raw.start * 1000), end=int(raw.end * 1000), index=i, text=raw.content)
        for i, raw in enumerate(assigned_talking_segments)
    ]

    postprocess_srts_move_extra_trailing_words(srts)
    srt_file = pysrt.SubRipFile(items=srts)
    srt_file.save(output_path)
    log.info(f'字幕结果已经保存到 {output_path}')
