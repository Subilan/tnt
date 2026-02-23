import argparse
import bisect
import concurrent
import os
from concurrent.futures import ThreadPoolExecutor

import hanlp
import pysrt
from joblib import Memory
from ollama import Client
from pydantic import BaseModel
from pysrt import SubRipItem

from translate_gemma_language_codes import get_language_name_from_code, LANGUAGE_MAP
from utils.cacheutil import default_cache_callback
from utils.logutil import get_logger

log = get_logger('translate')

memory = Memory(location='./cache', verbose=0)


def translate_gemma_prompt(source_lang_code: str, target_lang_code: str,
                           text: str):
    prompt = """
You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


{TEXT}
"""
    return prompt.format(
        SOURCE_LANG=get_language_name_from_code(source_lang_code),
        SOURCE_CODE=source_lang_code,
        TARGET_LANG=get_language_name_from_code(target_lang_code),
        TARGET_CODE=target_lang_code,
        TEXT=text
    )


@memory.cache(ignore=['ollama_client'], cache_validation_callback=default_cache_callback)
def translate(ollama_client: Client, text: str, translate_gemma_size: str, source_lang_code: str,
              target_lang_code: str):
    resp = ollama_client.chat(model=f'translategemma:{translate_gemma_size}', messages=[
        {
            'role': 'user',
            'content': translate_gemma_prompt(source_lang_code, target_lang_code, text)
        }
    ])
    return resp.message.content


class Slot(BaseModel):
    """
    表示一个字幕片段
    """
    idx: int  # 字幕片段在整个SRT中的编号
    origin: str  # 字幕片段原文
    duration: int  # 字幕片段持续时间，单位秒
    origin_tok: list[str] = []  # 字幕片段原文分词数组
    translation: str = ''  # 字幕片段翻译文本（回填结果）


class Sentence(BaseModel):
    """
    表示从多个字幕中拼合成的一个句子
    """
    text: str  # 完整句子
    slots: list[Slot]  # 该句子对应的所有字幕片段
    translation: str = ''  # 该完整句子的翻译结果
    translation_tok: list[str] = []  # 完整句子翻译结果的分词数组
    translation_pos: list[str] = []  # 分词数组的词性标注结果


def subs_to_sentences(subs: list[SubRipItem]) -> list[Sentence]:
    """
    将subs中的字幕拼合成句子用于翻译，同时记录这些句子原本所属的字幕片段存于Sentence对象上
    :param subs: 所有字幕项
    :return: 构建的句子项目
    """
    ending_puncts = ['.', '?', '!']
    sentence_text_buf = ''
    sentence_positions_buf: list[Slot] = []
    sentences: list[Sentence] = []

    for i, sub in enumerate(subs):
        sub_text = sub.text.strip()
        if len(sub_text) == 0:
            continue
        last_ch = sub_text[-1:]
        sentence_text_buf += ' ' + sub_text
        sentence_positions_buf.append(Slot(idx=i, origin=sub_text, duration=sub.duration.ordinal))
        if last_ch in ending_puncts:
            sentences.append(Sentence(text=sentence_text_buf.strip(), slots=sentence_positions_buf))
            sentence_text_buf = ''
            sentence_positions_buf = []

    return sentences


def translate_all(
        sentences: list[Sentence],
        ollama_client: Client,
        translate_gemma_size: str,
        source_language: str,
        target_language: str,
        max_workers=5,
        verbose=True
):
    """
    翻译sentences中的所有项目，并将翻译结果填入到对象的translation字段。注：此函数原地修改sentences。
    :param translate_gemma_size: TranslateGemma模型参数量
    :param source_language: 源语言代码
    :param target_language: 目标语言代码
    :param max_workers: ThreadPoolExecutor最大worker数量，默认5
    :param sentences: 所有句子
    :param ollama_client: Ollama客户端
    :param verbose: 是否显示进度
    :return:
    """
    progress_numerator = 0
    progress_denominator = len(sentences)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate, ollama_client, sentence.text, translate_gemma_size, source_language,
                                   target_language): i for
                   i, sentence in enumerate(sentences)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            sentences[i].translation = future.result()
            progress_numerator += 1
            if verbose:
                log.info(
                    f'翻译进度: {progress_numerator / progress_denominator * 100:.1f}%: {sentences[i].text} ---> {sentences[i].translation}')


def get_breakpoints_from_pos(pos_list: list[str]) -> set[int]:
    """根据词性标注返回可断点位置（词之间，索引从1到len-1）"""
    breakpoints = set()
    for i, tag in enumerate(pos_list):
        if tag == 'MSP':  # 其他小品词，如“来”
            breakpoints.add(i)  # 在该词之前断开
        elif tag == 'PU':  # 标点
            breakpoints.add(i + 1)  # 在该标点之后断开
        elif tag == 'DEG':  # 属格“的”
            breakpoints.add(i + 1)  # 在“的”之后断开
        elif tag == 'CC':  # 连词，如“和”
            breakpoints.add(i)  # 在该连词之前断开
        elif tag == 'NR':  # 专有名词
            breakpoints.add(i)  # 在该专有名词之前断开
    # 过滤掉边界（0 和 len）
    return {b for b in breakpoints if 0 < b < len(pos_list)}


def backfill_by_duration_and_breakpoints(sentences: list[Sentence]) -> None:
    """
    将每个句子的翻译结果按各字幕片段的时长比例分配到对应的 Slot 上，
    并设置 slot.translation 为拼接后的中文字符串。
    切分点只能位于由词性标注确定的允许位置。
    若允许断点不足，回退到无限制切分。
    """

    def unrestricted_split_by_duration(slots: list[Slot], tokens: list[str]) -> None:
        """
        根据时长比例将 tokens 无限制地分配到各 slot 上，直接修改 slot.translation。
        用于允许断点不足时的回退方案。
        """
        token_count = len(tokens)
        if token_count == 0:
            for slot in slots:
                slot.translation = ''
            return

        total_duration = sum(slot.duration for slot in slots)
        # 根据时长比例计算期望词数
        target_counts = [token_count * slot.duration / total_duration for slot in slots]
        base_counts = [int(c) for c in target_counts]  # 整数部分
        remainder = token_count - sum(base_counts)  # 剩余待分配词数

        # 按小数部分从大到小排序，分配剩余词数
        fractions = [(i, target_counts[i] - base_counts[i]) for i in range(len(slots))]
        fractions.sort(key=lambda x: x[1], reverse=True)
        counts = base_counts[:]
        for i in range(remainder):
            idx = fractions[i][0]
            counts[idx] += 1

        # 拼接并赋值
        start = 0
        for i, slot in enumerate(slots):
            end = start + counts[i]
            slot.translation = ''.join(tokens[start:end])
            start = end

    for sentence in sentences:
        slots = sentence.slots
        if not slots:
            continue

        trans_tokens = sentence.translation_tok
        token_count = len(trans_tokens)

        if token_count == 0:
            for slot in slots:
                slot.translation = ''
            continue

        n_slots = len(slots)
        needed_cuts = n_slots - 1

        # 特殊情况：只有一个字幕片段，直接分配全部词语
        if needed_cuts == 0:
            slots[0].translation = ''.join(trans_tokens)
            continue

        # 获取允许的切分点（词之间），并排序
        allowed_cuts = sorted(get_breakpoints_from_pos(sentence.translation_pos))
        allowed_cuts = [c for c in allowed_cuts if 1 <= c < token_count]

        # 如果允许切分点不足，回退到无限制切分
        if len(allowed_cuts) < needed_cuts:
            log.warning(
                f"句子 '{sentence.text[:30]}...' 的允许断点不足 (需要 {needed_cuts}, 实际 {len(allowed_cuts)})，将使用无限制切分"
            )
            unrestricted_split_by_duration(slots, trans_tokens)
            continue

        total_duration = sum(slot.duration for slot in slots)

        # 计算期望累积切分位置（基于时长比例）
        cum_duration = 0
        target_cum = []  # 期望的累积词数（浮点）
        for i in range(needed_cuts):
            cum_duration += slots[i].duration
            target_cum.append(cum_duration / total_duration * token_count)

        # 贪心选择最接近期望的切分点，同时保证严格递增
        selected_cuts = []
        prev = 0
        for target in target_cum:
            # 在 allowed_cuts 中查找大于 prev 且最接近 target 的点
            idx = bisect.bisect_left(allowed_cuts, target)
            candidates = []
            if idx < len(allowed_cuts) and allowed_cuts[idx] > prev:
                candidates.append(allowed_cuts[idx])
            if idx - 1 >= 0 and allowed_cuts[idx - 1] > prev:
                candidates.append(allowed_cuts[idx - 1])

            # 若无满足 > prev 的候选，则取第一个大于 prev 的点
            if not candidates:
                idx2 = bisect.bisect_right(allowed_cuts, prev)
                if idx2 < len(allowed_cuts):
                    candidates.append(allowed_cuts[idx2])

            if not candidates:
                # 无法找到合法切分点，回退到无限制切分
                log.warning(
                    f"句子 '{sentence.text[:30]}...' 无法找到合适的切分点，将使用无限制切分"
                )
                unrestricted_split_by_duration(slots, trans_tokens)
                break  # 跳出循环，当前句子处理完毕
            else:
                best = min(candidates, key=lambda x: abs(x - target))
                selected_cuts.append(best)
                prev = best
        else:
            # 成功选择了 needed_cuts 个切分点，计算每个 slot 的词数并赋值
            counts = [selected_cuts[0]] + \
                     [selected_cuts[i] - selected_cuts[i - 1] for i in range(1, len(selected_cuts))] + \
                     [token_count - selected_cuts[-1]]
            start = 0
            for i, slot in enumerate(slots):
                end = start + counts[i]
                slot.translation = ''.join(trans_tokens[start:end])
                start = end
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将SRT文件中的内容使用本地模型进行翻译，并分配到原有字幕上')
    parser.add_argument('input_srt', help='输入的字幕文件')
    parser.add_argument('--ollama-host', default='http://127.0.0.1:11434', help='Ollama主机地址（包含协议、端口）')
    parser.add_argument('-b', '--translate-gemma-size', choices=['4b', '12b', '27b', 'latest'], default='4b',
                        help='TranslateGemma翻译模型的参数量')
    parser.add_argument('-fromlang', '--source-language', choices=LANGUAGE_MAP.keys(), required=True)
    parser.add_argument('-tolang', '--target-language', choices=LANGUAGE_MAP.keys(), required=True)
    parser.add_argument('-o', '--output', required=True, help='输出SRT文件路径')

    args = parser.parse_args()

    if not args.input_srt.lower().endswith('.srt'):
        log.error('输入路径必须指向一个SRT文件')
        exit(1)

    if not args.output.lower().endswith('.srt'):
        log.error('输出路径必须指向一个SRT文件')
        exit(1)

    ollama_client = Client(host=args.ollama_host)

    log.info('读取原始字幕文件')
    subs: list[SubRipItem] = pysrt.open(args.input_srt)
    sentences = subs_to_sentences(subs)

    log.info(f'使用 translategemma:{args.translate_gemma_size} 进行翻译任务')

    translate_all(sentences=sentences, ollama_client=ollama_client, source_language=args.source_language,
                  target_language=args.target_language, translate_gemma_size=args.translate_gemma_size)

    log.info('加载分词引擎...')
    hanlp_tokenize = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    hanlp_split_and_tokenize = hanlp.pipeline() \
        .append(hanlp.utils.rules.split_sentence) \
        .append(hanlp_tokenize) \
        .append(lambda s: sum(s, []))
    log.info('加载词性标注引擎...')
    hanlp_pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

    log.info('分词与标注...')
    for sentence in sentences:
        # noinspection PyTypeChecker
        sentence.translation_tok = hanlp_split_and_tokenize(sentence.translation)
        sentence.translation_pos = hanlp_pos(sentence.translation_tok)
        for slot in sentence.slots:
            slot.origin_tok = hanlp_tokenize(slot.origin)

    backfill_by_duration_and_breakpoints(sentences)

    # 构建新的字幕项列表
    output_items = []
    for sentence in sentences:
        for slot in sentence.slots:
            original_sub = subs[slot.idx]  # 从原始字幕获取时间信息
            new_item = pysrt.SubRipItem(
                index=0,  # 临时索引，稍后重新编号
                start=original_sub.start,
                end=original_sub.end,
                text=slot.translation
            )
            output_items.append(new_item)

    # 重新设置正确的连续索引（从1开始）
    for i, item in enumerate(output_items, start=1):
        item.index = i

    srt_file = pysrt.SubRipFile(items=output_items)
    srt_file.save(args.output, encoding='utf-8')
    log.info(f'字幕结果已经保存到 {args.output}')
