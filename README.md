<div align="center"><img height="100px" src="./minecraft_tnt.png" alt="A TNT from Minecraft"/>
<h1>tnt</h1>
<h6><em>Transcribe and Translate</em></h6>
<p>本地优先的 SRT 字幕生成与翻译脚本工具</p>
</div>

本项目主要提供一套完整的本地化字幕处理流程，包括：从音频文件转录生成字幕，将字幕翻译为目标语言，将目标语言回填到原字幕的相应位置。

本项目的目标是实现高度自动化的语音转录、字幕生成和字幕翻译，并尽可能确保字幕出现时机与语义的对应性。

本项目主要基于下列开源技术

- [**Whisper**](https://github.com/openai/whisper) 和 [**whisper-imestamped**](https://github.com/linto-ai/whisper-timestamped)，用于完成语音的转录。whisper 是 OpenAI 开源的语音识别模型，可以完成音频到 TXT、JSON、SRT 等的转换。whisper-timestamped 支持提取出精细到单词级别的转录时间戳。
- [**Silero VAD**](https://github.com/snakers4/silero-vad) 是一个轻量级高准确率的语音活动检测模型，用于精确识别人声片段，在本项目中用于初步断句检测
- [**TranslateGemma**](https://huggingface.co/collections/google/translategemma) 是 Google 在 2026 年 1 月推出的翻译模型，本项目主要通过 Ollama 本地部署方式调用，以实现在端侧设备上基于 LLM 的机器翻译
- [**HanLP**](https://github.com/hankcs/HanLP) 是一套 NLP 工具包，主要用于对译得文本进行分词和词性标注
- **ffmpeg** 用于音频的转换和处理
- [**pysrt**](https://github.com/byroot/pysrt) 用于对 SRT 字幕文件的读写和操作

## 特点

- 本地优先。所有核心模型（whisper、Silero VAD、TranslateGemma）均可以在边缘设备正常运行，无需联网或调用外部 API。音频数据和处理结果完全保留在本地，适合处理敏感内容、在离线环境中使用或者单纯节省成本。
- 使用语音活动检测。使用 Silero VAD 检测音频中的人声片段，仅在有语音的区间生成字幕，避免在静音、背景音区域产生无效字幕，同时提高时间轴的对应程度和自然感。
- 带回退的智能回填。这主要用于实现双语字幕。由于语言之间存在语序区别，语义对齐的结果看起来不能使人满意。目前主要采用借助词性标注信息确定句子的合理断点后，大致按照字幕片段的持续时间比例进行回填的简单流程。
- 字幕整理。包括对字幕中不需要的标点符号的清理、字幕句尾不合理的多余词语的移动、时间轴的紧凑（避免闪烁）等。

## 使用方法

### 1. 音频格式转换

将任意格式的音频转换为 16kHz 单声道 WAV 格式，作为后续处理的输入：

```bash
python convert_to_16k_wav.py input.m4a
```

输出文件默认保存在输入文件同级目录，命名为 `input_16k.wav`。如需指定输出路径：

```bash
python convert_to_16k_wav.py input.m4a -o output/converted.wav
```

### 2. 语音转录

对音频进行转录并生成 SRT 字幕文件：

```bash
python transcribe.py input_16k.wav -o output.srt
```

上述命令使用默认的 `turbo` 模型，且自动检测音频语言。如需指定语言和模型：

```bash
python transcribe.py input_16k.wav -lang en -m medium -o output.srt
```

参数说明：

- `-lang` / `--language`：音频语言代码（如 `en`、`zh`、`ja`），如果不填默认为 `autodetect` 自动检测。自动检测会使用 whisper 截取音频的前 30 秒进行语言检测。
- `-m` / `--model`：whisper 模型，可选 `tiny`、`base`、`small`、`medium`、`large`、`turbo`，默认且推荐使用 `turbo`。有关各个模型的参数及显存占用，请参考 whisper repo 的 [Available models and languages](https://github.com/openai/whisper#available-models-and-languages) 一节
- `-o` / `--output`：输出 SRT 文件路径，默认在当前目录且与输入文件同名
- `-oj` / `--with-json-output`：同时输出 JSON 格式的转录结果
- `-otxt` / `--with-txt-output`：同时输出纯文本（TXT）格式的转录结果

### 3. 字幕翻译

> [!WARNING]
> 
> 目前仅测试了英文→中文的翻译流程，其余语言的效果可能有所偏差或不可用。
> 这主要与脚本中当前采用的语言隐含信息有关，例如标点符号和用于识别单词的正则表达式。

将 SRT 字幕翻译为目标语言：

```bash
python translate.py input.srt -fromlang en -tolang zh -o translated.srt
```

如需指定 TranslateGemma 模型大小和 Ollama 服务地址：

```bash
python translate.py input.srt -fromlang en -tolang zh -b 12b -o translated.srt --ollama-host http://192.168.1.100:11434
```

参数说明：

- `-fromlang` / `--source-language`：源语言代码（如 `en`、`ja`、`zh`），必填。支持的语言代码见 `translate_gemma_language_codes.py` 中的 `LANGUAGE_MAP` 或者 [Ollama 页面](https://ollama.com/library/translategemma)
- `-tolang` / `--target-language`：目标语言代码，必填
- `-b` / `--translate-gemma-size`：TranslateGemma 模型参数量，可选 `4b`、`12b`、`27b`、`latest`，默认 `4b`；latest 的具体含义见 Ollama 页面，截至目前 latest 等价于 4b。
- `--ollama-host`：Ollama 服务地址，默认 `http://127.0.0.1:11434`
- `-o` / `--output`：输出 SRT 文件路径，必填

### 完整流程示例

从原始音频文件到翻译后的字幕，完整流程如下：

```bash
python convert_to_16k_wav.py lecture.m4a
python transcribe.py lecture_16k.wav -lang en -o lecture_en.srt
python translate.py lecture_en.srt -fromlang en -tolang zh -o lecture_zh.srt
```

## 环境准备

### 依赖安装

首先使用 conda 或 uv 或 venv 创建虚拟环境。然后从 requirements.txt 安装依赖。

```bash
pip install -r requirements.txt
```

### Ollama 配置

翻译功能需要本地运行 Ollama 服务并拉取 TranslateGemma 模型：

```bash
# 拉取 TranslateGemma 模型（根据需求选择大小）
ollama pull translategemma:4b
ollama pull translategemma:12b
ollama pull translategemma:27b # 不建议：消费级设备不太能运行这个参数的模型

ollama serve # 默认端口是 11434
```

### 关于模型下载

除了 TranslateGemma 外，模型在首次使用之前都需要下载，此过程会自动进行。模型下载后将存储在本地机器上，并占用空间。脚本会自动下载的模型包括：

- whisper，首次运行 transcribe.py 时下载
- HanLP 分词模型 [COARSE_ELECTRA_SMALL_ZH](https://file.hankcs.com/hanlp/tok/coarse_electra_small_20220616_012050.zip)，首次运行 translate.py 时下载，下同
- HanLP 词性标注模型 [CTB9_POS_ELECTRA_SMALL](https://file.hankcs.com/hanlp/pos/pos_ctb_electra_small_20220215_111944.zip)

如果遇到下载速度慢，无法正常下载等问题，可单击模型名称使用官方链接手动下载。HanLP 的模型 zip 压缩包统一放在 `~/.hanlp/tok`（分词）或 `~/.hanlp/pos`（标注）下。

模型大小参考：

```shell
find ~/.cache/whisper/* ~/.hanlp/* | grep .pt | xargs du -sh

1.5G	/Users/***/.cache/whisper/large-v3-turbo.pt
 47M	/Users/***/.hanlp/pos/pos_ctb_electra_small_20220215_111944/model.pt
 47M	/Users/***/.hanlp/tok/coarse_electra_small_20220616_012050/model.pt
```

```shell
ollama list

NAME                            ID              SIZE      MODIFIED
translategemma:12b              c2f9a9ca1ec7    8.1 GB    11 days ago
translategemma:latest           c49d986b0764    3.3 GB    12 days ago
```

## 项目结构

```
tnt/
├── convert_to_16k_wav.py      # 音频格式转换
├── transcribe.py              # 语音转录生成字幕
├── translate.py               # 字幕翻译
├── translate_gemma_language_codes.py  # 语言代码映射
├── utils/
│   ├── logutil.py             # 日志工具
│   └── listutil.py            # 列表工具函数
├── cache/                     # 缓存目录（自动创建）
└── README.md
```

## 贡献

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/your-feature`）
3. 提交更改（`git commit -m 'Add some feature'`）
4. 推送到分支（`git push origin feature/your-feature`）
5. 提交 Pull Request

提交代码前请确保：

- 必要时添加或更新注释
- 验证功能正常工作，尤其是使用了 AI 编写代码时

## 其它

目前，本工具处于验证阶段，欢迎试用、提供建议。

此项目大概属于是兴趣项目，没有多少专业成分，还请不吝赐教。

## 协议

MIT
