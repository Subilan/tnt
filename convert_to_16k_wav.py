import ffmpeg
import argparse
import os

from utils.logutil import get_logger

log = get_logger('convert_16k')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用ffmpeg将任意音频文件转换为16K采样率的WAV音频以供后续处理')
    parser.add_argument('audio_input_path', help='输入音频文件的路径')
    parser.add_argument('-o', '--output', default=None, help='输出地址')

    args = parser.parse_args()

    if args.output is not None and not args.output.endswith('.wav'):
        print(f'输出地址必须是一个wav文件')
        exit(1)

    audio_input_path = args.audio_input_path
    audio_file_name, audio_file_ext = os.path.splitext(os.path.basename(audio_input_path))
    audio_16k_output_path = args.output if args.output is not None else os.path.join(os.path.dirname(audio_input_path),
                                                                                     f'{audio_file_name}_16k.wav')

    print(audio_16k_output_path)
    os.makedirs(os.path.dirname(audio_16k_output_path), exist_ok=True)

    # start_time = 0
    # duration = 240

    chain = (
        ffmpeg
        .input(audio_input_path)
        .output(filename=audio_16k_output_path, ac=1, ar=16000, acodec='pcm_s16le')
        .overwrite_output()
    )

    try:
        chain.run()
    except Exception as e:
        log.info(f'转换失败：{e}')

    log.info(f'转换成功：{audio_input_path} ---> {audio_16k_output_path}')
