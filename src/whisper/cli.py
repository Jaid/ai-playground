from math import trunc
import librosa
import torch
from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperProcessor, BatchFeature, WhisperTokenizer, WhisperFeatureExtractor
import argparse
from pathlib import Path
import time
import subprocess
from rich.console import Console
from rich.terminal_theme import MONOKAI as terminalTheme
from rich._export_format import CONSOLE_HTML_FORMAT as richHtmlExportFormat
from typing import List, Tuple, TypedDict
import re

Timestamp = Tuple[float, float]
class Chunk(TypedDict):
  timestamp: Timestamp
  text: str
class PipeOutput(TypedDict):
  chunks: List[Chunk]
  text: str

outFolder = Path(__file__).parent.parent.parent.joinpath('out', 'whisper')
outFolder.mkdir(parents=True, exist_ok=True)
try:
  console = Console(record=True, soft_wrap=True)
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file', type=str)
  parser.add_argument('--batch-size', type=int, default=1)
  parser.add_argument('--chunk-length', type=int)
  parser.add_argument('--force-cpu', action='store_true')
  parser.add_argument('--force-cuda', action='store_true')
  parser.add_argument('--language', type=str)
  parser.add_argument('--low-level', action='store_true')
  parser.add_argument('--low-memory', action='store_true')
  parser.add_argument('--max-new-tokens', type=int)
  parser.add_argument('--model', type=str, default='openai/whisper-large-v3')
  parser.add_argument('--output-file', type=str)
  parser.add_argument('--print-text', action='store_true')
  parser.add_argument('--resample-quality', type=int, default=28)
  parser.add_argument('--skip-ffmpeg', action='store_true')
  parser.add_argument('--timestamps', type=str, default='word')
  args = parser.parse_args()
  preparedInputFile = outFolder.joinpath('input.flac')
  ffprobeResult = subprocess.run([
    'ffprobe',
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    args.input_file
  ], stdout=subprocess.PIPE)
  console.out(ffprobeResult.args, ' → ', ffprobeResult.returncode)
  ffprobeResult.check_returncode()
  inputDuration = float(ffprobeResult.stdout)
  if not args.skip_ffmpeg:
    startTime = time.time()
    ffmpegResult = subprocess.run([
      'ffmpeg',
      '-hide_banner',
      '-strict',
      'experimental',
      '-hwaccel',
      'auto',
      '-y',
      '-i',
      args.input_file,
      '-af',
      f'aresample=resampler=soxr:precision={args.resample_quality}:dither_method=improved_e_weighted:out_sample_rate=16000:out_chlayout=mono',
      preparedInputFile
    ])
    console.out(ffmpegResult.args, ' → ', ffmpegResult.returncode)
    ffmpegResult.check_returncode()
    encodingDuration = time.time() - startTime
    encodingSpeed = inputDuration / encodingDuration * 100
    console.out(f'Prepared input file {args.input_file} in {encodingDuration} seconds ({trunc(encodingSpeed)}% speed)')
  startTime = time.time()
  if args.force_cuda:
    isCuda = True
  elif args.force_cpu:
    isCuda = False
  else:
    isCuda = torch.cuda.is_available()
  if isCuda:
    device = 'cuda'
    torchDtype = torch.float16
  else:
    device = 'cpu'
    torchDtype = torch.float32
  if isCuda and args.timestamps != 'word':
    attentionStrategy = 'flash_attention_2'
  else:
    attentionStrategy = 'eager'
  additionalModelLoadingArgs = {}
  if args.low_memory:
    additionalModelLoadingArgs['low_cpu_mem_usage'] = True
  model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=args.model,
    torch_dtype=torchDtype,
    use_safetensors=True,
    attn_implementation=attentionStrategy,
    **additionalModelLoadingArgs
  ) # type: ignore
  model.to(device) # type: ignore
  additionalTokenizerLoadingArgs = {}
  if args.timestamps:
    additionalTokenizerLoadingArgs['predict_timestamps'] = True
  tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.model,
    **additionalTokenizerLoadingArgs
  )
  additionalFeatureExtractorLoadingArgs = {}
  if args.chunk_length:
    additionalFeatureExtractorLoadingArgs['chunk_length'] = args.chunk_length
  featureExtractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path=args.model,
    **additionalFeatureExtractorLoadingArgs
  ) # type: ignore
  processor: WhisperProcessor = WhisperProcessor(
    feature_extractor=featureExtractor,
    tokenizer=tokenizer
  )
  [rawAudio, rawSampleRate] = librosa.load(preparedInputFile, sr=None)
  startTime = time.time()
  if args.low_level:
    rosaSampleRate = int(rawSampleRate)
    inputFeature: BatchFeature = processor.feature_extractor( # type: ignore
      raw_speech=rawAudio,
      do_normalize=True,
      sampling_rate=rosaSampleRate,
      return_tensors='pt',
      return_attention_mask=True
    )
    promptTokens = processor.get_decoder_prompt_ids(task='transcribe', no_timestamps=False)
    inputFeature.to(device, torchDtype)
    generatedData = model.generate(
      **inputFeature,
      return_timestamps=True,
      # return_token_timestamps=True,
    )
    result = processor.batch_decode(
      generatedData,
      skip_special_tokens=False,
      decode_with_timestamps=True,
      # time_precision=0.01
    )
    console.out(result)
  else:
    additionalPipeCreationArgs = {}
    if args.timestamps:
      additionalPipeCreationArgs['return_timestamps'] = args.timestamps
    if args.chunk_length:
      additionalPipeCreationArgs['chunk_length_s'] = args.chunk_length
    if args.max_new_tokens:
      additionalPipeCreationArgs['max_new_tokens'] = args.max_new_tokens
    elif args.chunk_length:
      additionalPipeCreationArgs['max_new_tokens'] = args.chunk_length * 5
    else:
      additionalPipeCreationArgs['max_new_tokens'] = 30 * 5
    pipe: AutomaticSpeechRecognitionPipeline = AutomaticSpeechRecognitionPipeline(
      model=model,
      tokenizer=processor.tokenizer, # type: ignore
      feature_extractor=processor.feature_extractor, # type: ignore
      batch_size=args.batch_size,
      torch_dtype=torchDtype,
      device=torch.device(device),
      **additionalPipeCreationArgs
    )
    inferenceOptions = {}
    if args.language:
      inferenceOptions['language'] = args.language
    console.out(f'Prepared model in {round(time.time() - startTime, 2)} seconds')
    startTime = time.time()
    output: PipeOutput = pipe(
      inputs=rawAudio,
      batch_size=args.batch_size,
      generate_kwargs=inferenceOptions
    ) # type: ignore
    console.out(output)
    result = output['chunks'] # type: ignore
  inferenceDuration = time.time() - startTime
  inferenceSpeed = inputDuration / inferenceDuration * 100
  console.out(f'Inference done in {round(inferenceDuration, 2)} seconds ({trunc(inferenceSpeed)}% speed)')
  inputFileStem = Path(args.input_file).stem
  outputs = {}
  if args.output_file:
    outputFile = args.output_file.replace('*', inputFileStem)
    srtOutputFolder = Path(outputFile).parent
    Path(srtOutputFolder).mkdir(parents=True, exist_ok=True)
    outputFileExtension = Path(args.output_file).suffix[1:]
    outputs[outputFileExtension] = outputFile
  else:
    srtOutputFolder = outFolder
    outputs['yml'] = f'{srtOutputFolder}/{inputFileStem}.yml'
    outputs['srt'] = f'{srtOutputFolder}/{inputFileStem}.srt'
  for outputFileExtension, outputFile in outputs.items():
    console.out('Writing', outputFileExtension, 'to', Path(outputFile))
    with open(outputFile, 'w') as file:
      if outputFileExtension == 'yml':
        import ruamel.yaml
        ruamel=ruamel.yaml.YAML()
        yamlContent = {
          'duration': inputDuration,
        }
        if args.low_level:
          yamlContent['text'] = result # type: ignore
        else:
          yamlContent['chunks'] = result # type: ignore
        ruamel.dump(yamlContent, file)
      if outputFileExtension == 'srt' and not args.low_level:
        import srt
        srtEntries = []
        for i, chunk in enumerate(result):
          givenStart = chunk['timestamp'][0]
          if givenStart == None or givenStart < 0:
            start = 0
          else:
            start = givenStart
          givenEnd = chunk['timestamp'][1]
          if givenEnd == None or givenEnd < inputDuration:
            end = inputDuration
          else:
            end = givenEnd
          srtEntries.append(srt.Subtitle(
            index=i + 1,
            start=srt.timedelta(seconds=start),
            end=srt.timedelta(seconds=end),
            content=chunk['text'].strip()
          ))
        fileContent = srt.compose(srtEntries)
        file.write(fileContent)
  if args.print_text:
    if args.low_level:
      text = result
    else:
      text = '\n'.join([chunk['text'] for chunk in result])
    console.out(text)
except Exception as e:
  console.print_exception()
  raise e
finally:
  console.out(locals())
  htmlFormat = re.sub('<pre .+', '<pre><code style="text-wrap: wrap; font-family:JetBrainsMono NF, JetBrains Mono, Symbols Nerd Font Mono, Symbols Nerd Font Mono Regular, monospace">{code}</code></pre>', richHtmlExportFormat)
  console.save_html(
    outFolder.joinpath('log.html').as_posix(),
    theme=terminalTheme,
    code_format=htmlFormat
  )
