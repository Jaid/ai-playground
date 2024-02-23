from calendar import c
from math import trunc
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutomaticSpeechRecognitionPipeline, WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
import argparse
from pathlib import Path
import time
import subprocess
from rich.console import Console
from rich.terminal_theme import MONOKAI as terminalTheme

console = Console(record=True, soft_wrap=True)
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--chunk-length', type=int, default=60)
parser.add_argument('--force-cpu', action='store_true')
parser.add_argument('--force-cuda', action='store_true')
parser.add_argument('--language', type=str)
parser.add_argument('--low-memory', action='store_true')
parser.add_argument('--max-new-tokens', type=int)
parser.add_argument('--model', type=str, default='openai/whisper-large-v3')
parser.add_argument('--output-file', type=str)
parser.add_argument('--print-text', action='store_true')
parser.add_argument('--resample-quality', type=int, default=28)
parser.add_argument('--word-accuracy', action='store_true')
args = parser.parse_args()
console.out('Args:', args)
startTime = time.time()
Path('out/whisper').mkdir(parents=True, exist_ok=True)
preparedInputFile = 'out/whisper/input.flac'
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
console.out('Input duration:', inputDuration, 'seconds')
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
  f'aresample=resampler=soxr:precision={args.resample_quality}:osr=16000',
  '-ac',
  '1',
  preparedInputFile
])
console.out(ffmpegResult.args, ' → ', ffmpegResult.returncode)
ffmpegResult.check_returncode()
console.out(f'Prepared input file {args.input_file} in {time.time() - startTime:.2f} seconds')
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
if isCuda and not args.word_accuracy:
  attentionStrategy = 'flash_attention_2'
else:
  attentionStrategy = 'eager'
modelLoadingOptions = {
  'torch_dtype': torchDtype,
  'use_safetensors': True,
  'attn_implementation': attentionStrategy
}
if args.low_memory:
  modelLoadingOptions['low_cpu_mem_usage'] = True
model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
  pretrained_model_name_or_path=args.model,
  **modelLoadingOptions
)
model.to(device)
processor: WhisperProcessor = WhisperProcessor.from_pretrained(args.model)
returnTimestamps = 'word' if args.word_accuracy else True
pipe: AutomaticSpeechRecognitionPipeline = AutomaticSpeechRecognitionPipeline(
  model=model,
  tokenizer=processor.tokenizer,
  feature_extractor=processor.feature_extractor,
  max_new_tokens=args.max_new_tokens or args.chunk_length * 5,
  chunk_length_s=args.chunk_length,
  batch_size=args.batch_size,
  return_timestamps=returnTimestamps,
  torch_dtype=torchDtype,
  device=torch.device(device),
)
inputFeatures = processor(
  file=preparedInputFile,
  return_tensors='pt'
)
inferenceOptions = {}
if args.language:
  inferenceOptions['language'] = args.language
console.out(f'Prepared model in {round(time.time() - startTime, 2)} seconds')
startTime = time.time()
def data():
  while True:
      # This could come from a dataset, a database, a queue or HTTP request
      # in a server
      # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
      # to use multiple threads to preprocess data. You can still have 1 thread that
      # does the preprocessing while the main runs the big inference
      yield "This is a test"
result = pipe(
  preparedInputFile,
  generate_kwargs=inferenceOptions
)
inferenceDuration = time.time() - startTime
inferenceSpeed = inputDuration / inferenceDuration * 100
console.out(f'Inference done in {round(inferenceDuration, 2)} seconds ({trunc(inferenceSpeed)}% speed)')
inputFileStem = Path(args.input_file).stem
outputs = {}
if args.output_file:
  outputFile = args.output_file.replace('*', inputFileStem)
  outputFolder = Path(outputFile).parent
  outputFileExtension = Path(args.output_file).suffix[1:]
  outputs[outputFileExtension] = outputFile
else:
  outputFolder = 'out/whisper'
  outputs['yml'] = f'{outputFolder}/{inputFileStem}.yml'
  outputs['srt'] = f'{outputFolder}/{inputFileStem}.srt'
Path(outputFolder).mkdir(parents=True, exist_ok=True)
for outputFileExtension, outputFile in outputs.items():
  console.out('Writing', outputFileExtension, 'to', Path(outputFile))
  with open(outputFile, 'w') as file:
    if outputFileExtension == 'yml':
      import ruamel.yaml
      ruamel=ruamel.yaml.YAML()
      ruamel.dump(result, file)
    if outputFileExtension == 'srt':
      import srt
      srtEntries = []
      for i, chunk in enumerate(result['chunks']):
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
  console.out(result['text'].strip())
console.out({
  'model': model,
  'pipe': pipe,
  'processor': processor
})
console.save_html(
  'out/whisper/log.html',
  theme=terminalTheme
)
