import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ruamel.yaml
import argparse
from pathlib import Path
import srt

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('--output-file', type=str, default='out/whisper/*.yml')
parser.add_argument('--language', type=str)
parser.add_argument('--word-accuracy', action='store_true')
args = parser.parse_args()
isCuda = torch.cuda.is_available() #and not args.word_accuracy
device = isCuda and 'cuda' or 'cpu'
torchDtype = isCuda and torch.float16 or torch.float32
modelId = 'openai/whisper-large-v3'
modelLoadingOptions = {
  'torch_dtype': torchDtype,
  'low_cpu_mem_usage': True,
  'use_safetensors': True,
  'attn_implementation': (isCuda and not args.word_accuracy) and 'flash_attention_2' or 'eager'
}
model = AutoModelForSpeechSeq2Seq.from_pretrained(
  modelId,
  **modelLoadingOptions
)
model.to(device)
processor = AutoProcessor.from_pretrained(modelId)
returnTimestamps = 'word' if args.word_accuracy else True
pipe = pipeline(
  'automatic-speech-recognition',
  model=model,
  tokenizer=processor.tokenizer,
  feature_extractor=processor.feature_extractor,
  max_new_tokens=128,
  chunk_length_s=30,
  batch_size=16,
  return_timestamps=returnTimestamps,
  torch_dtype=torchDtype,
  device=device,
)
inferenceOptions = {}
if args.language:
  inferenceOptions['language'] = args.language
result = pipe(
  args.input_file,
  generate_kwargs=inferenceOptions
)
ruamel=ruamel.yaml.YAML()
inputFileStem = Path(args.input_file).stem
outputFile = args.output_file.replace('*', inputFileStem)
outputFolder = Path(outputFile).parent
outputFolder.mkdir(parents=True, exist_ok=True)
outputFileExtension = Path(args.output_file).suffix[1:]
with open(outputFile, 'w') as file:
  if outputFileExtension == 'yml':
    ruamel.dump(result, file)
  if outputFileExtension == 'srt':
    srtEntries = []
    for i, chunk in enumerate(result['chunks']):
      srtEntries.append(srt.Subtitle(
        index=i + 1,
        start=srt.timedelta(seconds=chunk['timestamp'][0]),
        end=srt.timedelta(seconds=chunk['timestamp'][1]),
        content=chunk['text'].strip()
      ))
    fileContent = srt.compose(srtEntries)
    file.write(fileContent)
