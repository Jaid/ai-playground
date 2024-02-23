import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path

whisperFolder = Path('D:/git/ai-playground/out/whisper')
isCuda = torch.cuda.is_available()
device = isCuda and 'cuda' or 'cpu'
torchDtype = isCuda and torch.float16 or torch.float32
modelId = 'openai/whisper-large-v3'
chunkLength = 60
maxNewTokens = chunkLength * 5
modelLoadingOptions = {
  'torch_dtype': torchDtype,
  'use_safetensors': True,
  'attn_implementation': 'eager',
}
model = AutoModelForSpeechSeq2Seq.from_pretrained(
  modelId,
  **modelLoadingOptions
)
model.to(device)
processor = AutoProcessor.from_pretrained(modelId)
pipe = pipeline(
  'automatic-speech-recognition',
  model=model,
  tokenizer=processor.tokenizer,
  feature_extractor=processor.feature_extractor,
  max_new_tokens=maxNewTokens,
  chunk_length_s=chunkLength,
  batch_size=1,
  return_timestamps='word',
  torch_dtype=torchDtype,
  device=device,
)
result = pipe(
  whisperFolder.joinpath('input.flac').as_posix()
)
print(result['text'])
