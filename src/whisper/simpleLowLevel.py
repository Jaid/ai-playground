import torch
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer, BatchFeature, WhisperTimeStampLogitsProcessor
from pathlib import Path
from rich import print
import librosa

try:
  whisperFolder = Path('D:/git/ai-playground/out/whisper')
  isCuda = torch.cuda.is_available()
  device = isCuda and 'cuda' or 'cpu'
  torchDtype = isCuda and torch.float16 or torch.float32
  modelId = 'openai/whisper-large-v3'
  sampleRate = 16000
  chunkLength = 60
  maxNewTokens = chunkLength * 5
  model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
    modelId,
    use_safetensors=True,
    torch_dtype=torchDtype,
    attn_implementation='eager'
  ) # type: ignore
  model.to(device) # type: ignore
  tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
    modelId,
    predict_timestamps=True
  )
  featureExtractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(modelId) # type: ignore
  [rawAudio, rawSampleRate] = librosa.load(whisperFolder.joinpath('input.flac').as_posix(), sr=None)
  rosaSampleRate = int(rawSampleRate)
  inputFeature: BatchFeature = featureExtractor(
    raw_speech=rawAudio,
    # do_normalize=True,
    sampling_rate=rosaSampleRate,
    return_tensors='pt',
    return_attention_mask=True
  )
  inputFeature.to(device, torchDtype)
  generatedData = model.generate(
    **inputFeature,
    return_timestamps=True,
    # is_multilingual=True,
    time_precision=0.01,
    # return_token_timestamps=True,
    # return_dict_in_generate=True,
  )
  transcription = tokenizer.batch_decode(
    generatedData,
    skip_special_tokens=False,
    decode_with_timestamps=True,
    time_precision=0.01
  )
  # print(locals())
  print(transcription)
except:
  # print(locals())
  raise
