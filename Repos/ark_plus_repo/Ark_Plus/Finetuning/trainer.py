from utils import MetricLogger, ProgressLogger
from models import build_classification_model
import time
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

# args.amp_dtype is a string ("bf16" [default], "fp16", or "fp32"/None to disable AMP
# entirely) so it round-trips cleanly through argparse/experiment configs. fp16 needs
# gradient scaling to avoid underflow (bf16's wider exponent range doesn't); ConvNeXt V2's
# depthwise convs hit a cuDNN "no engine found" error under bf16 on at least some GPU/cuDNN
# combinations (verified locally) but work fine under fp16, hence the per-model override.
def resolve_amp_dtype(args):
  name = getattr(args, "amp_dtype", "bf16")
  if name in (None, "fp32"):
    return None
  return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def train_one_epoch(data_loader_train, device, model, criterion, optimizer, epoch, amp_dtype=torch.bfloat16, scaler=None, accum_steps=1):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  # Gradient accumulation (accum_steps > 1) keeps the protocol's batch of 64 on a GPU that can't hold
  # 64 images at once: e.g. two micro-batches of 32 make one optimizer step over the same 64 images
  # (the shuffled order doesn't depend on batch size). Each micro-batch's mean loss is weighted by its
  # share of the group's samples, so the gradient equals the 64-image batch's -- also for the smaller
  # last group of an epoch. Exact for LayerNorm models; NOT for BatchNorm ones (DenseNet-121), whose
  # batch statistics would come from 32 images. accum_steps=1 is the original code path, unchanged.
  _n_total = len(data_loader_train.dataset)
  _group_size = data_loader_train.batch_size * accum_steps

  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    # non_blocking=True overlaps the host->device copy with other GPU work instead of
    # blocking on it -- only actually async when the source tensor is pinned, which the
    # DataLoaders' pin_memory=True already guarantees.
    samples, targets = samples.float().to(device, non_blocking=True), targets.float().to(device, non_blocking=True)

    with torch.autocast(device_type="cuda", dtype=(amp_dtype or torch.bfloat16), enabled=amp_dtype is not None):
      outputs = model(samples)
      #outputs = torch.sigmoid(outputs)
      loss = criterion(outputs, targets)

    if accum_steps == 1:
      optimizer.zero_grad()
      if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()
    else:
      _n_group = min(_group_size, _n_total - (i // accum_steps) * _group_size)
      _weighted = loss * (samples.size(0) / _n_group)
      if i % accum_steps == 0:
        optimizer.zero_grad()
      if scaler is not None:
        scaler.scale(_weighted).backward()
      else:
        _weighted.backward()
      if (i + 1) % accum_steps == 0 or (i + 1) == len(data_loader_train):
        if scaler is not None:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)

  return losses.avg


def evaluate(data_loader_val, device, model, criterion, amp_dtype=torch.bfloat16):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      # non_blocking=True overlaps the host->device copy with other GPU work instead of
      # blocking on it -- only actually async when the source tensor is pinned, which the
      # DataLoaders' pin_memory=True already guarantees.
      samples, targets = samples.float().to(device, non_blocking=True), targets.float().to(device, non_blocking=True)

      with torch.autocast(device_type="cuda", dtype=(amp_dtype or torch.bfloat16), enabled=amp_dtype is not None):
        outputs = model(samples)
        #outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, targets)

      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = build_classification_model(args)

  # weights_only=False: same PyTorch >=2.6 default-change issue as models.py's checkpoint load --
  # this is our own saved checkpoint (state_dict/optimizer/scheduler + epoch/lossMIN/aucMAX), safe to trust.
  modelCheckpoint = torch.load(checkpoint, weights_only=False)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      _amp_dtype = resolve_amp_dtype(args)
      with torch.autocast(device_type="cuda", dtype=(_amp_dtype or torch.bfloat16), enabled=_amp_dtype is not None):
        out = model(varInput)
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def test_model(model, data_loader_test, args):
  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      _amp_dtype = resolve_amp_dtype(args)
      with torch.autocast(device_type="cuda", dtype=(_amp_dtype or torch.bfloat16), enabled=_amp_dtype is not None):
        out = model(varInput)
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test
