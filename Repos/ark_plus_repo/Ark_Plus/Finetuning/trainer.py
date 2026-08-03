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


def train_one_epoch(data_loader_train, device, model, criterion, optimizer, epoch, amp_dtype=torch.bfloat16, scaler=None):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device), targets.float().to(device)

    with torch.autocast(device_type="cuda", dtype=(amp_dtype or torch.bfloat16), enabled=amp_dtype is not None):
      outputs = model(samples)
      #outputs = torch.sigmoid(outputs)
      loss = criterion(outputs, targets)

    optimizer.zero_grad()
    if scaler is not None:
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
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
      samples, targets = samples.float().to(device), targets.float().to(device)

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

  modelCheckpoint = torch.load(checkpoint)
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
