
import os
import sys
import shutil
import time
import random
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
import csv

from models import build_classification_model, save_checkpoint
from utils import *
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from trainer import train_one_epoch, evaluate, test_classification, test_model, resolve_amp_dtype

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

sys.setrecursionlimit(40000)


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  # persistent_workers avoids respawning the worker pool every epoch (real overhead across a
  # 50-epoch run); only valid when num_workers>0. A rock-steady per-batch time with zero
  # jitter over hundreds of batches is the signature of a CPU-bound (not GPU-bound) step --
  # exactly what too few dataloader workers looks like, not a slow GPU.
  _persistent = args.workers > 0
  data_loader_test = DataLoader(dataset=dataset_test, batch_size=int(args.batch_size/2), shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=_persistent)
  # training phase
  if args.mode == "train":
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, persistent_workers=_persistent)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, persistent_workers=_persistent)
                           
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....")
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      # Explicit per-trial seeding (protocol: 1 seed for screening, 3 for headline results).
      # args.seed_list gives the actual seed values to report; falls back to the bare trial
      # index if unset, matching the original (implicit, non-reproducible) behavior.
      _seed = getattr(args, "seed_list", None)
      _seed = _seed[i] if _seed is not None else i
      torch.manual_seed(_seed)
      np.random.seed(_seed)
      random.seed(_seed)
      print("seed:", _seed)
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      best_val_auc = -1.0  # checkpoint selection criterion when args.test_every_epoch is True (see below) --
                            # higher is better, so -1.0 (below any real AUC) is the correct "nothing saved yet" sentinel
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      model = build_classification_model(args)
      print(model)

      # Class-weighted BCE (targets the imbalance behind low-AP classes like Consolidation)
      # and an optional co-occurrence-aware regularization on the classifier head's weights
      # (a simplified, batch-size-independent version of hierarchical/dependency-aware losses
      # in recent chest X-ray literature -- see utils.CoOccurrenceBCELoss). Both are opt-in via
      # args.pos_weight / args.cooccurrence_matrix; when neither is set this is plain BCE,
      # identical to the original behavior.
      _pos_weight = getattr(args, "pos_weight", None)
      _cooc_matrix = getattr(args, "cooccurrence_matrix", None)
      if _pos_weight is not None or _cooc_matrix is not None:
        criterion = CoOccurrenceBCELoss(
          model,
          pos_weight=_pos_weight.to(device) if _pos_weight is not None else None,
          cooccurrence_matrix=_cooc_matrix,
          cooc_weight=getattr(args, "cooc_weight", 0.05),
        )
      else:
        criterion = torch.nn.BCEWithLogitsLoss()
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        criterion = torch.nn.CrossEntropyLoss()
        print("use CrossEntropyLoss...")

      if args.freeze_encoder:
        print("===> freezing encoder...")
        ##freeze all layers but the head 
        for name, param in model.named_parameters():
          if name not in ['head.weight', 'head.bias']:
              param.requires_grad = False

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      #optimizer = torch.optim.Adam(parameters, lr=args.lr)
      # optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0, momentum=args.momentum, nesterov=False)
      # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
      #                                  threshold=0.0001, min_lr=0, verbose=True)
      optimizer = create_optimizer(args, model)
      loss_scaler = NativeScaler()

      lr_scheduler, _ = create_scheduler(args, optimizer)

      if args.resume:
        resume = os.path.join(model_path, experiment + '.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          best_val_loss = init_loss
          best_val_auc = checkpoint.get('aucMAX', -1.0)  # .get() for backward-compat with checkpoints saved before AUC-based selection
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f}, val_auc={:.5f})"
                .format(resume, start_epoch, init_loss, best_val_auc))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))

      mean_result_list, result_list = [], []
      accuracy = []
      epoch_table = []
      epoch_table_path = os.path.join(model_path, experiment + "_epoch_table.csv")
      # fp16 needs gradient scaling to avoid underflow; bf16 doesn't (same exponent range as
      # fp32). One scaler per trial, reused across epochs so its scale factor can adapt.
      _amp_dtype = resolve_amp_dtype(args)
      _scaler = torch.cuda.amp.GradScaler() if _amp_dtype == torch.float16 else None
      for epoch in range(start_epoch, args.epochs):
        if args.skip_training:
          break
        train_loss = train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch, amp_dtype=_amp_dtype, scaler=_scaler)

        val_loss = evaluate(data_loader_val, device,model, criterion, amp_dtype=_amp_dtype)

        lr_scheduler.step(val_loss)

        if args.test_every_epoch:
          # Per-epoch metrics (and checkpoint selection, below) are computed on the VALIDATION
          # set only. The held-out test set is touched exactly once, after all training is done,
          # on the single best-saved checkpoint (see the "start testing" block later in this
          # function). Evaluating on test every epoch and using those numbers to pick "best
          # epoch" -- which is what this block used to do -- leaks the held-out set into model
          # selection; the whole point of holding it out is that it never influences which
          # checkpoint you keep.
          y_val, p_val = test_model(model, data_loader_val, args)
          y_val = y_val.cpu().numpy()
          p_val = p_val.cpu().numpy()

          if args.data_set in ["RSNAPneumonia", "COVIDx"]:
            acc = accuracy_score(np.argmax(y_val,axis=1),np.argmax(p_val,axis=1))
            print(">>{}: ACCURACY = {}".format(experiment,acc))
            with open(output_file, 'a') as writer:
              writer.write(
                "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
            accuracy.append(acc)

          if test_diseases is not None:
            y_val = copy.deepcopy(y_val[:,test_diseases])
            p_val = copy.deepcopy(p_val[:,test_diseases])

          mAUC, auc_scores = meanAUC(y_val, p_val)
          mMCC, mcc_scores = meanMCC(y_val, p_val)
          mAP, ap_scores = meanAP(y_val, p_val)
          mF1, f1_scores = meanF1(y_val, p_val)
          mRecall, recall_scores = meanRecall(y_val, p_val)
          mAccuracy, accuracy_scores = meanAccuracy(y_val, p_val)
          mECE, ece_scores = meanECE(y_val, p_val)

          print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
          print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
          print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
          print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
          print(">> Mean Recall = {:.4f} \nRecall = {}".format(mRecall, np.array2string(np.array(recall_scores), precision=4, separator=',')))
          print(">> Mean Accuracy = {:.4f} \nAccuracy = {}".format(mAccuracy, np.array2string(np.array(accuracy_scores), precision=4, separator=',')))
          print(">> Mean ECE = {:.4f} \nECE = {}".format(mECE, np.array2string(np.array(ece_scores), precision=4, separator=',')))
          mean_result_list.append(mAUC)
          result_list.append([mAUC,mMCC,mAP,mF1,mRecall,mAccuracy,mECE])

          # Clean per-epoch table (Epoch, train_loss, val_loss, F1, AUC, recall, accuracy) for
          # easy display as a DataFrame, separate from the verbose text log above. Rewritten in
          # full after every epoch (not just at the end) so it survives an interrupted run.
          epoch_table.append({
            "Epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "F1": round(mF1, 4),
            "AUC": round(mAUC, 4),
            "recall": round(mRecall, 4),
            "accuracy": round(mAccuracy, 4),
            "ece": round(mECE, 4),
          })
          with open(epoch_table_path, "w", newline="") as f:
            _w = csv.DictWriter(f, fieldnames=["Epoch", "train_loss", "val_loss", "F1", "AUC", "recall", "accuracy", "ece"])
            _w.writeheader()
            _w.writerows(epoch_table)
        
        # Checkpoint selection: by validation macro-AUC when per-epoch val metrics are being
        # computed (the normal case -- args.test_every_epoch=True), falling back to val_loss
        # otherwise. AUC-based selection is what the protocol calls for: training runs the full
        # budgeted epoch count regardless, but the checkpoint that gets reported is whichever
        # epoch scored best on validation AUC, not whichever epoch happened to finish last.
        _select_by_auc = args.test_every_epoch
        _improved = (mAUC > best_val_auc) if _select_by_auc else (val_loss < best_val_loss)

        if _improved:
          if _select_by_auc:
            print("Epoch {:04d}: val AUC improved from {:.5f} to {:.5f} (val_loss={:.5f}), saving model to {}".format(
              epoch, best_val_auc, mAUC, val_loss, save_model_path))
            # plain float, not numpy.float64 -- meanAUC() returns np.mean(...), and a numpy
            # scalar stored in the checkpoint dict breaks torch.load's weights_only=True
            # default (PyTorch >=2.6) when reloading for the final test-set evaluation.
            best_val_auc = float(mAUC)
          else:
            print(
              "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                                save_model_path))
          best_val_loss = val_loss
          patience_counter = 0
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'aucMAX': best_val_auc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
          },  filename="{}".format(save_model_path, epoch))

        else:
          if _select_by_auc:
            print("Epoch {:04d}: val AUC did not improve from {:.5f} (val_loss={:.5f})".format(epoch, best_val_auc, val_loss))
          else:
            print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
          patience_counter += 1

        if patience_counter >= args.patience:
          print("Early Stopping")
          break
          
      #save_checkpoint({
      #      'state_dict': model.state_dict(),
      #    },  filename="{}({} epoch)".format(save_model_path, epoch))
  
      # log experiment
      with open(log_file, 'a') as f:
        f.write("{} ({}epoch)\n".format(experiment,epoch-args.patience))
        f.close()

      if len(mean_result_list) > 0:
        best_rest = max(mean_result_list)
        best_epoch = mean_result_list.index(best_rest)
        print("=====> Max result:  {} at epoch {}".format(best_rest, best_epoch) )
        print("mAUC, mMCC, mAP, mF1, mRecall, mAccuracy, mECE = {}".format(result_list[best_epoch]))
        with open(output_file, 'a') as writer:
          writer.write("=====> Max result:  {} at epoch {}\n".format(best_rest, best_epoch))
          writer.write("mAUC, mMCC, mAP, mF1, mRecall, mAccuracy, mECE = {}\n".format(result_list[best_epoch]))
 
  print ("start testing.....")


  log_file = os.path.join(model_path, "models.log")
  if not os.path.isfile(log_file):
    print("log_file ({}) not exists!".format(log_file))
  else:
    accuracy = []
    mean_auc, mean_mcc, mean_ap, mean_f1, mean_recall, mean_accuracy, mean_ece = [],[],[],[],[],[],[]
    metric_dict = {"auc": [], "mcc": [], "ap": [], "f1": [], "recall": [], "accuracy": [], "ece": []}
    with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
      experiment = reader.readline()
      print(">> Disease = {}".format(diseases))
      writer.write("Disease = {}\n".format(diseases))

      while experiment:
        experiment = experiment.split()[0]
        saved_model = os.path.join(model_path, experiment + ".pth.tar")
        pred_csv = os.path.join(model_path, experiment + ".csv")
        gt_csv = os.path.join(model_path, "gt.csv")
        if os.path.exists(pred_csv) and os.path.exists(gt_csv):
          y_test = read_from_csv(gt_csv)
          p_test = read_from_csv(pred_csv)
        else:
          y_test, p_test = test_classification(saved_model, data_loader_test, device, args)
          y_test = y_test.cpu().numpy()
          p_test = p_test.cpu().numpy()

        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
          print(">>{}: ACCURACY = {}".format(experiment,acc))
          writer.write(
            "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
          accuracy.append(acc)

        
        if test_diseases is not None:
          y_test = copy.deepcopy(y_test[:,test_diseases])
          p_test = copy.deepcopy(p_test[:,test_diseases])
        
        mAUC, auc_scores = meanAUC(y_test, p_test)
        mMCC, mcc_scores = meanMCC(y_test, p_test)
        mAP, ap_scores = meanAP(y_test, p_test)
        mF1, f1_scores = meanF1(y_test, p_test)
        mRecall, recall_scores = meanRecall(y_test, p_test)
        mAccuracy, accuracy_scores = meanAccuracy(y_test, p_test)
        mECE, ece_scores = meanECE(y_test, p_test)

        print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
        print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
        print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
        print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
        print(">> Mean Recall = {:.4f} \nRecall = {}".format(mRecall, np.array2string(np.array(recall_scores), precision=4, separator=',')))
        print(">> Mean Accuracy = {:.4f} \nAccuracy = {}".format(mAccuracy, np.array2string(np.array(accuracy_scores), precision=4, separator=',')))
        print(">> Mean ECE = {:.4f} \nECE = {}".format(mECE, np.array2string(np.array(ece_scores), precision=4, separator=',')))
        writer.write("AUC = {}\n MCC = {}\nAP = {}\nF1 = {}\nRecall = {}\nAccuracy = {}\nECE = {}\n".format(np.array2string(np.array(auc_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(mcc_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(ap_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(f1_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(recall_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(accuracy_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(ece_scores), precision=4, separator=',')))
        writer.write("{}: mAUC = {:.4f}, mMCC = {:.4f}, mAP = {:.4f}, mF1 = {:.4f}, mRecall = {:.4f}, mAccuracy = {:.4f}, mECE = {:.4f}\n".format(experiment, mAUC, mMCC, mAP, mF1, mRecall, mAccuracy, mECE))
        
        
        
        data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
        data = data + p_test.tolist()
        print(len(data[0]),len(data[1]))
        # Write data to CSV file
        with open(pred_csv, mode='w', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerows(data)

        mean_auc.append(mAUC)
        mean_mcc.append(mMCC)
        mean_ap.append(mAP)
        mean_f1.append(mF1)
        mean_recall.append(mRecall)
        mean_accuracy.append(mAccuracy)
        mean_ece.append(mECE)
        metric_dict["auc"].append(auc_scores)
        metric_dict["mcc"].append(mcc_scores)
        metric_dict["ap"].append(ap_scores)
        metric_dict["f1"].append(f1_scores)
        metric_dict["recall"].append(recall_scores)
        metric_dict["accuracy"].append(accuracy_scores)
        metric_dict["ece"].append(ece_scores)
        experiment = reader.readline()
    
      
      data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
      data = data + y_test.tolist()
      print(len(data[0]),len(data[1]))
      # Write data to CSV file
      with open(gt_csv, mode='w', newline='') as file:
          csvwriter = csv.writer(file)
          csvwriter.writerows(data)

      mean_auc,mean_mcc,mean_ap,mean_f1,mean_recall,mean_accuracy,mean_ece = np.array(mean_auc),np.array(mean_mcc),np.array(mean_ap),np.array(mean_f1),np.array(mean_recall),np.array(mean_accuracy),np.array(mean_ece)
      print(">> All trials: mAUC = {}\n mMCC = {}\n mAP = {}\n mF1 = {}\n mRecall = {}\n mAccuracy = {}\n mECE = {}\n ".format(np.array2string(mean_auc, precision=4, separator=','),
                                                                                   np.array2string(mean_mcc, precision=4, separator=','),
                                                                                   np.array2string(mean_ap, precision=4, separator=','),
                                                                                   np.array2string(mean_f1, precision=4, separator=','),
                                                                                   np.array2string(mean_recall, precision=4, separator=','),
                                                                                   np.array2string(mean_accuracy, precision=4, separator=','),
                                                                                   np.array2string(mean_ece, precision=4, separator=',')))
      writer.write("All trials: mAUC  = {}\n mMCC  = {}\n mAP = {}\n mF1 = {}\n mRecall = {}\n mAccuracy = {}\n mECE = {}\n ".format(np.array2string(mean_auc, precision=4, separator='\t'),
                                                                            np.array2string(mean_mcc, precision=4, separator='\t'),
                                                                            np.array2string(mean_ap, precision=4, separator='\t'),
                                                                            np.array2string(mean_f1, precision=4, separator='\t'),
                                                                            np.array2string(mean_recall, precision=4, separator='\t'),
                                                                            np.array2string(mean_accuracy, precision=4, separator='\t'),
                                                                            np.array2string(mean_ece, precision=4, separator='\t')))
      print(">> Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f}) aRecall = {:.4f}({:.4f}) aAccuracy = {:.4f}({:.4f}) aECE = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                            np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                            np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                            np.mean(mean_f1), np.std(mean_f1),
                                                                                                                                            np.mean(mean_recall), np.std(mean_recall),
                                                                                                                                            np.mean(mean_accuracy), np.std(mean_accuracy),
                                                                                                                                            np.mean(mean_ece), np.std(mean_ece)))
      writer.write("Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f}) aRecall = {:.4f}({:.4f}) aAccuracy = {:.4f}({:.4f}) aECE = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                                np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                                np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                                np.mean(mean_f1), np.std(mean_f1),
                                                                                                                                                np.mean(mean_recall), np.std(mean_recall),
                                                                                                                                                np.mean(mean_accuracy), np.std(mean_accuracy),
                                                                                                                                                np.mean(mean_ece), np.std(mean_ece)))
      class_wise_scores = []
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["auc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["mcc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["ap"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["f1"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["recall"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["accuracy"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["ece"]))
      writer.write("AUC/MCC/AP/F1/Recall/Accuracy/ECE(mean/std): \n{}\n".format(class_wise_scores))

      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        accuracy = np.array(accuracy)
        print(">> All trials: ACCURACY  = {}".format(np.array2string(accuracy, precision=4, separator=',')))
        writer.write("All trials: ACCURACY  = {}\n".format(np.array2string(accuracy, precision=4, separator='\t')))
      