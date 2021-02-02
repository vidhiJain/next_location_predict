import torch
import math

def batchify(data, bsz):
    # data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_accuracy(data, model, args, device):
    model.eval()
    # avg_acc = 0
    record_acc = [] 
    grand_total = 0
    with torch.no_grad():
        for i in range(0, data.size(0)-1, args.bptt):
            inputs, targets = get_batch(data, i, args.bptt)
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model(inputs)
            pred_probs = out.view(-1, args.ntokens)
            _, pred = pred_probs.max(1)
            correct = (pred==targets).sum().item()
            total = targets.size(0)            
            acc = correct*1./total
            # print(f'Accuracy for batch {i} = {acc}')
            # avg_acc += 1./(i+1) * (acc - avg_acc)
            record_acc.append(correct)
            grand_total += total
        # print(f'Avg Accuracy = {avg_acc}')
    # assert abs(avg_acc - sum(record_acc)/len(record_acc)) < 1e-3
    # return sum(record_acc)/grand_total
    # return sum(record_acc)/len(record_acc)
    # return avg_acc
    return sum(record_acc)/grand_total