def cudaify(batch, labels):
    if type(labels) == dict:
        if type(batch) == tuple:
            return (batch[0].cuda(), batch[1]), {k:v.cuda() for k,v in labels.items()}
        return batch.cuda(), {k:v.cuda() for k,v in labels.items()}
    else:
        if type(batch) == tuple:
            return (batch[0].cuda(), batch[1]), {k:v.cuda() for k,v in labels.items()}
        return batch.cuda(), labels.cuda()