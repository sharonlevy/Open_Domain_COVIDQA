import torch
import torch.nn.functional as F




def loss_dense(model, batch):
    """
    used by vanilla bert-base dense retrieval
    """
    outputs = model(batch)
    q = outputs['q']
    c = outputs['c']
    neg_c = outputs['neg_c']
    product_in_batch = torch.mm(q, c.t())
    product_neg = (q * neg_c).sum(-1).unsqueeze(1)
    product = torch.cat([product_in_batch, product_neg], dim=-1)

    target = torch.arange(product.size(0)).to(product.device)
    loss = F.cross_entropy(product, target)

    return {
        "retr_loss": loss,
    }



