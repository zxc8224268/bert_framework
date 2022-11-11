from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models import BertMaxPool
from dataloaders import TripletBertDataset
from losses import TripletDistanceMetric, TripletLoss, SoftMarginTripletLoss
import config

# -------------------------------------------------------
#   Train
# -------------------------------------------------------
def train(config, model, dataloader, loss_fn, optimizer, scheduler, tensorboard_writer):
    print('Start training ...')
    training_step = 1

    # train epochs
    for epoch in range(1, config.epochs + 1):
        model.zero_grad() # zero gradient
        model.train() # train mode

        # train epoch
        losses = []
        for batch in dataloader:
            # embed anchor, pos & neg
            emb_anchor = model(
                input_ids=batch['anchor_input_ids'].to(config.device),
                attention_mask=batch['anchor_attention_mask'].to(config.device)
            )
            emb_pos = model(
                input_ids=batch['pos_input_ids'].to(config.device),
                attention_mask=batch['pos_attention_mask'].to(config.device)
            )
            emb_neg = model(
                input_ids=batch['neg_input_ids'].to(config.device),
                attention_mask=batch['neg_attention_mask'].to(config.device)
            )

            # loss
            loss = loss_fn(emb_anchor, emb_pos, emb_neg)
            losses.append(loss.item())
            loss.backward()
            print('Epoch {}, step {}, loss: {:4f}'.format(epoch, training_step, loss.item()))
            tensorboard_writer.add_scalar('Loss/Train', loss.item(), training_step)

            # update optimizer & scheduler
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # evaluate every k steps
            if training_step != 1 and training_step % config.evaluation_steps == 0:
                # evlauation ---------------
                model.eval()
                inputs_a = tokenizer(['iphone'])
                emb_a = model(
                    input_ids=torch.tensor(inputs_a['input_ids']).to(config.device),
                    attention_mask=torch.tensor(inputs_a['attention_mask']).to(config.device)
                ).to('cpu').detach().numpy()
                inputs_b = tokenizer(['Apple 蘋果 iPhone 14 Pro 128G'])
                emb_b = model(
                    input_ids=torch.tensor(inputs_b['input_ids']).to(config.device),
                    attention_mask=torch.tensor(inputs_b['attention_mask']).to(config.device)
                ).to('cpu').detach().numpy()
                import numpy as np
                from numpy.linalg import norm
                sim = np.dot(emb_a, emb_b.T) / (norm(emb_a)*norm(emb_b))
                print(sim)
                tensorboard_writer.add_scalar('Metric/MAP@50', sim, training_step)
                # --------------------------
                model.zero_grad()
                model.train()

            # update training steps
            training_step += 1

# -------------------------------------------------------
#   Main
# -------------------------------------------------------
if __name__ == '__main__':
    # -------------------------------------------------------
    #   Data preprocessing
    # -------------------------------------------------------
    # dataset
    data_train=[
        ['錨點', '正樣本', '負樣本'],
        ['iphone 14 pro', 'Apple 蘋果 iPhone 14 Pro 128G', '3M 不留痕跡掛勾 (3入)'],
        ['iphone', 'Apple 蘋果 iPhone 14 Pro 128G', '米物誌瑞螢幕掛燈 青春版'],
        ['電視', 'JVC 32吋 LED液晶螢幕顯示器 32B(J)', '牛皮平底鞋 黑色'],
    ]
    # data_train=[
    #     ['iphone 14 pro', 'iphone 14 pro', 'iphone 14 pro'],
    #     ['電視', '電視', '電視'],
    # ]

    # tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    dataset = TripletBertDataset(data=data_train, tokenizer=tokenizer, max_length=config.max_length)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, collate_fn=TripletBertDataset.collate_fn)

    # -------------------------------------------------------
    #   Model
    # -------------------------------------------------------
    model = BertMaxPool(
        pretrained_model=config.pretrained_model
    ).to(config.device)

    # -------------------------------------------------------
    #   Optimizer
    # -------------------------------------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=2e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=len(dataloader) * config.epochs
    )

    # -------------------------------------------------------
    #   Train
    # -------------------------------------------------------
    train(
        config=config,
        model=model,
        dataloader=dataloader,
        loss_fn=SoftMarginTripletLoss(
            distance_metric=TripletDistanceMetric.EUCLIDEAN,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        tensorboard_writer=SummaryWriter(
            log_dir='./'
        )
    )






