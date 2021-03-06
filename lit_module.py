from matplotlib import transforms
import pytorch_lightning as pl
from tenacity import retry
from torch import nn
import torch

class TransformerTranslation(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def _share_step(self, batch, mode='train', **kwargs):
        src, tgt, tgt_pred = batch

        output = self.model(src, tgt)
        # y_hat.view(-1, self.model.config.vocab_size), labels.view(-1)
        loss = self.loss_fn(output.view(-1, self.model.trg_vocab_size), tgt_pred.view(-1))

        self.log(f"{mode}_loss", loss)

        return loss 
    
    def training_step(self, batch, batch_nb):
        # batch
        # labels = batch['labels']
        
        # # fwd
        # y_hat = self.model(batch)
        
        # # loss
        # loss_fct = torch.nn.CrossEntropyLoss()
        # masked_lm_loss = loss_fct(y_hat.view(-1, self.model.config.vocab_size), labels.view(-1))
        # self.log_dict({'train_loss':masked_lm_loss}, prog_bar=True)

        return self._share_step(batch, mode="train")

    # def validation_step(self, batch, batch_nb):
    #     # batch
    #     labels = batch['labels']

    #     # fwd
    #     y_hat = self.model(batch)
        
    #     # loss
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     masked_lm_loss = loss_fct(y_hat.view(-1, self.model.config.vocab_size), labels.view(-1))

    #     metrics = self.compute_metrics([y_hat, labels])
    #     # Calling self.log will surface up scalars for you in TensorBoard
    #     self.log_dict({'val_loss':masked_lm_loss, 'val_bleu':metrics['bleu'],'val_genlen':metrics['gen_len'] }, prog_bar=True)
    #     return masked_lm_loss



        
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # labels = batch['labels']
        # decoder_input_ids = batch['decoder_input_ids'] if 'decoder_input_ids' in batch.keys() else None
        # encoder_outputs = batch['encoder_outputs'] if 'encoder_outputs' in batch.keys() else None

        # if labels is not None:
        #     if decoder_input_ids is None:
        #         decoder_input_ids = self.shift_tokens_right(
        #             labels, self.config.pad_token_id, self.config.decoder_start_token_id
        #         )

        # outputs = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     encoder_outputs = encoder_outputs
        # )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        return lm_logits


if __name__ == "__main__":
    from data_module import TranslationDataModule
    from transformer import Transformer 
    
    tranlate_module = TranslationDataModule()
    tranlate_module.setup()

    src_vocab_size = tranlate_module.src_vocab_size
    tgt_vocab_size = tranlate_module.tgt_vocab_size

    print(src_vocab_size, tgt_vocab_size)
    src_pad_idx = tranlate_module.src_pad_idx
    tgt_pad_idx = tranlate_module.tgt_pad_idx
    max_length = tranlate_module.max_length
    print(src_pad_idx, tgt_pad_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, max_length=max_length).to(
        device
    )
    
    x, y, z = next(iter(tranlate_module.train_dataloader()))

    # emb = torch.nn.Embedding(src_vocab_size, 512)
    # print(emb(x).shape)

    out = model(x, y)
    print(out.shape)

    lit_module = TransformerTranslation(model)
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(gpus=gpus, max_epochs=1)
    trainer.fit(lit_module, tranlate_module)

    # print(x.shape, y.shape)

    # out = transformer_model(x[:1], y[:1])


    # lit = TransformerTranslation()
    