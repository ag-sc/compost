import csv
import os
import statistics
import sys
from typing import Optional

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import LightningModule
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, LlamaForCausalLM, \
    Olmo2ForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss

from dudes import utils
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint

class CompModel(LightningModule):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    sparql_endpoint: Optional[SPARQLEndpoint]

    def __init__(self,
                 model_name: str = "allenai/OLMo-2-1124-7B",
                 learning_rate: float = 1e-5,
                 ld: float = 0.9,
                 sparql_endpoint: Optional[SPARQLEndpoint] = None,
                 is_instruct: bool = False,
                 is_subgraph: bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["sparql_endpoint"])
        self.learning_rate = learning_rate
        self.ld = ld
        self.sparql_endpoint = sparql_endpoint
        self.is_instruct = is_instruct
        self.is_subgraph = is_subgraph
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in ["meta-llama/Llama-3.2-1B"]:
            self.model = LlamaForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.float16, load_in_8bit=True)#quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        elif model_name in ["allenai/OLMo-2-1124-7B", "allenai/OLMo-1B-hf", "allenai/OLMo-1B"]:
            self.model = Olmo2ForCausalLM.from_pretrained(model_name)#, quantization_config=BitsAndBytesConfig(load_in_8bit=True))#, torch_dtype=torch.float16, load_in_8bit=True)#quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id

    def forward(self, inputs, target=None):
        #print(inputs, flush=True)
        if target is None:
            preds = self.model.generate(inputs["input_token_ids"], max_new_tokens=2 * inputs["output_token_ids"].shape[1],
                                        eos_token_id=self.eos_token, pad_token_id=self.pad_token)
            sparql_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            return inputs, sparql_preds
        else:
            return self.model(input_ids=inputs, labels=target)

    def training_step(self, batch, batch_idx, log=True):
        inputs = batch["input_token_ids"]
        target = batch["output_token_ids"]
        pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        

        #res = self.model.generate.__wrapped__(self.model, inputs, output_logits=True, return_dict_in_generate=True, min_new_tokens=target.shape[1], max_new_tokens=target.shape[1])#(inputs, target)
        #batch_logits = torch.concat([t.unsqueeze(dim=2) for t in res.logits], dim=2)
        if self.is_instruct:
            targets_ignore = target.clone()
            targets_ignore[targets_ignore == pad_token] = -100
            #todo set input as irrelevant :inputs.shape[1] = -100 if not working
                        
            output = self.model(input_ids=target)
            loss = ForCausalLMLoss(logits=output.logits, labels=targets_ignore, vocab_size=output.logits.shape[-1], ignore_index=-100, num_items_in_batch=inputs.shape[0])
        else:
            targets_ignore = target.clone()
            targets_ignore[targets_ignore == pad_token] = -100
            targets_ignore = nn.functional.pad(targets_ignore, (inputs.shape[1], 0), value=-100)
            output = self.model(input_ids=torch.concat([inputs, target], dim=-1))
            loss = ForCausalLMLoss(logits=output.logits, labels=targets_ignore, vocab_size=output.logits.shape[-1], ignore_index=-100, num_items_in_batch=inputs.shape[0])

        # output = self.model(input_ids=torch.concat([inputs, target], dim=-1))#, use_cache=True)
        # #past_key_values = output.past_key_values
        # lossfn = nn.CrossEntropyLoss(ignore_index=-100)
        # loss = lossfn(output.logits[:, -(targets_ignore.shape[1]+1):-1, :].movedim(1,2), targets_ignore)#shift by one so logits < n actually predict n




        # pred_list = []
        # print("Custom:", output.logits[:, -1, :].argmax(dim=-1), flush=True)
        # output2 = self.model(input_ids=torch.concat([inputs, output.logits[:, -1:, :].argmax(dim=-1)], dim=-1))
        # print("Custom:", output2.logits[:, -1, :].argmax(dim=-1), flush=True)
        # output3 = self.model(input_ids=output.logits[:, -1:, :].argmax(dim=-1), past_key_values=past_key_values, use_cache=True)
        # print("Custom:", output3.logits[:, -1, :].argmax(dim=-1), flush=True)
        # test = self.model.generate(batch["input_token_ids"], max_new_tokens=2)


        # logits = []
        #
        # for i in range(1, target.shape[1]):
        #     if all((targets_ignore[:, i] == -100).tolist()):
        #         break
        #     output = self.model(input_ids=target[:, :i + 1], past_key_values=past_key_values, use_cache=True)
        #     logits.append(output.logits)
        #     #output = self.model(input_ids=torch.concat([inputs, target[:, :i+1]], dim=-1))
        #     curr_loss = lossfn(output.logits[:, -1, :], targets_ignore[:, i])
        #     #print(targets_ignore[:, i], curr_loss)
        #     loss += curr_loss
        #     #pred_list.append(output.logits[:, -1, :].argmax(dim=-1))



        # output = self.model(input_ids=inputs, use_cache=True)
        # past_key_values = output.past_key_values
        # lossfn = nn.CrossEntropyLoss(ignore_index=-100)
        # loss = None
        # pred_list = []
        #
        # test = self.model.generate(batch["input_token_ids"])
        #
        # for i in range(target.shape[1]):
        #     output = self.model(input_ids=target[:, i:i+1], past_key_values=past_key_values, use_cache=True)
        #     past_key_values = output.past_key_values
        #     if loss is None:
        #         loss = lossfn(output.logits[:, -1, :], targets_ignore[:, i])
        #     else:
        #         loss += lossfn(output.logits[:, -1, :], targets_ignore[:, i])
        #     pred_list.append(output.logits.argmax(dim=2))

        #loss = res.loss
        if log:
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        print(loss.item(), flush=True)
        print("Preds:", self.tokenizer.batch_decode(self.model.generate(batch["input_token_ids"][:1],
                                                                        max_new_tokens=2*batch["output_token_ids"].shape[1],
                                                                        eos_token_id=self.eos_token,
                                                                        pad_token_id=self.pad_token),
                                                    skip_special_tokens=True)
              ) # not useful as not the same as generate as after every step correct tokens are assumed
        #print("Preds:", self.tokenizer.batch_decode(torch.concat(pred_list, dim=1), skip_special_tokens=True)) # not useful as not the same as generate as after every step correct tokens are assumed
        #print("Target:", self.tokenizer.batch_decode(target, skip_special_tokens=True), flush=True)
        return loss

    def _do_eval(self, metric_name, batch):
        if self.sparql_endpoint is None:
            print("Warning: No SPARQL endpoint provided, only returning loss.", metric_name)
            loss = self.training_step(batch, None, log=False)
            self.log(metric_name, loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss.item()
        else:
            #inputs = batch["input_token_ids"].tolist()
            # data = self.test_data
            # if metric_name.startswith("val"):
            #     data = self.val_data
            inputs = self.tokenizer.batch_decode(batch["input_token_ids"], skip_special_tokens=True)
            #target = batch["output_token_ids"] #[data[id][subid-1]["target"] for id, subid in zip(batch["id"].tolist(), batch["subid"].tolist())]
            #print("Target", target)
            #sparql_golds = ["".join([chr(i) for i in elems]).strip() for elems in target]#self.tokenizer.batch_decode(target, skip_special_tokens=True)

            #batch["output_token_ids"][batch["output_token_ids"] == -100] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            sparql_golds = self.tokenizer.batch_decode(batch["output_token_ids"], skip_special_tokens=True)
            if self.is_instruct:
                sparql_golds = [g[len(s):] for s, g in zip(inputs, sparql_golds)]
            print("Golds", sparql_golds, flush=True)

            # allow up to twice as many tokens as in gold to still allow longer equivalent queries, but within some limits
            preds = self.model.generate(batch["input_token_ids"], max_new_tokens=2*batch["output_token_ids"].shape[1], eos_token_id=self.eos_token, pad_token_id=self.pad_token)
            sparql_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            sparql_preds = [g[len(s):] for s, g in zip(inputs, sparql_preds)]
            print("Preds", sparql_preds, flush=True)

            f1s = []

            for i, (gold, pred) in enumerate(zip(sparql_golds, sparql_preds)):
                try:
                    gold_res = self.sparql_endpoint.get_results_query(gold)
                    pred_res = self.sparql_endpoint.get_results_query(pred)

                    tstats = utils.compare_results(gold=gold_res, sys=pred_res)
                    f1s.append(tstats.f1 if tstats.f1 is not None else 0.0)
                    self.log(metric_name, 1.0 - tstats.f1 if tstats.f1 is not None else 1.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                    print(tstats)
                except Exception as e:
                    print(e, i, gold, pred)
                    self.log(metric_name, 1.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return (1.0 - statistics.mean(f1s)) if len(f1s) > 0 else 1.0 # invert to imitate loss, i.e. hyperparameter optimization into same direction, minimizing
    def validation_step(self, batch, batch_idx):
        self._do_eval(metric_name="val_loss", batch=batch)

    def test_step(self, batch, batch_idx):
        self._do_eval(metric_name="test_loss", batch=batch)

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.parameters())
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lambda1 = lambda epoch: self.ld ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
