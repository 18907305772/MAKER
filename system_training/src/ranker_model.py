import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class RankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, output_dim=12):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, output_dim)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertForRank(transformers.BertPreTrainedModel):
    def __init__(self, config, model_args):
        super().__init__(config)

        self.model_args = model_args
        self.bert = BertModel(config)
        if self.model_args.dataset_name == "mwoz_gptke":
            self.attr_num = 12
        elif self.model_args.dataset_name == "camrest":
            self.attr_num = 7
        elif self.model_args.dataset_name == "smd":
            self.attr_num = 21
        else:
            raise NotImplementedError
        self.ranker_head = RankerHead(config, output_dim=self.attr_num)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            times_matrix=None,
            step=None,
            retriever_top_k_dbs_scores=None,
            attr_dk_mask=None,
            generator_sep_id=None,
            generator_db_id=None,
            generator_input_ids=None,
            generator_attention_mask=None,
            **kwargs
    ):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            bsz = input_ids.size(0)
            input_ids = input_ids.view(input_ids.size(0) * self.model_args.top_k_dbs, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0) * self.model_args.top_k_dbs, -1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(token_type_ids.size(0) * self.model_args.top_k_dbs, -1)
        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = encoder_outputs[0].view(bsz, -1, self.config.hidden_size)  # (bs, db_num * (text_max_len + db_max_len), hidden_size)
        # get [top_k, attr_num] ranker scores and use retriever_top_k_dbs_scores to weighted pooling
        ranker_scores = hidden_states.view(bsz, self.model_args.top_k_dbs, -1, self.config.hidden_size)  # (bs, db_num, (text_max_len + db_max_len), hidden_size)
        if self.model_args.rank_attribute_pooling == "cls":
            ranker_scores = ranker_scores[:, :, 0, :]  # (bs, db_num, hidden_size)
        elif self.model_args.rank_attribute_pooling == "cls_wo_context":
            ranker_scores = ranker_scores[:, :, self.model_args.ranker_text_maxlength, :]  # (bs, db_num, hidden_size)
        elif self.model_args.rank_attribute_pooling == "avg":
            ranker_attention_mask = attention_mask.view(bsz, self.model_args.top_k_dbs, -1).unsqueeze(3)  # (bs, db_num, (text_max_len + db_max_len), 1)
            ranker_scores = ranker_scores.masked_fill(~ranker_attention_mask, 0.).sum(dim=2) / ranker_attention_mask.sum(dim=2)  # (bs, db_num, hidden_size)
        elif self.model_args.rank_attribute_pooling == "avg_wo_context":
            ranker_attention_mask = attention_mask.view(bsz, self.model_args.top_k_dbs, -1)[:, :, self.model_args.ranker_text_maxlength:].unsqueeze(3)  # (bs, db_num, db_max_len, 1)
            ranker_scores = ranker_scores[:, :, self.model_args.ranker_text_maxlength:, :].masked_fill(~ranker_attention_mask, 0.).sum(dim=2) / ranker_attention_mask.sum(dim=2)  # (bs, db_num, hidden_size)
        else:
            raise ValueError
        ranker_scores = self.ranker_head(ranker_scores)  # (bs, db_num, num_attribute)
        if retriever_top_k_dbs_scores is not None and self.model_args.rank_no_retriever_weighted is False:
            top_k_weight = F.softmax(retriever_top_k_dbs_scores, dim=-1).unsqueeze(2)  # (bs, db_num, 1)
            ranker_scores = (ranker_scores * top_k_weight).sum(dim=1)  # (bs, num_attribute)
        else:
            ranker_scores = ranker_scores.mean(dim=1)  # (bs, num_attribute)
        if self.model_args.ranker_attribute_ways == "threshold":
            ranker_scores = torch.sigmoid(ranker_scores)  # (bs, num_attribute)
        # use times matrix to calc loss
        outputs = ()
        if times_matrix is not None and self.model_args.ranker_times_matrix is True and step > self.model_args.ranker_times_matrix_start_step:
            if self.model_args.ranker_attribute_ways == "top_r":
                if self.model_args.ranker_times_matrix_loss_type == "kl":
                    times_target = times_matrix.float().mean(dim=1)  # (bs, num_attribute)
                    ranker_times_loss = self.kldivloss(ranker_scores, times_target)
                elif self.model_args.ranker_times_matrix_loss_type == "ce":
                    times_target = F.softmax(times_matrix.float().mean(dim=1), dim=-1)  # (bs, num_attribute)
                    ranker_times_loss = self.SoftCrossEntropy(ranker_scores, times_target)
                else:
                    raise ValueError
            elif self.model_args.ranker_attribute_ways == "threshold":
                if self.model_args.ranker_times_matrix_loss_type == "bce":
                    times_target = (times_matrix.float().mean(dim=1) > 0).float()  # (bs, num_attribute)
                    ranker_times_loss = self.BinaryCrossEntropy(ranker_scores, times_target)
                else:
                    raise ValueError
            else:
                raise ValueError
            outputs += (ranker_times_loss,)
        else:
            outputs += (None,)
        # ranker attribute score, get new attention mask
        if step > self.model_args.rank_attribute_start_step:
            if self.model_args.ranker_attribute_ways == "top_r":
                # if attr_dk_mask is not None:
                #     ranker_scores = ranker_scores.masked_fill(~attr_dk_mask.sum(dim=1).bool(), -1e9)  # (bs, num_attribute)
                ranker_top_r_attr_index = ranker_scores.sort(-1, True)[1][:, :self.model_args.top_r_attr]  # (bs, top_r)
                attr_attention_mask = self.get_topr_attr_mask(generator_sep_id, generator_db_id, generator_input_ids, generator_attention_mask, ranker_top_r_attr_index)
            elif self.model_args.ranker_attribute_ways == "threshold":
                # if attr_dk_mask is not None:
                #     ranker_scores = ranker_scores.masked_fill(~attr_dk_mask.sum(dim=1).bool(), 0)  # (bs, num_attribute)
                attr_attention_mask, ranker_threshold_attr_mask = self.get_threshold_attr_mask(generator_sep_id, generator_db_id, generator_input_ids, generator_attention_mask, ranker_scores)
            else:
                raise ValueError
            outputs += (attr_attention_mask,)
            if self.model_args.write_generate_result is True and self.model_args.ranker_attribute_ways == "threshold":
                outputs += (ranker_threshold_attr_mask,)
        else:
            outputs += (generator_attention_mask.view(bsz, self.model_args.top_k_dbs, -1),)
        return outputs

    def get_topr_attr_mask(self, sep_id, db_id, input_ids, attention_mask, ranker_top_r_attr_index):
        bsz = input_ids.size(0)
        num_db = self.model_args.top_k_dbs
        input_ids = input_ids.view(bsz, num_db, -1).view(bsz * num_db, -1)
        attention_mask = attention_mask.view(bsz, num_db, -1).view(bsz * num_db, -1)
        ranker_top_r_attr_index = ranker_top_r_attr_index.unsqueeze(1).repeat(1, num_db, 1).view(bsz * num_db, -1)
        sep_index = input_ids.eq(sep_id).nonzero()
        db_index = input_ids.eq(db_id).nonzero()
        attr_num = self.attr_num
        attr_true_mask = torch.ones_like(attention_mask).bool()
        assert len(sep_index) % attr_num == 0
        for i, idx in enumerate(sep_index):
            if i % attr_num == 0:
                start_idx = db_index[sep_index[i][0]][1]
            else:
                start_idx = sep_index[i - 1][1] + 1
            end_idx = sep_index[i][1] + 1
            if i % attr_num in ranker_top_r_attr_index[sep_index[i][0]]:
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = True
            else:
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = False
        return (attention_mask * attr_true_mask).view(bsz, num_db, -1)

    def get_threshold_attr_mask(self, sep_id, db_id, input_ids, attention_mask, ranker_scores):
        bsz = input_ids.size(0)
        num_db = self.model_args.top_k_dbs
        input_ids = input_ids.view(bsz, num_db, -1).view(bsz * num_db, -1)
        attention_mask = attention_mask.view(bsz, num_db, -1).view(bsz * num_db, -1)
        ranker_threshold_attr_mask = ranker_scores.unsqueeze(1).repeat(1, num_db, 1).view(bsz * num_db, -1) > self.model_args.threshold_attr
        sep_index = input_ids.eq(sep_id).nonzero()
        db_index = input_ids.eq(db_id).nonzero()
        attr_num = self.attr_num
        attr_true_mask = torch.ones_like(attention_mask).bool()
        assert len(sep_index) % attr_num == 0
        for i, idx in enumerate(sep_index):
            if i % attr_num == 0:
                start_idx = db_index[sep_index[i][0]][1]
            else:
                start_idx = sep_index[i - 1][1] + 1
            end_idx = sep_index[i][1] + 1
            if ranker_threshold_attr_mask[sep_index[i][0]][i % attr_num] == True:
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = True
            else:
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = False
        if self.model_args.write_generate_result is False:
            return (attention_mask * attr_true_mask).view(bsz, num_db, -1), None
        else:
            return (attention_mask * attr_true_mask).view(bsz, num_db, -1), ranker_threshold_attr_mask.view(bsz, num_db, -1)[:, 0, :]

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        loss_fct = torch.nn.KLDivLoss()
        return loss_fct(score, gold_score)

    def SoftCrossEntropy(self, inputs, target, reduction='mean'):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'mean':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        else:
            loss = torch.sum(torch.mul(log_likelihood, target))
        return loss

    def BinaryCrossEntropy(self, inputs, target):
        loss_fct = torch.nn.BCELoss()
        return loss_fct(inputs, target)
