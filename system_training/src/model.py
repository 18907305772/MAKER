import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.wrap_encoder()
        self.config = config
        self.model_args = model_args
        if self.model_args.generator_distill_retriever is True:
            self.overwrite_forward_crossattention()
            self.reset_score_storage()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L. only wrapper t5 encoder
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)  # (bs, context_num * max_src_len)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)  # (bs, context_num * max_src_len)
        return self.t5fcg_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def t5fcg_forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, decoder_input_ids=None,
                      decoder_attention_mask=None, decoder_past_key_value_states=None, use_cache=None, labels=None,
                      inputs_embeds=None, decoder_inputs_embeds=None, head_mask=None, output_attentions=None,
                      output_hidden_states=None, resp_delex_mask=None, step=None, **kwargs):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]  # (bs, db_num * (text_max_len + db_max_len), hidden_size)
        bsz = hidden_states.size(0)
        db_num = self.encoder.n_passages
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        if self.model_args.generator_distill_retriever is True and labels is not None:
            self.reset_score_storage()
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if self.model_args.generator_distill_retriever is True and step > self.model_args.generator_distill_retriever_start_step:
                decoder_cross_attention_scores = self.get_crossattention_scores()  # (bs, answer_length, db_num, context_maxlength+db_maxlength)
                if resp_delex_mask is not None:
                    resp_delex_mask = resp_delex_mask[:, :, None, None]
                    decoder_cross_attention_scores = decoder_cross_attention_scores.masked_fill(~resp_delex_mask, 0.).sum(dim=1) / resp_delex_mask.sum(dim=1)   # (bs, db_num, context_maxlength+db_maxlength)
                else:
                    decoder_cross_attention_scores = decoder_cross_attention_scores[:, 0, :, :]   # (bs, db_num, context_maxlength+db_maxlength)
                if self.model_args.generator_distill_retriever_pooling == "cls":
                    decoder_cross_attention_scores = decoder_cross_attention_scores[:, :, 0]  # (bs, db_num)
                elif self.model_args.generator_distill_retriever_pooling == "avg":
                    avg_attention_mask = attention_mask.view(bsz, db_num, -1)
                    decoder_cross_attention_scores = decoder_cross_attention_scores.masked_fill(~avg_attention_mask, 0.).sum(dim=2) / avg_attention_mask.sum(dim=2)  # (bs, db_num)
                elif self.model_args.generator_distill_retriever_pooling == "avg_wo_context":
                    avg_attention_mask = attention_mask.view(bsz, db_num, -1)
                    decoder_cross_attention_scores = decoder_cross_attention_scores.masked_fill(
                        ~avg_attention_mask, 0.)[:, :, self.model_args.generator_text_maxlength:].sum(dim=2) / avg_attention_mask[:, :, self.model_args.generator_text_maxlength:].sum(dim=2)  # (bs, db_num)
                else:
                    raise ValueError
                decoder_outputs = decoder_outputs + (decoder_cross_attention_scores,)
            else:
                decoder_outputs = decoder_outputs + (None,)
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)  # greedy search
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),  # (bs, context_num * src_max_len)
            attention_mask=attention_mask.view(attention_mask.size(0), -1),  # (bs, context_num * src_max_len)
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage.unsqueeze(
                1))  # (bs, 1, head_num, answer_max_len, (text_max_len + db_max_len) * db_num)
        scores = torch.cat(scores,
                           dim=1)  # (bs, layer_num, head_num, answer_max_len, (text_max_len + db_max_len) * db_num)
        bsz, n_layers, n_heads, answer_max_len, _ = scores.size()
        scores = scores.view(bsz, n_layers, n_heads, answer_max_len, self.encoder.n_passages,
                             -1)  # (bs, layer_num, head_num, answer_max_len, db_num, (text_max_len + db_max_len))
        return scores.mean(dim=[1, 2])  # (bs, answer_max_len, db_num, (text_max_len + db_max_len))

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs, ):  # rewrite T5 encoder forward
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)  # (bs * context_num, src_max_len)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)  # (bs * context_num, src_max_len)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)  # ((bs * context_num, src_max_len, hidden_size))
        outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
):
    """
    Self-attention (if kv is None) or attention over source sentence (provided by kv).
    """
    # Input is (bs, qlen, dim)
    # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
    # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
    bs, qlen, dim = input.size()

    if past_key_value_state is not None:
        assert self.is_decoder is True, "Encoder cannot cache past key value states"
        assert (
                len(past_key_value_state) == 2
        ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
            len(past_key_value_state)
        )
        real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
    else:
        real_qlen = qlen

    if kv is None:
        klen = real_qlen
    else:
        klen = kv.size(1)

    def shape(x):
        """  projection """
        return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

    def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

    q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

    if kv is None:
        k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
    elif past_key_value_state is None:
        k = v = kv
        k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

    if past_key_value_state is not None:
        if kv is None:
            k_, v_ = past_key_value_state
            k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
            v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
        else:
            k, v = past_key_value_state

    if self.is_decoder and use_cache is True:
        present_key_value_state = ((k, v),)
    else:
        present_key_value_state = (None,)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

    if position_bias is None:
        if not self.has_relative_attention_bias:
            raise ValueError("No position_bias provided and no weights to compute position_bias")
        position_bias = self.compute_bias(real_qlen, klen)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value_state is not None:
            position_bias = position_bias[:, :, -1:, :]

        if mask is not None:
            position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

    scores += position_bias

    if self.score_storage is None:  # use to save attention score
        self.score_storage = scores

    weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
    weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

    # Mask heads if we want to
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
    context = unshape(context)  # (bs, qlen, dim)

    context = self.o(context)

    outputs = (context,) + present_key_value_state

    if output_attentions:
        outputs = outputs + (weights,)
    if self.has_relative_attention_bias:
        outputs = outputs + (position_bias,)
    return outputs
