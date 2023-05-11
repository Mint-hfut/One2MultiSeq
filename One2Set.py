import os
import numpy as np
from typing import Optional, Tuple,List,Union
import torch.distributed as dist
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartModel, BartConfig,PreTrainedTokenizer,BartTokenizer
from transformers.models.bart.modeling_bart import BartDecoder, BartDecoderLayer, BartAttention,shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions, \
    Seq2SeqModelOutput, BaseModelOutput
from transformers.generation_utils import GreedySearchEncoderDecoderOutput,GreedySearchDecoderOnlyOutput
class One2SetBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig,seed=100):
        super().__init__(config)
        self.model = One2SetBartModel(config)
        config.output_attentions = False
        config.use_cache = False
        #config.dropout=0.2
        self._loss_weight = torch.ones(config.vocab_size).to("cuda:0")
        self.set_random_seeds(seed)
    def set_random_seeds(self,seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)  # 检测是否使用了随机算法，有使用随机算法就会报错，你需要一一解决
        print(f"Random seed set as {seed}")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            trg_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            state: Optional = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if input_ids is None:
            input_ids = self._cache_input_ids
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            trg_mask=trg_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            state=state,
        )
        return outputs
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        state=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "state": state,
        }

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional = None,
        stopping_criteria: Optional = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) :
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # if max_length is not None:
        #     warnings.warn(
        #         "`max_length` is deprecated in this function, use"
        #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
        #         UserWarning,
        #     )
        #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished

        cur_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        max_kp_num = self.model.decoder.max_kp_num
        if cur_len == 1:
            input_ids = input_ids.unsqueeze(1).repeat(1, max_kp_num, 1)
            input_ids = input_ids.view(batch_size*max_kp_num, -1)
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        this_peer_finished = False  # used by synced_gpus only
        encoder_last_hidden_state = model_kwargs['encoder_outputs'].get('last_hidden_state')
        encoder_attention_mask = model_kwargs['attention_mask']
        state = self.model.decoder.init_state_(encoder_last_hidden_state, encoder_attention_mask)
        model_kwargs['state'] = state
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need
            logits = outputs.logits
            logits = logits.view(batch_size, max_kp_num, 1, -1)
            next_token_logits = logits[:, :, -1, :]
            next_token_logits = next_token_logits.view(batch_size*max_kp_num,-1)
            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        input_ids = input_ids.view(batch_size, max_kp_num, -1)
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
from scipy.optimize import linear_sum_assignment
def hungarian_assign(decode_dist, target, ignore_indices, random=False):
    '''

    :param decode_dist: (batch_size, max_kp_num, kp_len, vocab_size)
    :param target: (batch_size, max_kp_num, kp_len)
    :return:
    '''

    batch_size, max_kp_num, kp_len = target.size()
    reorder_rows = torch.arange(batch_size)[..., None]
    if random:
        reorder_cols = np.concatenate([np.random.permutation(max_kp_num).reshape(1, -1) for _ in range(batch_size)], axis=0)
    else:
        score_mask = target.new_zeros(target.size()).bool()
        for i in ignore_indices:
            score_mask |= (target == i)
        score_mask = score_mask.unsqueeze(1)  # (batch_size, 1, max_kp_num, kp_len)

        score = decode_dist.new_zeros(batch_size, max_kp_num, max_kp_num, kp_len)
        for b in range(batch_size):
            for l in range(kp_len):
                score[b, :, :, l] = decode_dist[b, :, l][:, target[b, :, l]]
        score = score.masked_fill(score_mask, 0)
        score = score.sum(-1)  # batch_size, max_kp_num, max_kp_num

        reorder_cols = []
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(score[b].detach().cpu().numpy(), maximize=True)
            reorder_cols.append(col_ind.reshape(1, -1))
            # total_score += sum(score[b][row_ind, col_ind])
        reorder_cols = np.concatenate(reorder_cols, axis=0)
    return tuple([reorder_rows, reorder_cols])
EPS = 1e-8
def masked_cross_entropy(class_dist, target, trg_mask, loss_scales=None, scale_indices=None):
    """
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :return:
    """
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.reshape(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    target_flat = target.reshape(-1, 1)  # [batch*trg_seq_len, 1]
    log_dist_flat = torch.log(class_dist_flat + EPS)
    # label = log_dist_flat.argmax(-1)
    # a= label[:60]
    # #print(a)
    # b = target_flat[:60]
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat)  # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]

    if loss_scales is not None:
        for loss_scale, scale_index in zip(loss_scales, scale_indices):
            scale = losses.new_ones(losses.size()).detach()  # [batch, trg_seq_len]
            scale.masked_fill_(target == scale_index, loss_scale)
            losses = losses * scale

    if trg_mask is not None:
        losses = losses * trg_mask

    loss = losses.sum()
    return loss

class One2SetBartModel(BartModel):
    def __init__(self, config: BartConfig,max_kp_num=20,assign_steps=2):
        super().__init__(config)
        self.config = config
        self.decoder = One2SetBartDecoder(config, self.shared)
        self.max_kp_num = max_kp_num
        self.assign_steps = assign_steps
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', sep_token='<sep>')
        tokenizer.add_tokens(['<number>', '<url>', '<mention>','<eos>', '<NULL>'], special_tokens=True)
        self.tokenizer = tokenizer
        self.seperate_pre_ab = True
        self.fix_kp_num_len = True
        self.loss_scale_pre = 0.2
        self.loss_normalization = 'tokens'
        self.max_kp_len = 12
        self.loss_scale_ab=0.1
        self.output_layer = nn.Linear(config.d_model, len(self.tokenizer), bias=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        trg_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        state: Optional = None,
    ) :

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        #max_kp_num = 20
        #y_t_init = decoder_input_ids.new_ones(batch_size, max_kp_num, 1) * self.config.decoder_start_token_id
        if len(decoder_input_ids.size()) ==2:
            decoder_input_ids = decoder_input_ids.view(batch_size, self.max_kp_num, -1)
        is_inference_ = False
        if self.training:
            self.eval()
            with torch.no_grad():
                is_inference_ = True
                memory_bank = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                state = self.decoder.init_state_(memory_bank[0], attention_mask)
                control_embed = self.decoder.forward_seg(state)
                input_tokens = input_ids.new_ones(batch_size, self.max_kp_num, self.assign_steps + 2)
                decoder_dists = []
                input_tokens[:, :, 0] = self.config.decoder_start_token_id
                for t in range(1, self.assign_steps + 2):
                    decoder_inputs = input_tokens[:, :, :t]
                    # decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(self.config.vocab_size - 1),
                    #                                             self.tokenizer.unk_token_id)
                    decoder_output = self.decoder(input_ids=decoder_inputs,
                                                   attention_mask=decoder_attention_mask,
                                                   encoder_hidden_states=memory_bank[0],
                                                   encoder_attention_mask=attention_mask,
                                                  state=state,
                                                  control_embed=control_embed,
                                                  use_cache=False)
                    decoder_dist = decoder_output.last_hidden_state
                    decoder_dist = F.softmax(self.output_layer(decoder_dist), -1)
                    if t == 1:
                        input_tokens[:, :, t] = self.config.bos_token_id
                    else:
                        input_tokens[:, :, t] = decoder_dist.argmax(-1)
                        decoder_dists.append(decoder_dist.reshape(batch_size, self.max_kp_num, 1, -1))

                decoder_dists = torch.cat(decoder_dists, -2)

                if self.seperate_pre_ab:
                    mid_idx = self.max_kp_num // 2
                    pre_reorder_index = hungarian_assign(decoder_dists[:, :mid_idx],
                                                         labels[:, :mid_idx, 1:self.assign_steps+1],
                                                         ignore_indices=[self.tokenizer.convert_tokens_to_ids('<NULL>'),
                                                         self.config.pad_token_id])
                    labels[:, :mid_idx] = labels[:, :mid_idx][pre_reorder_index]
                    trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                    ab_reorder_index = hungarian_assign(decoder_dists[:, mid_idx:],
                                                        labels[:, mid_idx:, 1:self.assign_steps+1],
                                                        ignore_indices=[self.tokenizer.convert_tokens_to_ids('<NULL>'),
                                                         self.config.pad_token_id])
                    labels[:, mid_idx:] = labels[:, mid_idx:][ab_reorder_index]
                    trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]
                else:
                    reorder_index = hungarian_assign(decoder_dists, labels[:, :, :self.assign_steps],
                                                     [self.tokenizer.convert_tokens_to_ids('<NULL>'),
                                                      self.config.pad_token_id])
                    labels = labels[reorder_index]
                    trg_mask = trg_mask[reorder_index]
            self.train()
            is_inference_ = False
        else:
            is_inference_ = True
        # input_tgt = torch.cat([y_t_init, decoder_input_ids[:, :, :-1]], dim=-1)
        current_kp_len = decoder_input_ids.size(-1)
        if encoder_outputs is not None:
            memory_bank = encoder_outputs
            state = state
        else:
            memory_bank = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            state = self.decoder.init_state_(memory_bank[0], attention_mask)
        control_embed = self.decoder.forward_seg(state)

        #input_tgt = decoder_input_ids.masked_fill(decoder_input_ids.gt(len(self.tokenizer) - 1), self.tokenizer.unk_token_id)
        decoder_output = self.decoder(input_ids=decoder_input_ids,
                                       attention_mask=decoder_attention_mask,
                                       encoder_hidden_states=memory_bank[0],
                                       encoder_attention_mask=attention_mask,
                                        is_inference = is_inference_,
                                      control_embed=control_embed,
                                      state=state)
        decoder_last_hidden_state = decoder_output.last_hidden_state
        decoder_dist = F.softmax(self.output_layer(decoder_last_hidden_state), -1)
        if self.training:
            if self.fix_kp_num_len:
                if self.seperate_pre_ab:
                    mid_idx = self.max_kp_num // 2
                    pre_loss = masked_cross_entropy(
                        decoder_dist.reshape(batch_size, self.max_kp_num, self.max_kp_len, -1)[:, :mid_idx] \
                            .reshape(batch_size, self.max_kp_len * mid_idx, -1),
                        labels[:, :mid_idx].reshape(batch_size, -1),
                        trg_mask[:, :mid_idx].reshape(batch_size, -1),
                        loss_scales=[self.loss_scale_pre],
                        scale_indices=[self.tokenizer.convert_tokens_to_ids('<NULL>')])
                    ab_loss = masked_cross_entropy(
                        decoder_dist.reshape(batch_size, self.max_kp_num, self.max_kp_len, -1)[:, mid_idx:]
                        .reshape(batch_size, self.max_kp_len * mid_idx, -1),
                        labels[:, mid_idx:].reshape(batch_size, -1),
                        trg_mask[:, mid_idx:].reshape(batch_size, -1),
                        loss_scales=[self.loss_scale_ab],
                        scale_indices=[self.tokenizer.convert_tokens_to_ids('<NULL>')])
                    loss = pre_loss + ab_loss
                else:
                    loss = masked_cross_entropy(decoder_dist, labels.reshape(batch_size, -1),
                                                trg_mask.reshape(batch_size, -1),
                                                loss_scales=[self.loss_scale], scale_indices=[self.tokenizer.convert_tokens_to_ids('<NULL>')])
            else:
                loss = masked_cross_entropy(decoder_dist, labels, trg_mask)
            total_trg_tokens = trg_mask.sum().item()
            total_trg_sents = input_ids.size(0)
            if self.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
                normalization = total_trg_tokens
            elif self.loss_normalization == 'batches':  # use batch_size to normalize the loss
                normalization = total_trg_sents
            else:
                raise ValueError('The type of loss normalization is invalid.')
            assert normalization > 0, 'normalization should be a positive number'

            total_loss = loss.div(normalization)
        else:
            total_loss = 0

        #
        # if encoder_outputs is None:
        #     encoder_outputs = self.encoder(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        # # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )
        #
        # # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=encoder_outputs[0],
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=decoder_inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # if not return_dict:
        #     return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=decoder_dist,
            past_key_values=None,
            decoder_hidden_states=decoder_last_hidden_state,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=memory_bank[0],
            encoder_hidden_states=memory_bank.hidden_states,
            encoder_attentions=memory_bank.attentions,
        )
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):
        """
        每个Decoder都有对应的State对象用来承载encoder的输出以及当前时刻之前的decode状态。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        """
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        """
        返回的State中包含的是多少个sample的encoder状态，主要用于Generate的时候确定batch的大小。

        :return:
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        """
        当前Decode到哪个token了，decoder只会从decode_length之后的token开始decode, 为0说明还没开始decode。

        :return:
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)
class TransformerState(State):
    def __init__(self, encoder_output, encoder_mask, num_decoder_layer):
        """
        与TransformerSeq2SeqDecoder对应的State，

        :param torch.FloatTensor encoder_output: bsz x encode_max_len x encoder_output_size, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x encode_max_len 为1的地方需要attend
        :param int num_decoder_layer: decode有多少层
        """
        super().__init__(encoder_output, encoder_mask)
        self.encoder_key = [None] * num_decoder_layer  # 每一个元素 bsz x encoder_max_len x key_dim
        self.encoder_value = [None] * num_decoder_layer  # 每一个元素 bsz x encoder_max_len x value_dim
        self.decoder_prev_key = [None] * num_decoder_layer  # 每一个元素 bsz x decode_length x key_dim
        self.decoder_prev_value = [None] * num_decoder_layer  # 每一个元素 bsz x decode_length x key_dim

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)

    @property
    def decode_length(self):
        if self.decoder_prev_key[0] is not None:
            return self.decoder_prev_key[0].size(1)
        return 0
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    sinusoid的embedding，其中position的表示中，偶数维(0,2,4,...)是sin, 奇数(1,3,5...)是cos
    :param int n_position: 一共多少个position
    :param int d_hid: 多少维度，需要为偶数
    :param padding_idx:
    :return: torch.FloatTensor, shape为n_position x d_hid
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class One2SetBartDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, fix_kp_num_len=True, max_kp_len=12, max_kp_num=20):
        super().__init__(config)
        self.pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(1024, config.d_model, padding_idx=config.pad_token_id),
            freeze=True)
        self.decoder_layers = config.decoder_layers
        self.layers = nn.ModuleList([One2SetBARTDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.fix_kp_num_len = fix_kp_num_len
        if self.fix_kp_num_len:
            self.max_kp_len = max_kp_len
            self.max_kp_num = max_kp_num
            self.control_code = nn.Embedding(max_kp_num, config.d_model)
            self.control_code.weight.data.uniform_(-0.1, 0.1)
            self.self_attn_mask = self._get_self_attn_mask(max_kp_num, self.max_kp_len)

        self.input_fc = nn.Linear(config.d_model, config.d_model)
    def forward_seg(self, state):
        encoder_output = state.encoder_output
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        control_idx = torch.arange(0, self.max_kp_num).long().to(device).reshape(1, -1).repeat(batch_size, 1)
        control_embed = self.control_code(control_idx)

        return control_embed
    @staticmethod
    def _get_self_attn_mask(max_kp_num, max_kp_len):
        mask = torch.ones(max_kp_num * max_kp_len, max_kp_num * max_kp_len)
        mask = torch.tril(mask).bool()
        for i in range(1, max_kp_num + 1):
            mask[i * max_kp_len:(i + 1) * max_kp_len, :i * max_kp_len] = 0
        return mask
    def init_state_(self, encoder_output, encoder_mask):
        """
        初始化一个TransformerState用于forward
        :param torch.FloatTensor encoder_output: bsz x max_len x d_model, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为1的位置需要attend。
        :return: TransformerState
        """
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = TransformerState(encoder_output, encoder_mask, num_decoder_layer=self.decoder_layers)
        return state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        control_embed: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_inference: Optional[bool] = True,
        state: Optional = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            #input_ids = input_ids.view(-1, input_shape[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0




        batch_size, max_kp_num, kp_len = input_shape
        # control_idx = torch.arange(0, self.max_kp_num).long().to(self.device).reshape(1, -1).repeat(batch_size, 1)
        # control_embed = self.control_code(control_idx)
        if is_inference:
            decode_length = kp_len-1
        else:
            decode_length=0
        tokens = input_ids[:,:, decode_length:]
        kp_len = tokens.size(-1)
        max_tgt_len = max_kp_num * kp_len
        position = torch.arange(decode_length, decode_length + kp_len).long().to(self.device).reshape(1, 1, -1)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(tokens) * self.embed_scale

        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, input_shape_for_mask, inputs_embeds, past_key_values_length
        # )

        # expand encoder attention mask
        # if encoder_hidden_states is not None and encoder_attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape_for_mask[-1]*max_kp_num)

        # embed positions
        #positions = self.embed_positions(input_shape, past_key_values_length)
        position_embed = self.pos_embed(position)
        hidden_states = self.input_fc(inputs_embeds) + position_embed + control_embed.reshape(batch_size, max_kp_num, 1, -1)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states.view(batch_size, max_kp_num*kp_len, -1)


        if kp_len > 1:  # training
            attention_mask = self.self_attn_mask
        else:
            attention_mask = self.self_attn_mask.reshape(max_kp_num, self.max_kp_len, max_kp_num, self.max_kp_len) \
                [:, decode_length, :, :decode_length + 1] \
                .reshape(max_kp_num, max_kp_num * (decode_length + 1))
        # if kp_len ==6:  # training
        #     attention_mask = self.self_attn_mask
        # else:
        #     attention_mask = self._get_self_attn_mask(max_kp_num, kp_len)
        if attention_mask.device is not tokens.device:
            attention_mask = attention_mask.to(tokens.device)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () #if output_attentions else None
        all_cross_attentions = () #if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    state=state,
                    layer_idx=idx,
                )
            hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)


            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class One2SetBARTDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.self_attn = One2SetBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn = One2SetBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        state: Optional = None,
        layer_idx: Optional = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            state=state,
            layer_idx=layer_idx,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                state=state,
                layer_idx=layer_idx,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            #present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)


        outputs += (self_attn_weights, cross_attn_weights)

        # if use_cache:
        #     outputs += (present_key_value,)

        return outputs
class One2SetBartAttention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            fix_kp_num_len=True
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)

        self.fix_kp_num_len = fix_kp_num_len
        self.max_kp_num=20
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            state: Optional = None,
            layer_idx: Optional = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        key_states = value_states = None
        prev_k = prev_v = None
        bsz, tgt_len, _ = hidden_states.size()

        if state is not None:
            assert layer_idx is not None
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        if isinstance(state, TransformerState):  # 说明此时在inference阶段
            if not is_cross_attention:  # 此时在decoder self attention
                prev_k = state.decoder_prev_key[layer_idx]
                prev_v = state.decoder_prev_value[layer_idx]
            else:  # 此时在decoder-encoder attention，直接将保存下来的key装载起来即可
                key_states = state.encoder_key[layer_idx]
                value_states = state.encoder_value[layer_idx]

        if key_states is None:
            if is_cross_attention:
                key_states = key_value_states
                value_states = key_value_states

            else:
                key_states = hidden_states
                value_states = hidden_states
            key_states = self.k_proj(key_states)
            value_states = self.v_proj(value_states)



        if prev_k is not None:
            if self.fix_kp_num_len:
                batch_size, max_len, d = prev_k.size()
                prev_k = prev_k.reshape(batch_size, self.max_kp_num, -1, d)
                prev_v = prev_v.reshape(batch_size, self.max_kp_num, -1, d)
                key_states = torch.cat((prev_k, key_states.unsqueeze(-2)), dim=-2).reshape(batch_size, -1, d)
                value_states = torch.cat((prev_v, value_states.unsqueeze(-2)), dim=-2).reshape(batch_size, -1, d)
            else:
                key_states = torch.cat((prev_k, key_states), dim=1)
                value_states = torch.cat((prev_v, value_states), dim=1)
        if isinstance(state, TransformerState):
            if not is_cross_attention:
                state.decoder_prev_key[layer_idx] = key_states
                state.decoder_prev_value[layer_idx] = value_states
            else:
                state.encoder_key[layer_idx] = key_states
                state.encoder_value[layer_idx] = value_states
        # get key, value proj
        # if is_cross_attention and past_key_value is not None:
        #     # reuse k,v, cross_attentions
        #     key_states = past_key_value[0]
        #     value_states = past_key_value[1]
        # elif is_cross_attention:
        #     # cross_attentions
        #     key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        # elif past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # else:
        #     # self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        #
        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_states, value_states)

        #proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        batch_size, q_len, d_model = query_states.size()
        k_len, v_len = key_states.size(1), value_states.size(1)
        query_states = query_states.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, v_len, self.num_heads, self.head_dim)

        src_len = key_states.size(1)
        attn_weights = torch.einsum('bqnh,bknh->bqkn', query_states, key_states)

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )
        if is_cross_attention:
            if attention_mask is not None:
                _key_mask = ~attention_mask[:, None, :, None].bool()  # batch,1,k_len,1
                attn_weights = attn_weights.masked_fill(_key_mask, -float('inf'))
        else:
            if attention_mask is not None:
                _attn_mask = attention_mask[None, :, :, None].eq(0)  # 1,q_len,k_len,n_head
                attn_weights = attn_weights.masked_fill(_attn_mask, -float('inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            pass

        #attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.einsum('bqkn,bknh->bqnh', attn_weights, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)
        #
        # # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # # partitioned aross GPUs when using tensor-parallelism.
        # attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value
