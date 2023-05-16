import torch
import transformers
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import json
import copy

from src.options import Options
import src.slurm
import src.util
import src.model
import src.simcse_model
import src.ranker_model


def retriever_embedding_db(model, dataloader):
    """embedding all db text"""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for k, (index, text_ids, text_mask, text_token_type, attr_mask) in enumerate(dataloader):
            embeddings = model(input_ids=text_ids.long().cuda(), attention_mask=text_mask.long().cuda(),
                               token_type_ids=text_token_type.long().cuda(), output_hidden_states=True,
                               return_dict=True,
                               sent_emb=True).pooler_output
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (all_db_num, hidden_size)
    model.train()
    return all_embeddings


def get_all_dbs_inputs(db_dataloader):
    all_db_ids = []
    all_db_mask = []
    all_db_token_type = []
    all_db_attr_mask = []
    for k, (index, text_ids, text_mask, text_token_type, attr_mask) in enumerate(db_dataloader):
        all_db_ids.append(text_ids)
        all_db_mask.append(text_mask)
        if text_token_type is not None:
            all_db_token_type.append(text_token_type)
        if attr_mask is not None:
            all_db_attr_mask.append(attr_mask)
    all_db_ids = torch.cat(all_db_ids, 0)
    all_db_mask = torch.cat(all_db_mask, 0)
    all_db_token_type = torch.cat(all_db_token_type, 0) if len(all_db_token_type) > 0 else None
    all_db_attr_mask = torch.cat(all_db_attr_mask, 0) if len(all_db_attr_mask) > 0 else None
    return all_db_ids, all_db_mask, all_db_token_type, all_db_attr_mask


def concat_context_and_dbs_input(context_input, dbs_input):
    context_input = context_input.unsqueeze(1).repeat(1, dbs_input.size(1), 1)
    return torch.cat([context_input, dbs_input], dim=2)


def kldivloss(score, gold_score):
    gold_score = torch.softmax(gold_score, dim=-1)
    score = F.log_softmax(score, dim=-1)
    loss_fct = torch.nn.KLDivLoss()
    return loss_fct(score, gold_score)


def SoftCrossEntropy(inputs, target, reduction='mean'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def train(generator_model, retriever_model, ranker_model, generator_tokenizer, retriever_tokenizer, ranker_tokenizer,
          generator_optimizer, generator_scheduler, retriever_optimizer, retriever_scheduler, ranker_optimizer,
          ranker_scheduler, step, train_dial_dataset, eval_dial_dataset, test_dial_dataset, dial_collator, db_dataset,
          generator_db_collator, retriever_db_collator, ranker_db_collator, opt, best_dev_score, checkpoint_path):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed)  # different seed for different sampling depending on global_rank
    train_dial_sampler = RandomSampler(
        train_dial_dataset)  # if load_data use global rank and world size to distribute load，we dont need DistributedSampler
    train_dial_dataloader = DataLoader(
        train_dial_dataset,
        sampler=train_dial_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=dial_collator
    )
    db_sampler = SequentialSampler(db_dataset)
    generator_db_dataloader = DataLoader(db_dataset,
                                         sampler=db_sampler,
                                         batch_size=222,
                                         drop_last=False,
                                         num_workers=10,
                                         collate_fn=generator_db_collator)
    retriever_db_dataloader = DataLoader(db_dataset,
                                         sampler=db_sampler,
                                         batch_size=222,
                                         drop_last=False,
                                         num_workers=10,
                                         collate_fn=retriever_db_collator)
    ranker_db_dataloader = DataLoader(db_dataset,
                                      sampler=db_sampler,
                                      batch_size=222,
                                      drop_last=False,
                                      num_workers=10,
                                      collate_fn=ranker_db_collator)
    generator_all_dbs_ids, generator_all_dbs_mask, _, _ = get_all_dbs_inputs(generator_db_dataloader)
    retriever_all_dbs_ids, retriever_all_dbs_mask, retriever_all_dbs_token_type, _ = get_all_dbs_inputs(retriever_db_dataloader)
    ranker_all_dbs_ids, ranker_all_dbs_mask, ranker_all_dbs_token_type, _ = get_all_dbs_inputs(ranker_db_dataloader)
    curr_loss = 0.0
    epoch = 0
    generator_model.train()
    retriever_model.train()
    ranker_model.train()
    training_steps = min(opt.total_steps, opt.end_eval_step)
    while step < training_steps:
        epoch += 1
        for i, batch in enumerate(tqdm(train_dial_dataloader)):
            step += 1
            if opt.use_gt_dbs is False and (step - 1) % opt.db_emb_update_steps == 0:
                retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model,
                                                                      retriever_db_dataloader)  # no grad
            elif opt.use_gt_dbs is True:
                if opt.use_retriever_for_gt is True and (step - 1) % opt.db_emb_update_steps == 0:
                    retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model,
                                                                          retriever_db_dataloader)  # no grad
                elif opt.use_retriever_for_gt is False:
                    retriever_all_dbs_embeddings = None
            (index, resp_ori_input_ids, resp_ori_mask, generator_context_input_ids, generator_context_mask,
             retriever_context_input_ids, retriever_context_mask, retriever_context_token_type,
             ranker_context_input_ids, ranker_context_mask, ranker_context_token_type,
             resp_delex_mask, gt_db_idx, times_matrix) = batch
            # retriever model get top-k db index
            if opt.use_gt_dbs is False:
                retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                               attention_mask=retriever_context_mask.long().cuda(),
                                                               token_type_ids=retriever_context_token_type.long().cuda(),
                                                               output_hidden_states=True,
                                                               return_dict=True,
                                                               sent_emb=True).pooler_output  # have grad
                retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                        retriever_all_dbs_embeddings)  # (bs, all_db_num)
                retriever_top_k_dbs_index = retriever_all_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs].unsqueeze(2)  # (bs, top_k, 1)
            else:
                if opt.use_retriever_for_gt is False:
                    retriever_top_k_dbs_index = gt_db_idx[:, :opt.top_k_dbs].unsqueeze(2)  # (bs, top_k, 1)
                else:
                    # retriever model get top-k db index
                    retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                                   attention_mask=retriever_context_mask.long().cuda(),
                                                                   token_type_ids=retriever_context_token_type.long().cuda(),
                                                                   output_hidden_states=True,
                                                                   return_dict=True,
                                                                   sent_emb=True).pooler_output  # have grad
                    retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                            retriever_all_dbs_embeddings)  # (bs, all_db_num)
                    retriever_gt_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                           gt_db_idx.long())  # (bs, gt_db_num)
                    retriever_top_k_dbs_index = retriever_gt_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs]  # (bs, top_k)
                    retriever_top_k_dbs_index = torch.gather(gt_db_idx, 1, retriever_top_k_dbs_index.long()).unsqueeze(2)  # (bs, top_k, 1)
            # get top-k db generator inputs and concat with context inputs and forward into generator model
            bsz = retriever_top_k_dbs_index.size(0)
            generator_db_len = generator_all_dbs_ids.size(-1)
            generator_top_k_dbs_ids = torch.gather(generator_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                   retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_top_k_dbs_mask = torch.gather(generator_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                    retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_context_top_k_dbs_input_ids = concat_context_and_dbs_input(generator_context_input_ids,
                                                                                 generator_top_k_dbs_ids)
            generator_context_top_k_dbs_mask = concat_context_and_dbs_input(generator_context_mask,
                                                                            generator_top_k_dbs_mask)
            # get top-k db ranker inputs and concat with context inputs and forward into ranker model
            ranker_db_len = ranker_all_dbs_ids.size(-1)
            ranker_top_k_dbs_ids = torch.gather(ranker_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_top_k_dbs_mask = torch.gather(ranker_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                 retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_top_k_dbs_token_type = torch.gather(ranker_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                       retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_context_top_k_dbs_input_ids = concat_context_and_dbs_input(ranker_context_input_ids,
                                                                              ranker_top_k_dbs_ids)
            ranker_context_top_k_dbs_mask = concat_context_and_dbs_input(ranker_context_mask,
                                                                         ranker_top_k_dbs_mask)
            ranker_context_top_k_dbs_token_type = concat_context_and_dbs_input(ranker_context_token_type,
                                                                               ranker_top_k_dbs_token_type)
            if times_matrix is not None and opt.dataset_name != "smd":  # maybe all zeros
                times_matrix = torch.gather(times_matrix, 1, retriever_top_k_dbs_index.long().repeat(1, 1, times_matrix.size(2)))  # (bs, top_k, num_attr)

            if opt.use_gt_dbs is False:
                # re-calc top-k dbs embedding (have grad) then get top-k retrieve scores
                retriever_db_len = retriever_all_dbs_ids.size(-1)
                retriever_top_k_dbs_ids = torch.gather(retriever_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                       retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))
                retriever_top_k_dbs_mask = torch.gather(retriever_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                        retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))
                retriever_top_k_dbs_token_type = torch.gather(
                    retriever_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                    retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))
                retriever_top_k_dbs_embeddings = retriever_model(
                    input_ids=retriever_top_k_dbs_ids.view(-1, retriever_db_len).long().cuda(),
                    attention_mask=retriever_top_k_dbs_mask.view(-1, retriever_db_len).long().cuda(),
                    token_type_ids=retriever_top_k_dbs_token_type.view(-1, retriever_db_len).long().cuda(),
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True).pooler_output.view(bsz, opt.top_k_dbs, -1)  # have grad
                retriever_top_k_dbs_scores = torch.einsum("bad,bkd->bak", retriever_context_embeddings.unsqueeze(1),
                                                          retriever_top_k_dbs_embeddings).squeeze(1)  # (bs, top_k)
            else:
                if opt.use_retriever_for_gt is False:
                    retriever_top_k_dbs_scores = None
                else:
                    retriever_db_len = retriever_all_dbs_ids.size(-1)
                    retriever_top_k_dbs_ids = torch.gather(retriever_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                           retriever_top_k_dbs_index.long().repeat(1, 1,
                                                                                                   retriever_db_len))
                    retriever_top_k_dbs_mask = torch.gather(retriever_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                            retriever_top_k_dbs_index.long().repeat(1, 1,
                                                                                                    retriever_db_len))
                    retriever_top_k_dbs_token_type = torch.gather(
                        retriever_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                        retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))
                    retriever_top_k_dbs_embeddings = retriever_model(
                        input_ids=retriever_top_k_dbs_ids.view(-1, retriever_db_len).long().cuda(),
                        attention_mask=retriever_top_k_dbs_mask.view(-1, retriever_db_len).long().cuda(),
                        token_type_ids=retriever_top_k_dbs_token_type.view(-1, retriever_db_len).long().cuda(),
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True).pooler_output.view(bsz, opt.top_k_dbs, -1)  # have grad
                    retriever_top_k_dbs_scores = torch.einsum("bad,bkd->bak", retriever_context_embeddings.unsqueeze(1),
                                                              retriever_top_k_dbs_embeddings).squeeze(1)  # (bs, top_k)
            if opt.use_ranker is True:  # step operations is inside
                ranker_outputs = ranker_model(
                    input_ids=ranker_context_top_k_dbs_input_ids.long().cuda(),
                    attention_mask=ranker_context_top_k_dbs_mask.cuda(),
                    token_type_ids=ranker_context_top_k_dbs_token_type.cuda(),
                    times_matrix=times_matrix.cuda() if times_matrix is not None else None,
                    step=step,
                    retriever_top_k_dbs_scores=retriever_top_k_dbs_scores.detach() if retriever_top_k_dbs_scores is not None else None,
                    generator_sep_id=generator_db_collator.sep_id,
                    generator_db_id=generator_db_collator.db_id,
                    generator_input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                    generator_attention_mask=generator_context_top_k_dbs_mask.cuda(),
                )
                ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = ranker_outputs
            else:
                ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = None, generator_context_top_k_dbs_mask

            generator_outputs = generator_model(
                input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                attention_mask=generator_context_top_k_dbs_top_r_attr_mask.cuda(),
                labels=resp_ori_input_ids.long().cuda(),
                resp_delex_mask=resp_delex_mask.cuda() if resp_delex_mask is not None else None,
                step=step,
            )  # (loss, logits,)
            generator_loss, decoder_cross_attention_scores = generator_outputs[0], generator_outputs[3]

            if decoder_cross_attention_scores is not None and retriever_top_k_dbs_scores is not None:
                distill_label = decoder_cross_attention_scores.detach()
                if opt.generator_distill_retriever_loss_type == "kl":
                    retriever_loss = kldivloss(retriever_top_k_dbs_scores, distill_label)
                elif opt.generator_distill_retriever_loss_type == "ce":
                    distill_label = F.softmax(distill_label, dim=-1)
                    retriever_loss = SoftCrossEntropy(retriever_top_k_dbs_scores, distill_label)
                else:
                    raise ValueError
            else:
                retriever_loss = None

            train_loss = generator_loss
            if ranker_times_loss is not None:
                train_loss = train_loss + opt.ranker_times_matrix_loss_alpha * ranker_times_loss
            if retriever_loss is not None:
                train_loss = train_loss + opt.generator_distill_retriever_loss_alpha * retriever_loss

            train_loss = train_loss / opt.accumulation_steps
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(), opt.clip)
                generator_optimizer.step()
                generator_scheduler.step()
                generator_model.zero_grad()
                if retriever_loss is not None:
                    torch.nn.utils.clip_grad_norm_(retriever_model.parameters(), opt.clip)
                    retriever_optimizer.step()
                    retriever_scheduler.step()
                    retriever_model.zero_grad()
                if ranker_times_loss is not None:
                    torch.nn.utils.clip_grad_norm_(ranker_model.parameters(), opt.clip)
                    ranker_optimizer.step()
                    ranker_scheduler.step()
                    ranker_model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if (step - 1) % opt.eval_freq == 0 and step > opt.start_eval_step:
                if opt.is_main:
                    logger.warning("Start evaluation")
                dev_score, dev_metric = evaluate(generator_model, retriever_model, ranker_model, eval_dial_dataset,
                                                 dial_collator, generator_tokenizer, opt, retriever_all_dbs_embeddings,
                                                 generator_all_dbs_ids, generator_all_dbs_mask, ranker_all_dbs_ids,
                                                 ranker_all_dbs_mask, ranker_all_dbs_token_type, step,
                                                 generator_db_collator)
                test_score, test_metric = evaluate(generator_model, retriever_model, ranker_model, test_dial_dataset,
                                                   dial_collator, generator_tokenizer, opt, retriever_all_dbs_embeddings,
                                                   generator_all_dbs_ids, generator_all_dbs_mask, ranker_all_dbs_ids,
                                                   ranker_all_dbs_mask, ranker_all_dbs_token_type, step,
                                                   generator_db_collator)
                if opt.is_main:
                    logger.warning("Continue training")
                generator_model.train()
                retriever_model.train()
                ranker_model.train()
                if opt.is_main:
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        src.util.save(generator_model, generator_optimizer, generator_scheduler, step, best_dev_score,
                                      opt, checkpoint_path, 'generator_best_dev')
                        src.util.save(retriever_model, retriever_optimizer, retriever_scheduler, step, best_dev_score,
                                      opt, checkpoint_path, 'retriever_best_dev')
                        src.util.save(ranker_model, ranker_optimizer, ranker_scheduler, step, best_dev_score,
                                      opt, checkpoint_path, 'ranker_best_dev')
                        if opt.use_gt_dbs is False or (opt.use_gt_dbs is True and opt.use_retriever_for_gt is True):
                            np.save(checkpoint_path / "checkpoint" / "retriever_best_dev" /
                                    "retriever_all_dbs_embeddings.npy", retriever_all_dbs_embeddings.numpy())
                        metric_path = checkpoint_path / "checkpoint" / "generator_best_dev" / 'metric.json'
                        final_metric = {"val": copy.deepcopy(dev_metric), "test": copy.deepcopy(test_metric)}
                        for s, metric in final_metric.items():
                            for k, v in metric.items():
                                final_metric[s][k] = str(v)
                        with open(metric_path, 'w', encoding='utf-8') as fout:
                            json.dump(final_metric, fout, indent=4)
                    log = f"{step} / {training_steps} |"
                    log += f" Train Loss: {curr_loss / opt.eval_freq * opt.accumulation_steps:.3f} |"
                    for key, val in dev_metric.items():
                        log += f" Evaluation {key}: {val:.2f} |"
                    for key, val in test_metric.items():
                        log += f" Test {key}: {val:.2f} |"
                    log += f" glr: {generator_scheduler.get_last_lr()[0]:.5f}"
                    log += f" rlr: {retriever_scheduler.get_last_lr()[0]:.5f}"
                    log += f" klr: {ranker_scheduler.get_last_lr()[0]:.5f}"
                    logger.warning(log)
                    if tb_logger is not None:
                        for key, val in dev_metric.items():
                            tb_logger.add_scalar("Evaluation {}".format(key), val, step)
                        for key, val in test_metric.items():
                            tb_logger.add_scalar("Test {}".format(key), val, step)
                        tb_logger.add_scalar("Training Loss", curr_loss / opt.eval_freq, step)
                    curr_loss = 0.

            if opt.is_main and (step - 1) % opt.save_freq == 0 and step > opt.start_eval_step:
                src.util.save(generator_model, generator_optimizer, generator_scheduler, step, best_dev_score,
                              opt, checkpoint_path, f"generator_step-{step}")
                src.util.save(retriever_model, retriever_optimizer, retriever_scheduler, step, best_dev_score,
                              opt, checkpoint_path, f"retriever_step-{step}")
                src.util.save(ranker_model, ranker_optimizer, ranker_scheduler, step, best_dev_score,
                              opt, checkpoint_path, f"ranker_step-{step}")
                if opt.use_gt_dbs is False or (opt.use_gt_dbs is True and opt.use_retriever_for_gt is True):
                    np.save(checkpoint_path / "checkpoint" / f"retriever_step-{step}" /
                            "retriever_all_dbs_embeddings.npy", retriever_all_dbs_embeddings.numpy())
            if step > training_steps:
                break


def evaluate(generator_model, retriever_model, ranker_model, eval_dial_dataset, dial_collator, generator_tokenizer, opt,
             retriever_all_dbs_embeddings, generator_all_dbs_ids, generator_all_dbs_mask, ranker_all_dbs_ids,
             ranker_all_dbs_mask, ranker_all_dbs_token_type, step, generator_db_collator):
    sampler = SequentialSampler(eval_dial_dataset)
    eval_dial_dataloader = DataLoader(eval_dial_dataset,
                                      sampler=sampler,
                                      batch_size=opt.per_gpu_eval_batch_size,
                                      drop_last=False,
                                      num_workers=10,
                                      collate_fn=dial_collator
                                      )
    generator_model.eval()
    retriever_model.eval()
    ranker_model.eval()
    results = []
    retrieve_results = []
    raw_data = []
    generator_model = generator_model.module if hasattr(generator_model, "module") else generator_model
    retriever_model = retriever_model.module if hasattr(retriever_model, "module") else retriever_model
    ranker_model = ranker_model.module if hasattr(ranker_model, "module") else ranker_model
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dial_dataloader)):
            (index, resp_ori_input_ids, resp_ori_mask, generator_context_input_ids, generator_context_mask,
             retriever_context_input_ids, retriever_context_mask, retriever_context_token_type,
             ranker_context_input_ids, ranker_context_mask, ranker_context_token_type,
             resp_delex_mask, gt_db_idx, times_matrix) = batch
            if opt.use_gt_dbs is False:
                # retriever model get top-k db index
                retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                               attention_mask=retriever_context_mask.long().cuda(),
                                                               token_type_ids=retriever_context_token_type.long().cuda(),
                                                               output_hidden_states=True,
                                                               return_dict=True,
                                                               sent_emb=True).pooler_output  # have grad
                retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                        retriever_all_dbs_embeddings)  # (bs, all_db_num)
                retriever_top_k_dbs_index = retriever_all_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs].unsqueeze(2)  # (bs, top_k, 1)
            else:
                if opt.use_retriever_for_gt is False:
                    retriever_top_k_dbs_index = gt_db_idx[:, :opt.top_k_dbs].unsqueeze(2)  # (bs, top_k, 1)
                    retriever_all_dbs_scores = torch.zeros(
                        [retriever_top_k_dbs_index.size(0), generator_all_dbs_ids.size(0)])  # (bs, all_db_num)
                    retriever_all_dbs_scores = torch.scatter(retriever_all_dbs_scores, 1,
                                                             retriever_top_k_dbs_index.squeeze(-1).long(),
                                                             torch.ones_like(retriever_top_k_dbs_index.squeeze(-1),
                                                                             dtype=retriever_all_dbs_scores.dtype))  # (bs, all_db_num)
                else:
                    # retriever model get top-k db index
                    retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                                   attention_mask=retriever_context_mask.long().cuda(),
                                                                   token_type_ids=retriever_context_token_type.long().cuda(),
                                                                   output_hidden_states=True,
                                                                   return_dict=True,
                                                                   sent_emb=True).pooler_output  # have grad
                    retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                            retriever_all_dbs_embeddings)  # (bs, all_db_num)
                    retriever_gt_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                           gt_db_idx.long())  # (bs, gt_db_num)
                    retriever_top_k_dbs_index = retriever_gt_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs]  # (bs, top_k)
                    retriever_top_k_dbs_index = torch.gather(gt_db_idx, 1, retriever_top_k_dbs_index.long()).unsqueeze(2)  # (bs, top_k, 1)
            # get top-k db generator inputs and concat with context inputs and forward into generator model
            bsz = retriever_top_k_dbs_index.size(0)
            generator_db_len = generator_all_dbs_ids.size(-1)
            generator_top_k_dbs_ids = torch.gather(generator_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                   retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_top_k_dbs_mask = torch.gather(generator_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                    retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_context_top_k_dbs_input_ids = concat_context_and_dbs_input(generator_context_input_ids,
                                                                                 generator_top_k_dbs_ids)
            generator_context_top_k_dbs_mask = concat_context_and_dbs_input(generator_context_mask,
                                                                            generator_top_k_dbs_mask)
            # get top-k db ranker inputs and concat with context inputs and forward into ranker model
            ranker_db_len = ranker_all_dbs_ids.size(-1)
            ranker_top_k_dbs_ids = torch.gather(ranker_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_top_k_dbs_mask = torch.gather(ranker_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                 retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_top_k_dbs_token_type = torch.gather(ranker_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                       retriever_top_k_dbs_index.long().repeat(1, 1, ranker_db_len))
            ranker_context_top_k_dbs_input_ids = concat_context_and_dbs_input(ranker_context_input_ids,
                                                                              ranker_top_k_dbs_ids)
            ranker_context_top_k_dbs_mask = concat_context_and_dbs_input(ranker_context_mask,
                                                                         ranker_top_k_dbs_mask)
            ranker_context_top_k_dbs_token_type = concat_context_and_dbs_input(ranker_context_token_type,
                                                                               ranker_top_k_dbs_token_type)
            if opt.use_gt_dbs is False:
                retriever_top_k_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                          retriever_top_k_dbs_index.long().squeeze(2))  # (bs, top_k)
            else:
                if opt.use_retriever_for_gt is False:
                    retriever_top_k_dbs_scores = None
                else:
                    retriever_top_k_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                              retriever_top_k_dbs_index.long().squeeze(2))  # (bs, top_k)
            if opt.use_ranker is True:  # step operations is inside
                ranker_outputs = ranker_model(
                    input_ids=ranker_context_top_k_dbs_input_ids.long().cuda(),
                    attention_mask=ranker_context_top_k_dbs_mask.cuda(),
                    token_type_ids=ranker_context_top_k_dbs_token_type.cuda(),
                    times_matrix=times_matrix,
                    step=step,
                    retriever_top_k_dbs_scores=retriever_top_k_dbs_scores.cuda() if retriever_top_k_dbs_scores is not None else None,
                    generator_sep_id=generator_db_collator.sep_id,
                    generator_db_id=generator_db_collator.db_id,
                    generator_input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                    generator_attention_mask=generator_context_top_k_dbs_mask.cuda(),
                )
                ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = ranker_outputs
            else:
                ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = None, generator_context_top_k_dbs_mask

            generator_outputs = generator_model.generate(
                input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                attention_mask=generator_context_top_k_dbs_top_r_attr_mask.cuda(),
                max_length=opt.answer_maxlength,
            )
            for k, o in enumerate(generator_outputs):
                result = []
                ans = generator_tokenizer.decode(o, skip_special_tokens=True).lower()
                example = eval_dial_dataset.get_example(index[k])
                raw_data.append(example)
                if opt.dataset_name == "mwoz_gptke":
                    result.append(ans)
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])
                    if "data1" in opt.eval_data:
                        result.append(example["kb"])
                    result.append(example["type"])
                elif opt.dataset_name == "camrest":
                    result.append(ans)
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])
                elif opt.dataset_name == "smd":
                    result.append(ans)
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])
                else:
                    raise NotImplementedError
                results.append(result)
            retrieve_results.append(retriever_all_dbs_scores.detach().cpu())
    retrieve_results = torch.cat(retrieve_results, dim=0)
    if opt.is_distributed:
        output = [None for _ in range(opt.world_size)]
        dist.all_gather_object(output, results)
        new_results = []
        for r in output:
            new_results += r
        results = new_results
    if opt.dataset_name == "mwoz_gptke" and "data1" in opt.eval_data:
        if opt.metric_version == "new1":
            METRIC = evaluation.Metric_data1_new1(results)
        else:
            raise NotImplementedError
    elif opt.dataset_name == "camrest" and "data0" in opt.eval_data:
        if opt.metric_version == "new1":
            METRIC = evaluation.Metric_data0_new1(results)
        else:
            raise NotImplementedError
    elif opt.dataset_name == "smd" and "data0" in opt.eval_data:
        if opt.metric_version == "new1":
            METRIC = evaluation.Metric_data0_new1(results)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    reader_metrics = METRIC.baseline_reader_metric()
    if opt.dataset_name != "smd":
        RETRIEVE_METRIC = evaluation.Retrieve_Metric(retrieve_results, data=raw_data, db=data_turn_batch.load_dbs(opt.dbs))
        retrieve_metrics = RETRIEVE_METRIC.calc_recall(level="turn_level", top_k=opt.top_k_dbs, first_turn_name=True)
        for k, v in retrieve_metrics.items():
            v, _ = src.util.weighted_average(v, len(raw_data), opt)
            retrieve_metrics[k] = v
        reader_metrics.update(retrieve_metrics)
    return reader_metrics[opt.model_select_metric], reader_metrics


def run(opt, checkpoint_path):
    if opt.dataset_name == "mwoz_gptke":
        special_tokens = ["<user>", "<sys>", "<api>", "<sys-api>", "<database>", "<sep_attributes>"]
    elif opt.dataset_name == "camrest":
        special_tokens = ["<user>", "<sys>", "<database>", "<sep_attributes>"]
    elif opt.dataset_name == "smd":
        special_tokens = ["<user>", "<sys>", "<database>", "<sep_attributes>"]
    else:
        raise NotImplementedError
    # generator model
    generator_model_name = 't5-' + opt.model_size
    generator_model_class = src.model.FiDT5
    generator_tokenizer = transformers.T5Tokenizer.from_pretrained(generator_model_name)
    _ = generator_tokenizer.add_tokens(special_tokens)
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(generator_model_name)
    t5.resize_token_embeddings(len(generator_tokenizer))  # len vocabulary is not absolutely correct.
    generator_model = generator_model_class(t5.config, model_args=opt)
    generator_model.load_t5(t5.state_dict())
    generator_model = generator_model.to(opt.local_rank)
    generator_model.set_checkpoint(opt.use_checkpoint)
    generator_optimizer, generator_scheduler = src.util.set_optim(opt, generator_model)
    # retriever model
    retriever_model_name = opt.retriever_model_name
    retriever_model_class = src.simcse_model.BertForCL
    retriever_tokenizer = transformers.BertTokenizer.from_pretrained(retriever_model_name)
    retriever_model = retriever_model_class.from_pretrained(retriever_model_name)  # dont need to add token
    retriever_model = retriever_model.to(opt.local_rank)
    retriever_optimizer, retriever_scheduler = src.util.set_retriever_optim(opt, retriever_model)
    # ranker model
    ranker_model_name = "bert-base-uncased"
    ranker_model_class = src.ranker_model.BertForRank
    ranker_tokenizer = transformers.BertTokenizer.from_pretrained(ranker_model_name)
    _ = ranker_tokenizer.add_tokens(special_tokens)
    ranker_model = ranker_model_class.from_pretrained(ranker_model_name, model_args=opt)
    ranker_model.resize_token_embeddings(len(ranker_tokenizer))
    ranker_model = ranker_model.to(opt.local_rank)
    ranker_optimizer, ranker_scheduler = src.util.set_ranker_optim(opt, ranker_model)

    if opt.is_distributed:
        generator_model = torch.nn.parallel.DistributedDataParallel(
            generator_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        retriever_model = torch.nn.parallel.DistributedDataParallel(
            retriever_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        ranker_model = torch.nn.parallel.DistributedDataParallel(
            ranker_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    # use global rank and world size to split the train set on multiple gpus
    train_dial_examples = data_turn_batch.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dial_dataset = data_turn_batch.DialDataset(train_dial_examples, use_delex=opt.use_delex,
                                                     use_times_matrix=opt.ranker_times_matrix, use_gt_dbs=opt.use_gt_dbs)
    # use global rank and world size to split the eval set on multiple gpus
    eval_dial_examples = data_turn_batch.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dial_dataset = data_turn_batch.DialDataset(eval_dial_examples, use_gt_dbs=opt.use_gt_dbs)
    # use global rank and world size to split the eval set on multiple gpus
    test_dial_examples = data_turn_batch.load_data(
        opt.test_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    test_dial_dataset = data_turn_batch.DialDataset(test_dial_examples, use_gt_dbs=opt.use_gt_dbs)
    dial_collator = data_turn_batch.DialCollator(generator_tokenizer, retriever_tokenizer, ranker_tokenizer,
                                                 opt.generator_text_maxlength, opt.retriever_text_maxlength,
                                                 opt.ranker_text_maxlength, opt.answer_maxlength)

    db_examples = data_turn_batch.load_dbs(opt.dbs)
    db_dataset = data_turn_batch.DBDataset(db_examples, opt.db_type, use_dk=opt.use_dk, dk_mask=opt.dk_mask)
    generator_db_collator = data_turn_batch.DBCollator(generator_tokenizer, opt.generator_db_maxlength,
                                                       type="generator")
    retriever_db_collator = data_turn_batch.DBCollator(retriever_tokenizer, opt.retriever_db_maxlength,
                                                       type="retriever")
    ranker_db_collator = data_turn_batch.DBCollator(ranker_tokenizer, opt.ranker_db_maxlength, type="ranker")

    step, best_dev_score = 0, 0.0
    if opt.is_main:
        logger.warning("Start training")
    train(
        generator_model,
        retriever_model,
        ranker_model,
        generator_tokenizer,
        retriever_tokenizer,
        ranker_tokenizer,
        generator_optimizer,
        generator_scheduler,
        retriever_optimizer,
        retriever_scheduler,
        ranker_optimizer,
        ranker_scheduler,
        step,
        train_dial_dataset,
        eval_dial_dataset,
        test_dial_dataset,
        dial_collator,
        db_dataset,
        generator_db_collator,
        retriever_db_collator,
        ranker_db_collator,
        opt,
        best_dev_score,
        checkpoint_path
    )


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_eval_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    add_exp_name = f"_seed-{opt.seed}_gtml-{opt.generator_text_maxlength}_rtml-{opt.retriever_text_maxlength}_rkml-{opt.ranker_text_maxlength}" \
                   f"_asml-{opt.answer_maxlength}_gdml-{opt.generator_db_maxlength}_rdml-{opt.retriever_db_maxlength}_kdml-{opt.ranker_db_maxlength}" \
                   f"_dus-{opt.db_emb_update_steps}_topd-{opt.top_k_dbs}"
    # assert opt.ranker_times_matrix_start_step <= opt.rank_attribute_start_step <= opt.generator_distill_retriever_start_step
    assert opt.ranker_times_matrix_start_step <= opt.rank_attribute_start_step
    if opt.use_ranker is True:
        assert opt.use_dk is True
        if opt.ranker_attribute_ways == "top_r":
            add_exp_name += f"_topa-{opt.top_r_attr}"
        elif opt.ranker_attribute_ways == "threshold":
            add_exp_name += f"_tha-{opt.threshold_attr}"
        add_exp_name += f"_rk-{opt.rank_attribute_start_step}-{opt.rank_attribute_pooling}"
        if opt.rank_no_retriever_weighted is True:
            add_exp_name += "-nrw"
        if opt.ranker_times_matrix is True:
            add_exp_name += f"-tm-{opt.ranker_times_matrix_start_step}-{opt.ranker_times_matrix_loss_type}-{opt.ranker_times_matrix_loss_alpha}-{opt.ranker_times_matrix_query}"
        if opt.ranker_lr != 1e-4:
            add_exp_name += f"-rklr-{opt.ranker_lr}"
    if opt.generator_distill_retriever is True:
        add_exp_name += f"_gdre-{opt.generator_distill_retriever_start_step}-{opt.generator_distill_retriever_pooling}-{opt.generator_distill_retriever_loss_type}-{opt.generator_distill_retriever_loss_alpha}"
        if opt.retriever_lr != 1e-4:
            add_exp_name += f"-relr-{opt.retriever_lr}"
    if opt.use_delex is True:
        add_exp_name += "_dlx"
    if opt.use_dk is True:
        add_exp_name += "_dk"
        if opt.dk_mask is True:
            add_exp_name += "-mk"
    if opt.end_eval_step != 32000:
        add_exp_name += f"_es-{opt.end_eval_step}"
    if opt.use_gt_dbs is True:
        add_exp_name += "_gtdb"
    if opt.model_select_metric != "MICRO-F1":
        add_exp_name += f"_msm-{opt.model_select_metric}"
    opt.name += add_exp_name

    opt.scheduler_steps = opt.total_steps // opt.accumulation_steps
    opt.warmup_steps = opt.warmup_steps // opt.accumulation_steps

    opt.retriever_scheduler_steps = (opt.retriever_total_steps - opt.generator_distill_retriever_start_step) // opt.retriever_accumulation_steps
    opt.retriever_warmup_steps = (opt.retriever_warmup_steps - opt.generator_distill_retriever_start_step) // opt.retriever_accumulation_steps if (opt.retriever_warmup_steps - opt.generator_distill_retriever_start_step) > 0 else 0

    opt.ranker_scheduler_steps = (opt.ranker_total_steps - opt.ranker_times_matrix_start_step) // opt.ranker_accumulation_steps
    opt.ranker_warmup_steps = (opt.ranker_warmup_steps - opt.ranker_times_matrix_start_step) // opt.ranker_accumulation_steps if (opt.ranker_warmup_steps - opt.ranker_times_matrix_start_step) > 0 else 0

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )
    if not checkpoint_exists and opt.is_main:
        options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    if opt.dataset_name == "mwoz_gptke":
        from data_code.mwoz_gptke import data_turn_batch, evaluation
    elif opt.dataset_name == "camrest":
        from data_code.camrest import data_turn_batch, evaluation
    elif opt.dataset_name == "smd":
        from data_code.smd import data_turn_batch, evaluation
    else:
        raise NotImplementedError
    run(opt, checkpoint_path)
