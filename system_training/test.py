import copy
import json
import os
import csv
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import src.slurm
import src.util
from src.options import Options
import src.model
import src.simcse_model
import src.ranker_model


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
                    retriever_all_dbs_scores = torch.zeros([retriever_top_k_dbs_index.size(0), generator_all_dbs_ids.size(0)])  # (bs, all_db_num)
                    retriever_all_dbs_scores = torch.scatter(retriever_all_dbs_scores, 1,
                                                             retriever_top_k_dbs_index.squeeze(-1).long(),
                                                             torch.ones_like(retriever_top_k_dbs_index.squeeze(-1), dtype=retriever_all_dbs_scores.dtype))  # (bs, all_db_num)
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
                    retriever_gt_dbs_scores = torch.gather(retriever_all_dbs_scores, 1, gt_db_idx.long())  # (bs, gt_db_num)
                    retriever_top_k_dbs_index = retriever_gt_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs]  # (bs, top_k)
                    retriever_top_k_dbs_index = torch.gather(gt_db_idx, 1, retriever_top_k_dbs_index.long()).unsqueeze(2)  # (bs, top_k, 1)
                    retriever_all_dbs_scores = torch.zeros([retriever_top_k_dbs_index.size(0), generator_all_dbs_ids.size(0)])  # (bs, all_db_num)
                    retriever_all_dbs_scores = torch.scatter(retriever_all_dbs_scores, 1,
                                                             retriever_top_k_dbs_index.squeeze(-1).long(),
                                                             torch.ones_like(retriever_top_k_dbs_index.squeeze(-1),
                                                                             dtype=retriever_all_dbs_scores.dtype))  # (bs, all_db_num)
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
                if opt.write_generate_result is False:
                    ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = ranker_outputs
                elif opt.ranker_attribute_ways == "threshold":
                    ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask, ranker_threshold_attr_mask = ranker_outputs
                else:
                    raise ValueError
            else:
                ranker_times_loss, generator_context_top_k_dbs_top_r_attr_mask = None, generator_context_top_k_dbs_mask

            generator_outputs = generator_model.generate(
                input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                attention_mask=generator_context_top_k_dbs_top_r_attr_mask.cuda(),
                max_length=opt.answer_maxlength,
                num_beams=opt.num_beams,
                repetition_penalty=opt.repetition_penalty,
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
                    if "data1" in opt.test_data:
                        result.append(example["kb"])
                    result.append(example["type"])
                    if opt.write_generate_result is True and opt.use_ranker is True and opt.ranker_attribute_ways == "threshold":
                        result.append(ranker_threshold_attr_mask[k].tolist())
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
    if opt.dataset_name == "mwoz_gptke" and "data1" in opt.test_data:
        if opt.metric_version == "new1":
            METRIC = [evaluation.Metric_data1_new1(results)]
        else:
            raise NotImplementedError
    elif opt.dataset_name == "camrest" and "data0" in opt.test_data:
        if opt.metric_version == "new1":
            METRIC = [evaluation.Metric_data0_new1(results)]
        else:
            raise NotImplementedError
    elif opt.dataset_name == "smd" and "data0" in opt.test_data:
        if opt.metric_version == "new1":
            METRIC = [evaluation.Metric_data0_new1(results)]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    for metric in METRIC:
        reader_metrics = metric.baseline_reader_metric()
        if opt.dataset_name != "smd":
            RETRIEVE_METRIC = evaluation.Retrieve_Metric(retrieve_results, data=raw_data, db=data_turn_batch.load_dbs(opt.dbs))
            retrieve_metrics = RETRIEVE_METRIC.calc_recall(level="turn_level", top_k=opt.top_k_dbs, first_turn_name=True)
            for k, v in retrieve_metrics.items():
                v, _ = src.util.weighted_average(v, len(raw_data), opt)
                retrieve_metrics[k] = v
            reader_metrics.update(retrieve_metrics)
        print(json.dumps(reader_metrics))
    return results


def run(opt):
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
    generator_model = generator_model_class.from_pretrained(opt.test_model_path, model_args=opt)
    generator_model = generator_model.to(opt.device)
    # retriever model
    retriever_model_name = opt.retriever_model_name
    retriever_model_class = src.simcse_model.BertForCL
    retriever_tokenizer = transformers.BertTokenizer.from_pretrained(retriever_model_name)
    retriever_model = retriever_model_class.from_pretrained(opt.test_model_path.replace("generator", "retriever"))  # dont need to add token
    retriever_model = retriever_model.to(opt.local_rank)
    # ranker model
    ranker_model_name = "bert-base-uncased"
    ranker_model_class = src.ranker_model.BertForRank
    ranker_tokenizer = transformers.BertTokenizer.from_pretrained(ranker_model_name)
    _ = ranker_tokenizer.add_tokens(special_tokens)
    ranker_model = ranker_model_class.from_pretrained(opt.test_model_path.replace("generator", "ranker"), model_args=opt)
    ranker_model = ranker_model.to(opt.local_rank)
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
    ranker_db_collator = data_turn_batch.DBCollator(ranker_tokenizer, opt.ranker_db_maxlength, type="ranker")
    db_sampler = SequentialSampler(db_dataset)
    generator_db_dataloader = DataLoader(db_dataset,
                                         sampler=db_sampler,
                                         batch_size=222,
                                         drop_last=False,
                                         num_workers=10,
                                         collate_fn=generator_db_collator)
    ranker_db_dataloader = DataLoader(db_dataset,
                                      sampler=db_sampler,
                                      batch_size=222,
                                      drop_last=False,
                                      num_workers=10,
                                      collate_fn=ranker_db_collator)
    generator_all_dbs_ids, generator_all_dbs_mask, _, _ = get_all_dbs_inputs(generator_db_dataloader)
    ranker_all_dbs_ids, ranker_all_dbs_mask, ranker_all_dbs_token_type, _ = get_all_dbs_inputs(ranker_db_dataloader)
    if opt.is_main:
        print("Start eval")
    if opt.use_gt_dbs is False:
        retriever_all_dbs_embeddings = torch.tensor(np.load(opt.test_model_path.replace("generator", "retriever") +
                                                            "/retriever_all_dbs_embeddings.npy"))
    else:
        if opt.use_retriever_for_gt is False:
            retriever_all_dbs_embeddings = None
        else:
            try:
                retriever_all_dbs_embeddings = torch.tensor(np.load(opt.test_model_path.replace("generator", "retriever") +
                                                            "/retriever_all_dbs_embeddings.npy"))
            except:
                retriever_db_collator = data_turn_batch.DBCollator(retriever_tokenizer, opt.retriever_db_maxlength,
                                                                   type="retriever")
                retriever_db_dataloader = DataLoader(db_dataset,
                                                     sampler=db_sampler,
                                                     batch_size=222,
                                                     drop_last=False,
                                                     num_workers=10,
                                                     collate_fn=retriever_db_collator)
                retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model,
                                                                      retriever_db_dataloader)  # no grad
    prediction_results = evaluate(generator_model, retriever_model, ranker_model, test_dial_dataset,
                                  dial_collator, generator_tokenizer, opt, retriever_all_dbs_embeddings,
                                  generator_all_dbs_ids, generator_all_dbs_mask, ranker_all_dbs_ids,
                                  ranker_all_dbs_mask, ranker_all_dbs_token_type, opt.end_eval_step,
                                  generator_db_collator)
    if opt.write_generate_result is True:
        test_data_to_write = []
        for i in range(len(test_dial_examples)):
            if "<sys-api>" in test_dial_examples[i]["output_used"]:
                continue
            curr_data = dict()
            curr_data["did"] = test_dial_examples[i]["did"]
            curr_data["turn_num"] = test_dial_examples[i]["turn_num"]
            curr_data["type"] = test_dial_examples[i]["type"]
            curr_data["context"] = test_dial_examples[i]["context"]
            curr_data["gold_entities"] = test_dial_examples[i]["gold_entities"]
            if opt.use_ranker is True and opt.ranker_attribute_ways == "threshold":
                idx2attr = ["address", "area", "domain", "food", "internet", "name", "parking", "phone", "postcode",
                            "pricerange", "stars", "type"]
                attr_list = []
                for idx, mask in enumerate(prediction_results[i][-1]):
                    if mask is True:
                        attr_list.append(idx2attr[idx])
                curr_data["attribute_mask"] = attr_list
            curr_data["output"] = test_dial_examples[i]["output"]
            curr_data["output_generated"] = src.util.preprocess_text(src.util.clean_gen_sentence(prediction_results[i][0]))
            curr_data["gt_db_idx"] = test_dial_examples[i]["gt_db_idx"]
            curr_data["gt_db"] = []
            for idx in curr_data["gt_db_idx"]:
                curr_data["gt_db"].append(db_examples[idx])
            test_data_to_write.append(curr_data)
        with open(opt.generate_result_path, 'w', encoding='utf-8') as fout:
            json.dump(test_data_to_write, fout, indent=4)


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    print(opt.test_model_path)
    print(f"answer_maxlength-{opt.answer_maxlength}_num_beams-{opt.num_beams}_repetition_penalty-{opt.repetition_penalty}")
    if opt.dataset_name == "mwoz_gptke":
        from data_code.mwoz_gptke import data_turn_batch, evaluation
    elif opt.dataset_name == "camrest":
        from data_code.camrest import data_turn_batch, evaluation
    elif opt.dataset_name == "smd":
        from data_code.smd import data_turn_batch, evaluation
    else:
        raise NotImplementedError
    run(opt)
