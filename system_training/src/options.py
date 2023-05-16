import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=0, help='the step meaning is same as total_step')
        self.parser.add_argument('--total_steps', type=int, default=32000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=32)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adamw')
        self.parser.add_argument('--scheduler', type=str, default='linear')
        self.parser.add_argument('--weight_decay', type=float, default=0.01)
        self.parser.add_argument('--fixed_lr', type=bool, default=False)

        self.parser.add_argument('--retriever_warmup_steps', type=int, default=0, help='the step meaning is same as total_step')
        self.parser.add_argument('--retriever_total_steps', type=int, default=32000)
        self.parser.add_argument('--retriever_scheduler_steps', type=int, default=None,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--retriever_accumulation_steps', type=int, default=32)
        self.parser.add_argument('--retriever_dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--retriever_lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--retriever_clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--retriever_optim', type=str, default='adamw')
        self.parser.add_argument('--retriever_scheduler', type=str, default='linear')
        self.parser.add_argument('--retriever_weight_decay', type=float, default=0.01)
        self.parser.add_argument('--retriever_fixed_lr', type=bool, default=False)

        self.parser.add_argument('--ranker_warmup_steps', type=int, default=0,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--ranker_total_steps', type=int, default=32000)
        self.parser.add_argument('--ranker_scheduler_steps', type=int, default=None,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--ranker_accumulation_steps', type=int, default=32)
        self.parser.add_argument('--ranker_dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--ranker_lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--ranker_clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--ranker_optim', type=str, default='adamw')
        self.parser.add_argument('--ranker_scheduler', type=str, default='linear')
        self.parser.add_argument('--ranker_weight_decay', type=float, default=0.01)
        self.parser.add_argument('--ranker_fixed_lr', type=bool, default=False)

    def add_eval_options(self):
        self.parser.add_argument('--test_data', type=str, default='others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_times-cr-dyn/test.json', help='path of test data')
        self.parser.add_argument('--test_model_path', type=str, default='none', help='path of test model')
        self.parser.add_argument('--num_beams', type=int, default=1, help='num beams')
        self.parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition penalty')
        self.parser.add_argument('--write_generate_result', type=bool, default=False, help='write generate result')
        self.parser.add_argument('--generate_result_path', type=str, default='none', help='generate result path')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default='others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_times-cr-dyn/train.json', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_times-cr-dyn/val.json', help='path of eval data')
        self.parser.add_argument('--dbs', type=str, default='others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_times-cr-dyn/all_db.json', help='path of dbs')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the t5 encoder')
        self.parser.add_argument('--generator_text_maxlength', type=int, default=200,
                                 help='maximum number of tokens in context')
        self.parser.add_argument('--retriever_text_maxlength', type=int, default=128,
                                 help='maximum number of tokens in context')
        self.parser.add_argument('--ranker_text_maxlength', type=int, default=200,
                                 help='maximum number of tokens in context')
        self.parser.add_argument('--answer_maxlength', type=int, default=64,
                                 help='maximum number of tokens used of answer to train the model, no truncation if -1')
        self.parser.add_argument('--generator_db_maxlength', type=int, default=100,
                                 help='maximum number of tokens in db text')
        self.parser.add_argument('--retriever_db_maxlength', type=int, default=128,
                                 help='maximum number of tokens in db text')
        self.parser.add_argument('--ranker_db_maxlength', type=int, default=100,
                                 help='maximum number of tokens in db text')
        self.parser.add_argument('--db_type', type=str, default="entrance", help='db type could be triplet or entrance')
        self.parser.add_argument('--db_emb_update_steps', type=int, default=100, help='step to update db text embedding')
        self.parser.add_argument('--top_k_dbs', type=int, default=7, help='top k db num')

        self.parser.add_argument('--use_ranker', type=bool, default=False, help='use ranker')
        self.parser.add_argument('--ranker_attribute_ways', type=str, default="top_r", help='ranker attribute ways top_r/threshold')
        self.parser.add_argument('--top_r_attr', type=int, default=5, help='top r attribute num')
        self.parser.add_argument('--threshold_attr', type=float, default=0.5, help='attribute threshold')
        self.parser.add_argument('--rank_attribute_pooling', type=str, default="avg_wo_context", help='cls/cls_wo_context/avg/avg_wo_context')
        self.parser.add_argument('--rank_attribute_start_step', type=int, default=0, help='rank attribute start step')
        self.parser.add_argument('--rank_no_retriever_weighted', type=bool, default=False, help='rank no retriever weighted')

        self.parser.add_argument('--ranker_times_matrix', type=bool, default=False, help='use times matrix for ranker')
        self.parser.add_argument('--ranker_times_matrix_query', type=str, default="cr", help='times matrix query r/lastcr/cr/cr-iap')
        self.parser.add_argument('--ranker_times_matrix_start_step', type=int, default=0, help='times matrix start step')
        self.parser.add_argument('--ranker_times_matrix_loss_alpha', type=float, default=1.0, help='times matrix loss alpha')
        self.parser.add_argument('--ranker_times_matrix_loss_type', type=str, default="ce", help='times matrix loss type kl/ce/bce')

        self.parser.add_argument('--generator_distill_retriever', type=bool, default=False, help='generator distill retriever')
        self.parser.add_argument('--generator_distill_retriever_start_step', type=int, default=0, help='distill start step')
        self.parser.add_argument('--generator_distill_retriever_pooling', type=str, default="avg_wo_context", help='cls/avg/avg_wo_context')
        self.parser.add_argument('--generator_distill_retriever_loss_alpha', type=float, default=1.0, help='distill loss alpha')
        self.parser.add_argument('--generator_distill_retriever_loss_type', type=str, default="kl", help='distill loss type: kl/ce')
        self.parser.add_argument('--use_delex', type=bool, default=False, help='use resp delex for generator distill')

        self.parser.add_argument('--use_dk', type=bool, default=False, help='use dontknow attr')
        self.parser.add_argument('--dk_mask', type=bool, default=False, help='use dontknow attr with dontknow mask')

        self.parser.add_argument('--use_gt_dbs', type=bool, default=False, help='use gt dbs')
        self.parser.add_argument('--use_retriever_for_gt', type=bool, default=False, help='use retriever for gt')

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='others/result/mwoz_gptke/RRG_v3_micro', help='models are saved here')
        self.parser.add_argument('--retriever_model_name', type=str, default='others/models/RRG/retriever_train_new_trunc_data_used_new_v0_seed-111_bert-base-uncased_ep-10_lr-5e-5_wd-0.01_maxlen-128_bs-32_ngpu-4_pln-128_tmp-0.05_hnw-0', help='path for retriever model')
        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=2, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                                 help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                                 help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                                 help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=111, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=2000,
                                 help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=40000,
                                 help='save model every <save_freq> steps during training')
        self.parser.add_argument('--start_eval_step', type=int, default=1,
                                 help='evaluate start step during training')
        self.parser.add_argument('--end_eval_step', type=int, default=32000,
                                 help='evaluate end step during training')
        self.parser.add_argument('--metric_record_file', type=str, default="metric_record.csv",
                                 help="file to write all metric")
        self.parser.add_argument('--dataset_name', type=str, default="mwoz_gptke", help="dataset name")
        self.parser.add_argument('--metric_version', type=str, default="new1", help="metric version")
        self.parser.add_argument('--model_select_metric', type=str, default="MICRO-F1", help="model select metric")

    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        with open(expr_dir / 'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt
