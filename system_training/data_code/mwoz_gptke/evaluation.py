"""evaluation code"""
import os
import tempfile
import subprocess
import re
import json
import numpy as np
from src import util
import string


class Retrieve_Metric(object):
    def __init__(self, retrieve_score, data, db):
        self.retrieve_score = retrieve_score  # (num_example, num_db)
        self.data = data
        self.db = db

    def get_retrieve_entity(self, index, all_db, gt_db):
        retrieve_entity = []
        ref, choice = None, None
        for x in gt_db:
            if "ref" in x:
                ref = x["ref"]
            if "choice" in x:
                choice = x["choice"]
        for i in index:
            entity = all_db[i]
            if ref is not None:
                entity["ref"] = ref.replace("_", " ")
            if choice is not None:
                entity["choice"] = choice.replace("_", " ")
            retrieve_entity.append(entity)
        return retrieve_entity

    def get_match(self, gt_value, retrieve_entity):
        for e in retrieve_entity:
            for pred_value in e.values():
                if pred_value.replace(" ", "") == gt_value.replace(" ", ""):
                    return 1
        return 0

    def get_first_turn_name(self, data):
        pre_entity = (None, None)
        keep_index_list = []
        for i, d in enumerate(data):
            turn_kb_name = [entity["name"] for entity in d["kb"]]
            for x in d["gold_entities"]:
                if x in turn_kb_name and (x != pre_entity[0] or (x == pre_entity[0] and d["did"] != pre_entity[1])):
                    keep_index_list.append(i)
                    pre_entity = (x, d["did"])
                    break
        return keep_index_list

    def calc_recall(self, level="turn_level", top_k=7, first_turn_name=True):
        top_k_index = self.retrieve_score.sort(-1, True)[1][:,:top_k].tolist()  # (num_example, top_k_entity)
        total, match = 0, 0
        if first_turn_name is True:
            keep_idx_list = self.get_first_turn_name(self.data)
        for i in range(len(self.data)):
            if "<sys-api>" in self.data[i]["output_used"]:
                continue
            if first_turn_name is True and i not in keep_idx_list:
                continue  # only calculate the first time entity name appear in an dialog
            turn_data = self.data[i]
            turn_retrieve_index = top_k_index[i]
            turn_gt_values = [x.replace("_", " ") for x in turn_data["gold_entities"]]
            turn_retrieve_entity = self.get_retrieve_entity(turn_retrieve_index, self.db, turn_data["kb"])
            if level == "value_level":
                total += len(turn_gt_values)
                for v in turn_gt_values:
                    tmp = self.get_match(v, turn_retrieve_entity)
                    match += tmp
            elif level == "turn_level":
                total += 1
                curr_match = 0
                for v in turn_gt_values:
                    curr_match += self.get_match(v, turn_retrieve_entity)
                if curr_match == len(turn_gt_values):
                    match += 1
        recall = match / total
        return {f"RECALL@{top_k}_{level}": recall * 100}


class Metric_data1_new1(object):
    """
    BLEU:
    F1:
    """

    def __init__(self, data):
        self.data = data
        self.all_domain = ["attraction", "hotel", "restaurant"]
        self.dataset = "MultiWOZ"
        self.entities_path = "others/data/mwoz_gptke/data_raw/data_QTOD/entities.json"
        self.entities = self._load_entities(self.entities_path)

    def moses_multi_bleu(self, hypotheses, references, lowercase=False):
        """Calculate the bleu score for hypotheses and references
        using the MOSES ulti-bleu.perl script.
        Args:
        hypotheses: A numpy array of strings where each string is a single example.
        references: A numpy array of strings where each string is a single example.
        lowercase: If true, pass the "-lc" flag to the multi-bleu script
        Returns:
        The BLEU score as a float32 value.
        """

        if np.size(hypotheses) == 0:
            return np.float32(0.0)

        multi_bleu_path = "data_code/mwoz_gptke/multi-bleu.perl"
        os.chmod(multi_bleu_path, 0o755)

        # Dump hypotheses and references to tempfiles
        hypothesis_file = tempfile.NamedTemporaryFile()
        hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
        hypothesis_file.write(b"\n")
        hypothesis_file.flush()
        reference_file = tempfile.NamedTemporaryFile()
        reference_file.write("\n".join(references).encode("utf-8"))
        reference_file.write(b"\n")
        reference_file.flush()

        # Calculate BLEU using multi-bleu script
        with open(hypothesis_file.name, "r") as read_pred:
            bleu_cmd = [multi_bleu_path]
            if lowercase:
                bleu_cmd += ["-lc"]
            bleu_cmd += [reference_file.name]
            try:
                bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
                bleu_out = bleu_out.decode("utf-8")
                bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
                bleu_score = float(bleu_score)
            except subprocess.CalledProcessError as error:
                if error.output is not None:
                    print("multi-bleu.perl script returned non-zero exit code")
                    print(error.output)
                    bleu_score = np.float32(0.0)

        # Close temp files
        hypothesis_file.close()
        reference_file.close()
        return bleu_score

    def clean_gen_sentence(self, text):
        text = text.replace("<sys> ", "").replace("<sys-api> ", "").replace(" </s>", ""). \
            replace("<user>", "").replace("<api>", "").replace("<database>", "").replace("<sep_attributes>", "").strip()
        return text

    def preprocess_text(self, text):
        """Preprocess utterance and table value."""
        text = text.strip().replace("\t", " ").lower()
        for p in string.punctuation:
            text = text.replace(p, f" {p} ")
        text = " ".join(text.split())
        return text

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1, entity_precision, entity_recall = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1, entity_precision, entity_recall

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(val["address"])
                        entities.add(val["poi"])
                        entities.add(val["type"])
                    elif slot == "distance":
                        entities.add(f"{val} miles")
                    elif slot == "temperature":
                        entities.add(f"{val}f")
                    else:
                        entities.add(val)

            # add missing entities
            missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "]
            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove("hr")

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    entities.add(val)

        processed_entities = []
        for val in entities:
            processed_entities.append(val.lower())
        processed_entities.sort(key=lambda x: len(x), reverse=True)
        return processed_entities

    def _extract_entities(self, response):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for entity in self.entities:
            if self.dataset == "MultiWOZ":
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append(entity)
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity)

        return extracted_entities

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return f1, precision, recall

    def baseline_reader_metric(self):
        """
        in this version, data = [ans, output_used, gold_entities, kb, type]
        :return:
        """
        preds = []
        refs = []
        for i in range(0, len(self.data)):
            if "<sys-api>" in self.data[i][1]:
                continue  # dont calc these
            pred = self.data[i][0]
            # remove special tokens and add space for pred sentence
            pred = self.preprocess_text(self.clean_gen_sentence(pred))
            # remove special tokens for gold sentence, it has already been tokenized
            gold = self.data[i][1]
            gold = self.clean_gen_sentence(gold)
            preds.append(pred)
            refs.append(gold)

        BLEU = self.moses_multi_bleu(np.array(preds), np.array(refs), lowercase=True)
        F1, P, R = self.evaluate(preds, refs)

        return {
            "BLEU": BLEU,
            "MICRO-F1": F1 * 100,
            "P": P * 100,
            "R": R * 100,
        }
