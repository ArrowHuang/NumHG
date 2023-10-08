import argparse, re, os
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

def cal_num_acc(predict_path, num_gt_path, num_path):
    pred_all, gt_all = [], []
    pred_copy, gt_copy = [], []
    pred_cal, gt_cal = [], []
    pattern = re.compile(r'\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+')

    with open(predict_path) as pred, open(num_gt_path) as target, open(num_path) as num_type:
        total_num, count_copy, count_cal = 0, 0, 0
        for (hyp, gt, num) in zip(pred, target, num_type):
            generated_num_list = pattern.findall(hyp)
            if(str(num).strip()=='0'): # Copy
                count_copy += 1
                if(str(gt).strip() in generated_num_list and len(generated_num_list)==1):
                    pred_copy.append(1)
                    pred_all.append(1)
                else:
                    pred_copy.append(0)
                    pred_all.append(0)
                gt_copy.append(1)
                gt_all.append(1)
            else:
                count_cal += 1
                if(str(gt).strip() in generated_num_list and len(generated_num_list)==1):
                    pred_cal.append(1)
                    pred_all.append(1)
                else:
                    pred_cal.append(0)
                    pred_all.append(0)
                gt_cal.append(1)
                gt_all.append(1)
            total_num += 1
        print("All Accuracy: %.6f, Copy Accuracy: %.6f, Cal Accuracy: %.6f"%(accuracy_score(gt_all, pred_all), accuracy_score(gt_copy, pred_copy), accuracy_score(gt_cal, pred_cal)))


def cal_rouge_score(target_path, predict_path):
    rouge1, rouge2, rougeLsum = 0, 0, 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))

    with open(predict_path) as pred, open(target_path) as target:
        total_num = 0
        for (hyp, ref) in zip(pred, target):
            hyp = hyp.strip()
            ref = ref.strip()
            hyp = process(hyp)
            ref = process(ref)
            score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
            rouge1 += score["rouge1"].fmeasure
            rouge2 += score["rouge2"].fmeasure
            rougeLsum += score["rougeLsum"].fmeasure
            total_num += 1
        rouge1 = rouge1 / total_num
        rouge2 = rouge2 / total_num
        rougeLsum = rougeLsum / total_num
        print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def cal_mover_score(target_path, predict_path):
    from moverscore_v2 import word_mover_score, get_idf_dict
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['MOVERSCORE_MODEL'] = "roberta-large"
    with open(predict_path) as pred, open(target_path) as target:
        total_num, sentence_score = 0, 0
        for (hyp, ref) in tqdm(zip(pred, target)):
            hyp = [hyp.strip()]
            ref = [ref.strip()]
            idf_dict_hyp = get_idf_dict(hyp)
            idf_dict_ref = get_idf_dict(ref)
            
            sentence_score += np.mean(word_mover_score(ref, hyp, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True))
            total_num += 1
        mover_score = sentence_score / total_num
        print("evaluation MoverScore: %.6f"%(mover_score))


def cal_bert_score(target_path, predict_path):
    os.system("bert-score -r {} -c {} --lang en --rescale_with_baseline".format(target_path, predict_path))


def main(args):
    print('Calculating Rouge Score......')
    cal_rouge_score(args.tgt_path, args.pre_path)

    print('\nCalculating Numeral Accuracy......')
    cal_num_acc(args.pre_path, args.num_gt_path, args.num_type_path)

    print('\nCalculating Moverscore......')
    cal_mover_score(args.tgt_path, args.pre_path)

    print('\nCalculating BERTScore......')
    cal_bert_score(args.tgt_path, args.pre_path)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--tgt_path", default="", type=str, help="target path")
    parser.add_argument("--pre_path", default="", type=str, help="predict path")
    parser.add_argument("--num_gt_path", default="", type=str, help="numerical ground truth path")
    parser.add_argument("--num_type_path", default="", type=str, help="type of each summary, 1:Reasoning, 0:Copy")
    args = parser.parse_args()
    main(args)