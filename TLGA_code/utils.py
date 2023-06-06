#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
import pyrouge
import logging
import tensorflow as tf

def print_results(article, abstract, decoded_output):
    print ("")
    print('ARTICLE:  %s', article)
    print('REFERENCE SUMMARY: %s', abstract)
    print('GENERATED SUMMARY: %s', decoded_output)
    print( "")


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1","2","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..."%(results_file))
    with open(results_file, "w") as f:
        f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return running_avg_loss

#这里改成输出也是id
def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir, batch):
    # decoded_sents = []
    reference_id = batch.dec_batch[0]
    reference_id = list(filter(lambda x: x != 1, reference_id))
    reference_id = list(filter(lambda x: x != 2, reference_id))

    decoded_words = decoded_words[0:len(decoded_words) - 1]
    # while len(decoded_words) > 0:
    #     try:
    #         fst_period_idx = decoded_words.index(".")
    #     except ValueError:
    #         fst_period_idx = len(decoded_words)
    #     sent = decoded_words[:fst_period_idx + 1]
    #     decoded_words = decoded_words[fst_period_idx + 1:]
    #     decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(str(w)) for w in decoded_words]
    reference_id = [make_html_safe(str(w)) for w in reference_id]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_id):
            f.write(sent) if idx == len(reference_id) - 1 else f.write(sent + " ")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + " ")


if __name__ == '__main__':
    dec_rootdir = "/data1/lhz/pg_network_torch/transformer_log/decode_model_219776_20210602_141532/"
    ref_dir, dec_dir = os.path.join(dec_rootdir,"rouge_ref"), os.path.join(dec_rootdir, "rouge_dec_dir")
    results_dict = rouge_eval(ref_dir, dec_dir) 
    print("results_dict:", results_dict)
    rouge_log(results_dict, "/data1/lhz/pg_network_torch/transformer_log/decode_model_219776_20210602_141532/")
