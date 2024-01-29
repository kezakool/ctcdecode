#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "boost/shared_ptr.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "binding.h"
#include "decoder.h"
#include "utf8.h"

namespace py = pybind11;

template <typename T>
inline std::vector<T> py_list_to_std_vector(const boost::python::object& iterable)
{
    return std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                          boost::python::stl_input_iterator<T>());
}

template <class T>
inline boost::python::list std_vector_to_py_list(std::vector<T> vector)
{
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

/**
 * @brief Method to do beam decoding with Decoder object given the decoder input
 *
 * @param decoder_input, DecoderInput object containing probabilities and hotwords if provided
 * @param decoder, Decoder object to run decoding with the decoder_input and options
 * @param th_output, Output tensor to store the decoded token sequence
 * @param th_timesteps, Tensor to store the token timesteps
 * @param th_scores, Tensor to store the scores for top beams
 * @param th_out_length, Tensor to store the output tokens sizes
 * @return true
 */
int paddle_beam_decode(void* decoder_input,
                       void* decoder,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_scores,
                       at::Tensor th_out_length)
{

    DecoderInput* ctc_decoder_input = static_cast<DecoderInput*>(decoder_input);
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);

    DecoderOutput output = ctc_decoder->decode(ctc_decoder_input);

    BatchResults& batch_results = output.results;

    auto outputs_accessor = th_output.accessor<int, 3>();
    auto timesteps_accessor = th_timesteps.accessor<int, 3>();
    auto scores_accessor = th_scores.accessor<float, 2>();
    auto out_length_accessor = th_out_length.accessor<int, 2>();

    for (int b = 0; b < batch_results.size(); ++b) {
        ScoredOuputEntries results = batch_results[b];
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;
            std::vector<int> output_timesteps = output.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t) {
                outputs_accessor[b][p][t] = output_tokens[t]; // fill output tokens
                timesteps_accessor[b][p][t] = output_timesteps[t];
            }
            scores_accessor[b][p] = n_path_result.first;
            out_length_accessor[b][p] = output_tokens.size();
        }
    }
    return 1;
}

/**
 * @brief Method to convert the given tensor probabilities to vector
 *
 * @param th_probs, Acoustic model tensor probabilites
 * @param th_seq_lens, Length of the probability input to be considered for each in the batch
 * @return inputs, vectorized probabilities
 */
std::vector<std::vector<std::vector<double>>> get_vector_probs(at::Tensor& th_probs,
                                                               at::Tensor& th_seq_lens)
{
    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs(batch_size);
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();

    for (int b = 0; b < batch_size; ++b) {
        // avoid a crash by ensuring that an
        // erroneous seq_len doesn't have us try to access memory
        // we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time);
        std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t) {
            for (int n = 0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs[b] = temp;
    }

    return inputs;
}

/**
 * @brief Method to create DecoderInput object with the given probabilities and hotwords and return
 * it
 *
 * @param th_probs, Acoustic model tensor probabilites
 * @param th_seq_lens, Length of the probability input to be considered for each in the batch
 * @param hotword_scorer, Hotword Scorer object
 * @return decoder_input, DecoderInput object casted to void*
 */
void* get_decoder_input_with_hotwords(at::Tensor th_probs,
                                      at::Tensor th_seq_lens,
                                      void* hotword_scorer)
{
    std::vector<std::vector<std::vector<double>>> inputs = get_vector_probs(th_probs, th_seq_lens);

    HotwordScorer* ext_hotword_scorer = nullptr;
    if (hotword_scorer != nullptr) {
        ext_hotword_scorer = static_cast<HotwordScorer*>(hotword_scorer);
    }

    DecoderInput* decoder_input = new DecoderInput(inputs, ext_hotword_scorer);
    return static_cast<void*>(decoder_input);
}

/**
 * @brief Method to create DecoderInput object with the given probabilities and return it
 *
 * @param th_probs, Acoustic model tensor probabilites
 * @param th_seq_lens, Length of the probability input to be considered for each in the batch
 * @return decoder_input, DecoderInput object casted to void*
 */
void* get_decoder_input(at::Tensor th_probs, at::Tensor th_seq_lens)
{
    return get_decoder_input_with_hotwords(th_probs, th_seq_lens, nullptr);
}

/**
 * @brief Method to create Decoder object
 *
 * @param vocab,A vector of vocabulary (labels).
 * @param cutoff_top_n,Cutoff number in pruning. Only the top cutoff_top_n characters
        with the highest probability in the vocab will be used in beam search.
 * @param cutoff_prob,Cutoff probability in pruning. 1.0 means no pruning.
 * @param beam_width,This controls how broad the beam search is. Higher values are more
        likely to find top beams, but they also will make your beam search exponentially slower.
 * @param num_processes, Parallelize the batch using num_processes workers.
 * @param blank_id,Index of the CTC blank token used when training your model.
 * @param log_probs_input,False if the model has passed through a softmax and output
        probabilities sum to 1.
 * @param is_bpe_based,True if the labels contains bpe tokens else False
 * @param unk_score,Score to give to an unknown token.
 * @param token_separator,Separator between tokens if the labels are bpe based.
 * @param log_filepath,Path to the file to log statements
 * @param log_level,Loglevel for the logger, if level is "none", nothing will be written to the
        output file
 */
void* paddle_get_decoder(std::vector<std::string> vocab,
                         size_t cutoff_top_n,
                         double cutoff_prob,
                         size_t beam_width,
                         size_t num_processes,
                         size_t blank_id,
                         bool log_probs_input,
                         bool is_bpe_based,
                         float unk_score,
                         char token_separator,
                         std::string log_filepath,
                         std::string log_level)
{

    VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");

    // Create Decoder Options
    DecoderOptions decoder_options(vocab,
                                   cutoff_top_n,
                                   cutoff_prob,
                                   beam_width,
                                   num_processes,
                                   blank_id,
                                   log_probs_input,
                                   is_bpe_based,
                                   unk_score,
                                   token_separator);

    // Create Decoder
    Decoder* decoder = new Decoder(decoder_options, log_filepath, log_level);
    return static_cast<void*>(decoder);
}

/**
 * @brief Method to initialize the language model scorer of the Decoder class
 *
 * @param decoder, Decoder object
 * @param alpha, Weighting associated with the LMs probabilities.
                A weight of 0 means the LM has no effect.
 * @param beta, Weight associated with the number of words within our beam.
 * @param lm_path, The path to your external KenLM language model(LM)
 * @param lm_type, The type of LM used. Options are "character", "bpe" and "word"
 * @param lexicon_fst_path, The path to external lexicon fst file
 */
void create_lm_scorer(void* decoder,
                      double alpha,
                      double beta,
                      std::string lm_path,
                      std::string lm_type,
                      std::string lexicon_fst_path)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);
    Scorer* scorer = Decoder::create_lm_scorer(
        alpha, beta, lm_path, ctc_decoder->options.vocab, lm_type, lexicon_fst_path);
    ctc_decoder->set_lm_scorer(scorer);
}

/**
 * @brief Method to initialize the hotword scorer using the decoder options present in Decoder
 * object
 *
 * @param decoder, Decoder object
 * @param hotwords, list of tokenized hotwords
 * @param hotword_weights, Weight list for the above hotwords
 * @return scorer, HotwordScorer object casted to void*
 */
void* get_hotword_scorer(void* decoder,
                         std::vector<std::vector<std::string>> hotwords,
                         std::vector<float> hotword_weights)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);
    HotwordScorer* scorer = new HotwordScorer(ctc_decoder->options.vocab,
                                              hotwords,
                                              hotword_weights,
                                              ctc_decoder->options.token_separator,
                                              ctc_decoder->options.is_bpe_based);
    return static_cast<void*>(scorer);
}

std::pair<torch::Tensor, torch::Tensor>
beam_decode_with_given_state(at::Tensor th_probs,
                             at::Tensor th_seq_lens,
                             size_t num_processes,
                             std::vector<void*>& states,
                             const std::vector<bool>& is_eos_s,
                             at::Tensor th_scores,
                             at::Tensor th_out_length)
{
    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();

    for (int b = 0; b < batch_size; ++b) {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory
        // we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time);
        std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t) {
            for (int n = 0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    BatchResults batch_results
        = ctc_beam_search_decoder_batch_with_states(inputs, num_processes, states, is_eos_s);

    int max_result_size = 0;
    int max_output_tokens_size = 0;
    for (int b = 0; b < batch_results.size(); ++b) {
        ScoredOuputEntries results = batch_results[b];
        if (batch_results[b].size() > max_result_size) {
            max_result_size = batch_results[b].size();
        }
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;

            if (output_tokens.size() > max_output_tokens_size) {
                max_output_tokens_size = output_tokens.size();
            }
        }
    }

    torch::Tensor output_tokens_tensor
        = torch::randint(1, { batch_results.size(), max_result_size, max_output_tokens_size });
    torch::Tensor output_timesteps_tensor
        = torch::randint(1, { batch_results.size(), max_result_size, max_output_tokens_size });

    auto scores_accessor = th_scores.accessor<float, 2>();
    auto out_length_accessor = th_out_length.accessor<int, 2>();

    for (int b = 0; b < batch_results.size(); ++b) {
        ScoredOuputEntries results = batch_results[b];
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;
            std::vector<int> output_timesteps = output.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t) {
                output_tokens_tensor[b][p][t] = output_tokens[t]; // fill output tokens
                output_timesteps_tensor[b][p][t] = output_timesteps[t];
            }
            scores_accessor[b][p] = n_path_result.first;
            out_length_accessor[b][p] = output_tokens.size();
        }
    }

    return { output_tokens_tensor, output_timesteps_tensor };
}

std::pair<torch::Tensor, torch::Tensor>
paddle_beam_decode_with_given_state(at::Tensor th_probs,
                                    at::Tensor th_seq_lens,
                                    size_t num_processes,
                                    std::vector<void*> states,
                                    std::vector<bool> is_eos_s,
                                    at::Tensor th_scores,
                                    at::Tensor th_out_length)
{

    return beam_decode_with_given_state(
        th_probs, th_seq_lens, num_processes, states, is_eos_s, th_scores, th_out_length);
}

void* paddle_get_decoder_state(void* decoder)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);

    DecoderState* state
        = new DecoderState(ctc_decoder->options, ctc_decoder->get_lm_scorer(), nullptr);
    return static_cast<void*>(state);
}

void paddle_release_state(void* state) { delete static_cast<DecoderState*>(state); }

/**
 * @brief Method to delete Decoder object
 *
 * @param decoder, Pointer to Decoder object
 */
void paddle_release_resources(void* decoder) { delete static_cast<Decoder*>(decoder); }

/**
 * @brief Method to delete DecoderInput object
 *
 * @param decoder_input, Pointer to DecoderInput object
 */
void paddle_release_decoder_input(void* decoder_input)
{
    delete static_cast<DecoderInput*>(decoder_input);
    decoder_input = nullptr;
}

/**
 * @brief Method to delete hotword scorer object
 *
 * @param scorer, Pointer to HotwordScorer object
 */
void paddle_release_hotword_scorer(void* scorer)
{
    delete static_cast<HotwordScorer*>(scorer);
    scorer = nullptr;
}

int is_character_based(void* decoder)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);
    return ctc_decoder->is_character_based();
}

size_t get_max_order(void* decoder)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);
    return ctc_decoder->get_max_order();
}

size_t get_lexicon_size(void* scorer)
{
    Scorer* ext_scorer = static_cast<Scorer*>(scorer);
    return ext_scorer->get_lexicon_size();
}

void reset_params(void* decoder, double alpha, double beta)
{
    Decoder* ctc_decoder = static_cast<Decoder*>(decoder);
    ctc_decoder->reset_params(alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("paddle_beam_decode", &paddle_beam_decode, "paddle_beam_decode");
    m.def("get_decoder_input_with_hotwords",
          &get_decoder_input_with_hotwords,
          "get_decoder_input_with_hotwords",
          py::arg("th_probs"),
          py::arg("th_seq_lens"),
          py::arg("hotword_scorer").none(true));
    m.def("get_decoder_input", &get_decoder_input, "get_decoder_input");
    m.def("paddle_get_decoder", &paddle_get_decoder, "paddle_get_decoder");
    m.def("create_lm_scorer", &create_lm_scorer, "create_lm_scorer");
    m.def("get_hotword_scorer", &get_hotword_scorer, "get_hotword_scorer");
    m.def("paddle_release_resources", &paddle_release_resources, "paddle_release_resources");
    m.def("paddle_release_decoder_input",
          &paddle_release_decoder_input,
          "paddle_release_decoder_input");
    m.def("paddle_release_hotword_scorer",
          &paddle_release_hotword_scorer,
          "paddle_release_hotword_scorer");
    m.def("is_character_based", &is_character_based, "is_character_based");
    m.def("get_max_order", &get_max_order, "get_max_order");
    m.def("get_lexicon_size", &get_lexicon_size, "get_max_order");
    m.def("reset_params", &reset_params, "reset_params");
    m.def("paddle_get_decoder_state", &paddle_get_decoder_state, "paddle_get_decoder_state");
    m.def("paddle_beam_decode_with_given_state",
          &paddle_beam_decode_with_given_state,
          "paddle_beam_decode_with_given_state");
    m.def("paddle_release_state", &paddle_release_state, "paddle_release_state");
    // paddle_beam_decode_with_given_state
}
