#include "decoder.h"

Logger Decoder::logger;

/**
 * @brief Destructs DecoderOptions, LM scorer and the threadpool objects
 */
Decoder::~Decoder()
{
    if (_lm_scorer != nullptr) {
        delete _lm_scorer;
    }
}

/**
 * @brief Initialize Decoder by initializing the logger and thread pool
 *
 */

void Decoder::Init()
{
    // Initialize logger
    init_logger(this->log_level, this->log_filepath);

    logger.Log(LogLevel::INFO,
               "Decoder options provided are - cutoff_top_n: ",
               this->options.cutoff_top_n,
               ", cutoff_prob: ",
               this->options.cutoff_prob,
               ", beam_width: ",
               this->options.beam_width,
               ", num_processes: ",
               this->options.num_processes,
               ", blank_id: ",
               this->options.blank_id,
               ", log_probs_input: ",
               this->options.log_probs_input,
               ", is_bpe_based: ",
               this->options.is_bpe_based,
               ", log_filepath",
               this->log_filepath,
               ", log_level",
               this->log_level);
}

/**
 * @brief Static method to construct LM scorer object given lm path
 *
 * @param alpha, Weighting associated with the LMs probabilities.
                A weight of 0 means the LM has no effect.
 * @param beta, Weight associated with the number of words within our beam.
 * @param lm_path, The path to your external KenLM language model(LM)
 * @param new_vocab, A vector of vocabulary (labels).
 * @param lm_type, The type of LM used. Supported types: character, bpe, word
 * @param lexicon_fst_path, The path to your lexicon fst file.
 * @return scorer, External LM scorer object
 */
Scorer* Decoder::create_lm_scorer(double alpha,
                                  double beta,
                                  const std::string& lm_path,
                                  const std::vector<std::string>& new_vocab,
                                  const std::string& lm_type,
                                  const std::string& lexicon_fst_path)
{
    Scorer* scorer = new Scorer(alpha, beta, lm_path, new_vocab, lm_type, lexicon_fst_path);
    return scorer;
}

/**
 * @brief Method to decode batched decoder input probabilities
 *
 * @param decoder_input, DecoderInput object containing acoustic model output probabilities shape
 * and hotwords if provided
 * @return output, DecoderOutput object that contains batched results
 */
DecoderOutput Decoder::decode(DecoderInput* decoder_input)
{

    BatchResults batch_results = ctc_beam_search_decoder_batch(
        decoder_input->probs, options, _lm_scorer, decoder_input->hotword_scorer);

    DecoderOutput output { batch_results };

    return output;
}
