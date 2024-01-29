#ifndef DECODER_H
#define DECODER_H

#include "ctc_beam_search_decoder.h"
#include "decoder_options.h"
#include "decoder_utils.h"
#include "hotword_scorer.h"
#include "logger.h"
#include "output.h"
#include "scorer.h"

class DecoderInput {
public:
    /* Initialize DecoderInput for CTC beam decoding
     *
     * Parameters:
     *     probs: Acoustic model output probabilities shape (batch x num_timesteps x num_labels)
     *     hotword_scorer: HotwordScorer object to boost the given words scores( default = nullptr )
     */
    DecoderInput(const std::vector<std::vector<std::vector<double>>>& probs,
                 HotwordScorer* hotword_scorer = nullptr)
        : probs(probs)
        , hotword_scorer(hotword_scorer)
    {
    }

    ~DecoderInput() { hotword_scorer = nullptr; }
    std::vector<std::vector<std::vector<double>>> probs;
    HotwordScorer* hotword_scorer = nullptr;
};

class Decoder {
public:
    /* Initialize Decoder for CTC beam decoding
     *
     * Parameters:
     *     options: Configuration for ctc beam decoding
     *     log_filepath: Path to log file
     *     loglevel: Logger level
     */
    Decoder(DecoderOptions& options,
            std::string log_filepath = LOG_FILENAME,
            std::string loglevel = "none")
        : options(options)
        , log_filepath(log_filepath)
        , log_level(loglevel)
    {
        Init();
    }

    ~Decoder();

    static Scorer* create_lm_scorer(double alpha,
                                    double beta,
                                    const std::string& lm_path,
                                    const std::vector<std::string>& new_vocab,
                                    const std::string& lm_type,
                                    const std::string& lexicon_fst_path);

    DecoderOutput decode(DecoderInput* decoder_input);

    /**
     * @brief Initialize logger with the given level and filename
     *
     * @param level, logging level
     * @param filepath, path to the file to log statements
     */
    static void init_logger(const std::string& level, const std::string& filepath)
    {
        if (!filepath.empty()) {
            logger.SetFile(filepath, true);
        }
        logger.SetLevelString(level);
    }

    // set lm scorer
    void set_lm_scorer(Scorer* scorer) { _lm_scorer = scorer; }

    // return lm scorer object
    Scorer* get_lm_scorer() { return _lm_scorer; }

    // return the max order
    size_t get_max_order() { return _lm_scorer->get_max_order(); }

    // return the dictionary size of language model
    size_t get_lexicon_size() { return _lm_scorer->get_lexicon_size(); }

    // retrun true if the language model is character based
    bool is_character_based() { return _lm_scorer->is_character_based(); }

    // retrun true if the language model is bpe based
    bool is_bpe_based() { return _lm_scorer->is_bpe_based(); }

    // reset params alpha & beta
    void reset_params(float alpha, float beta) { _lm_scorer->reset_params(alpha, beta); }

    // Initializes the decoder class
    void Init();

    DecoderOptions options;
    static Logger logger;
    std::string log_level = "none";
    std::string log_filepath = LOG_FILENAME;

private:
    Scorer* _lm_scorer = nullptr;
};

#endif // DECODER_H
