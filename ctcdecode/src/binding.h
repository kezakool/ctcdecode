#ifndef BINDING_H
#define BINDING_H

int paddle_beam_decode(void* decoder_input,
                       void* decoder,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_scores,
                       at::Tensor th_out_length);

void* get_decoder_input_with_hotwords(at::Tensor th_probs,
                                      at::Tensor th_seq_lens,
                                      void* hotword_scorer);

void* get_decoder_input(at::Tensor th_probs, at::Tensor th_seq_lens);

std::vector<std::vector<std::vector<double>>> get_vector_probs(at::Tensor& th_probs,
                                                               at::Tensor& th_seq_lens);

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
                         std::string log_level);

void create_lm_scorer(void* decoder,
                      double alpha,
                      double beta,
                      std::string lm_path,
                      std::string lm_type,
                      std::string lexicon_fst_path);

void* get_hotword_scorer(void* decoder,
                         std::vector<std::vector<std::string>> hotwords,
                         std::vector<float> hotword_weights);

void* paddle_get_decoder_state(void* decoder);

void paddle_release_resources(void* decoder);
void paddle_release_hotword_scorer(void* decoder);
void paddle_release_state(void* state);
void paddle_release_decoder_input(void* decoder_input);

int is_character_based(void* decoder);
size_t get_max_order(void* decoder);
size_t get_lexicon_size(void* decoder);
void reset_params(void* decoder, double alpha, double beta);

#endif // BINDING_H