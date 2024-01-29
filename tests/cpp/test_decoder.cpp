#include <gtest/gtest.h>
#include <torch/torch.h>

#include "binding.h"
#include "build_fst.h"
#include "decoder.h"

std::vector<char> get_the_bytes(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

torch::Tensor load_test_logits()
{
    std::vector<char> f = get_the_bytes(std::string(TEST_FIXTURES_DIR) + "/logits.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor logits = x.toTensor().cpu();

    return logits;
}

// load_labels
std::vector<std::string> load_labels()
{
    std::string label_path = std::string(TEST_FIXTURES_DIR) + "/vocab.txt";

    std::vector<std::string> labels = get_bpe_vocab(label_path);

    return labels;
}

TEST(DecoderTest, TestDecoder)
{

    std::vector<std::string> labels = load_labels();

    // create decoder object
    DecoderOptions decoder_opts(labels);
    Decoder* decoder = new Decoder(decoder_opts);

    torch::Tensor logits = load_test_logits();

    EXPECT_EQ(logits.size(0), 1);
    EXPECT_EQ(logits.size(1), 86);
    EXPECT_EQ(logits.size(2), 512);

    int batch_size = logits.size(0);
    int max_seq_len = logits.size(1);

    torch::Tensor seq_lens = torch::full({ batch_size }, max_seq_len, torch::kInt);

    // convert tensor to vector
    std::vector<std::vector<std::vector<double>>> probs = get_vector_probs(logits, seq_lens);

    // create decoder input and call Decoder `decode` method
    DecoderInput* decoder_input = new DecoderInput(probs);
    DecoderOutput output = decoder->decode(decoder_input);

    EXPECT_EQ(output.results.size(), batch_size);
    std::cout << "output.results.size(): " << output.results.size() << std::endl;
    std::cout << "output.results[0].size(): " << output.results[0].size() << std::endl;
    // std::cout << output.results[0] << std::endl;
    EXPECT_EQ(output.results[0].size(), 100); // default beam size
}

TEST(DecoderTest, TestWithBindingMethods)
{

    // create decoder object
    std::vector<std::string> labels { "'", " ", "a", "b", "c", "d", "_" };
    void* decoder = paddle_get_decoder(
        labels, 40, 1.0, 100, 4, 6, true, true, -5, '#', "sample.log", "debug");

    // lm scorer
    std::string lm_path = std::string(TEST_FIXTURES_DIR) + "/test.arpa";
    create_lm_scorer(decoder, 0.5, 1.0, lm_path, "word", "");

    int i = 0;
    while (i < 10) {
        // create decoder input
        torch::Tensor logits = torch::randn({ 20, 269, labels.size() }, torch::kFloat).cpu();
        int batch_size = logits.size(0);
        int max_seq_len = logits.size(1);
        int beam_size = 100;
        torch::Tensor seq_lens = torch::full({ batch_size }, max_seq_len, torch::kInt);
        torch::Tensor out = torch::zeros({ batch_size, beam_size, max_seq_len }, torch::kInt);
        torch::Tensor timesteps = torch::zeros({ batch_size, beam_size, max_seq_len }, torch::kInt);
        torch::Tensor scores = torch::zeros({ batch_size, beam_size }, torch::kFloat);
        torch::Tensor out_seq_len = torch::zeros({ batch_size, beam_size }, torch::kInt);
        void* decoder_input = get_decoder_input(logits, seq_lens);

        // decode call
        paddle_beam_decode(decoder_input, decoder, out, timesteps, scores, out_seq_len);

        torch::IntArrayRef shape = out.sizes();
        EXPECT_EQ(shape[0], batch_size);
        EXPECT_EQ(shape[1], beam_size);
        EXPECT_EQ(shape[2], max_seq_len);

        i++;
        if (i % 10 == 0) {
            std::cout << "iteration: " << i << std::endl;
        }
        delete decoder_input;
    }

    paddle_release_resources(decoder);
}