#ifndef OUTPUT_H_
#define OUTPUT_H_

/* Struct for the beam search output, containing the tokens based on the vocabulary indices, and the
 * timesteps for each token in the beam search output
 */
struct Output {
    std::vector<int> tokens, timesteps;
};

using ScoredOuputEntries
    = std::vector<std::pair<double, Output>>; // for each prefix, we have a pair of score and output
using BatchResults = std::vector<ScoredOuputEntries>; // results for each sample in the batch
struct DecoderOutput {
    BatchResults results;
};

#endif // OUTPUT_H_
