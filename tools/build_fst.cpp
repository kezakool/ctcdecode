#include "build_fst.h"

/**
 * @brief This method parses the labels file and returns a vector of labels
 *
 * @param vocab_path, Path to the file containing bpe labels. Each line in the file should contain a
 *                      label.
 * @return labels, A Vector of labels
 */
std::vector<std::string> get_bpe_vocab(const std::string vocab_path)
{

    std::ifstream inputFile(vocab_path);
    std::vector<std::string> labels;
    if (!inputFile.is_open()) {
        Decoder::logger.Log(LogLevel::ERROR, "Failed to open the the vocabulary file.");
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        labels.push_back(line);
    }

    Decoder::logger.Log(LogLevel::INFO, "Size of labels: ", labels.size());

    inputFile.close();

    return labels;
}

/**
 * @brief Returns character map for the given labels
 *
 * @param labels, A Vector of labels
 * @return char_map, A map of characters/tokens to their corresponding integer ids starting from 1
 */
std::unordered_map<std::string, int> get_char_map(const std::vector<std::string>& labels)
{

    std::unordered_map<std::string, int> char_map;
    for (int i = 0; i < labels.size(); ++i) {
        char_map[labels[i]] = i + 1;
    }

    return char_map;
}

/**
 * @brief This method reads the FST from the given file
 *
 * @param input_path, The path to the file containing the FST
 * @return dictionary, The FST read from the file
 */
fst::StdVectorFst* read_fst(const std::string input_path)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    // Read the FST from the file
    fst::StdVectorFst* dict = fst::StdVectorFst::Read(input_path);
    if (!dict) {
        Decoder::logger.Log(LogLevel::ERROR, "Failed to read FST from file: ", input_path);
        exit(EXIT_FAILURE);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

    Decoder::logger.Log(LogLevel::INFO, "Time taken for reading the FST: ", duration, " seconds");
    Decoder::logger.Log(LogLevel::INFO, "Number of states in FST are ", dict->NumStates());

    return dict;
}

/**
 * @brief This method writes the given FST to the given file
 *
 * @param dictionary, The FST to be written
 * @param output_path, The path to the file to which the FST is to be written
 */
void write_fst(fst::StdVectorFst* dictionary, const std::string output_path)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    dictionary->Write(output_path);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    Decoder::logger.Log(
        LogLevel::INFO, "Time taken for writing the FST to file: ", duration, " seconds");
}

/**
 * @brief This method optimizes the FST by removing the epsilon transitions and minimizing it
 *
 * @param dictionary, The FST to be optimized
 */
void optimize_fst(fst::StdVectorFst* dictionary)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    fst::RmEpsilon(dictionary);
    fst::Determinize(*dictionary, dictionary);
    fst::Minimize(dictionary);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    Decoder::logger.Log(LogLevel::INFO, "Time taken to optimize FST: ", duration, " seconds");
}

/**
 * @brief This method adds a word to the given FST
 *
 * @param characters, A vector of characters/tokens in the word
 * @param char_map, A map of characters/tokens to their corresponding integer ids
 * @param dictionary, The FST to which the word is to be added
 * @param current_state, The current state in the FST from which the word is to be added
 * @return is_word_added, Returns true if the word is added newly, else false
 */
bool add_word_to_fst(const std::vector<std::string>& characters,
                     const std::unordered_map<std::string, int>& char_map,
                     fst::StdVectorFst* dictionary,
                     fst::StdVectorFst::StateId current_state)
{
    bool is_word_added = false;
    for (auto& c : characters) {
        // Find the symbol ID for the character
        auto int_c = char_map.find(c);
        int symbol_id;
        if (int_c == char_map.end()) {
            Decoder::logger.Log(LogLevel::ERROR, "Character/token not found: ", c);
            exit(EXIT_FAILURE);
        } else {
            symbol_id = int_c->second;
        }

        // Check if the arc already exists
        bool arc_exists = false;

        for (fst::ArcIterator<fst::StdVectorFst> aiter(*dictionary, current_state); !aiter.Done();
             aiter.Next()) {
            const fst::StdArc& arc = aiter.Value();
            if (arc.ilabel == symbol_id) {
                arc_exists = true;
                current_state = arc.nextstate;
                break;
            }
        }
        if (!arc_exists) {
            // Add a new arc
            fst::StdVectorFst::StateId next_state = dictionary->AddState();
            dictionary->AddArc(current_state, fst::StdArc(symbol_id, symbol_id, 0, next_state));
            current_state = next_state;
            is_word_added = true;
        }
    }

    // Set the current state as final
    dictionary->SetFinal(current_state, fst::StdArc::Weight::One());

    return is_word_added;
}

/**
 * @brief This method parses the lexicon file and adds the words in it to the given FST
 *
 * @param lexicon_path, The path to the lexicon file
 * @param dictionary, The FST to which the words are to be added
 * @param char_map, A map of characters/tokens to their corresponding integer ids
 * @param freq_threshold, Frequency threshold, words having frequency greater than or equal to this
 *              threshold will be considered
 * @return word_count, The number of words added to the FST
 */
int parse_lexicon_and_add_to_fst(const std::string& lexicon_path,
                                 fst::StdVectorFst* dictionary,
                                 const std::unordered_map<std::string, int>& char_map,
                                 const int freq_threshold)
{
    int word_count = 0;
    std::ifstream file(lexicon_path);
    if (!file) {
        Decoder::logger.Log(LogLevel::ERROR, "Error opening lexicon file: ", lexicon_path);
        exit(EXIT_FAILURE);
    }

    Decoder::logger.Log(LogLevel::INFO, "Loading words from unigrams path provided");

    std::string line;
    fst::StdVectorFst::StateId start_state;

    if (dictionary->NumStates() == 0) {
        Decoder::logger.Log(LogLevel::INFO, "Setting dictionary start state");
        start_state = dictionary->AddState();
        assert(start_state == 0);
        dictionary->SetStart(start_state);
    }
    start_state = dictionary->Start();
    int count = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> characters;
        count += 1;
        int i = 0;
        bool skip = false;
        while (std::getline(iss, token, ' ')) {
            if (i == 0 && freq_threshold != -1) {

                int freq = std::stoi(token);
                if (freq < freq_threshold) {
                    skip = true;
                }
            }

            if (i != 0 && i != 1 && !skip) {
                characters.push_back(token);
            }
            ++i;
        }

        if (count % 100000 == 0) {
            Decoder::logger.Log(LogLevel::INFO, "Processed ", count, " records");
        }

        if (characters.size() > 0 && !skip) {
            word_count += add_word_to_fst(characters, char_map, dictionary, start_state);
        }
    }
    Decoder::logger.Log(
        LogLevel::INFO, "Constructed the fst for the given lexicon path: ", lexicon_path);
    Decoder::logger.Log(LogLevel::INFO, "Number of words in the given path are ", count);

    return word_count;
}

/**
 * @brief This method constructs the FST from the given lexicon files
 *
 * @param vocab_path, Path to the file containing labels
 * @param lexicon_paths, A vector of paths to lexicon files
 * @param fst_path, The path to the FST file. If empty, a new FST will be created or
 *                  else the words will be added on top of this FST
 * @param output_path, The path to the file to which the FST is to be written
 * @param freq_threshold, Frequency threshold, words having frequency greater than or equal to this
 *              threshold will be considered ( Default = -1 , i.e all are considered in this case)
 * @param optimize, If true, the FST will be optimized ( Default = true, two output files will be
 * generated in this case, one is optimized and other is unoptimized  )
 */
void construct_fst(const std::string vocab_path,
                   const std::vector<std::string>& lexicon_paths,
                   const std::string fst_path,
                   std::string output_path,
                   const int freq_threshold = -1,
                   bool optimize = true)
{
    // Load vocabulary
    std::vector<std::string> labels = get_bpe_vocab(vocab_path);
    // get character map
    std::unordered_map<std::string, int> char_map = get_char_map(labels);

    fst::StdVectorFst* dictionary;

    // Load the FST from the given file
    if (!fst_path.empty()) {
        Decoder::logger.Log(LogLevel::INFO, "Reading the fst from ", fst_path);
        dictionary = read_fst(fst_path);
    } else {
        dictionary = new fst::StdVectorFst;
    }

    int dict_size = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Parse each lexicon file and add the words in it to the FST
    for (auto lexicon_path : lexicon_paths) {
        int word_count
            = parse_lexicon_and_add_to_fst(lexicon_path, dictionary, char_map, freq_threshold);
        Decoder::logger.Log(
            LogLevel::INFO, "Number of words added to the dictionary are ", word_count);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    Decoder::logger.Log(LogLevel::INFO, "Time taken to create FST: ", duration, " seconds");

    // Write the FST to the given output file
    // output file with `.opt` extension will be created if optimize is true
    write_fst(dictionary, output_path);
    if (optimize) {
        optimize_fst(dictionary);
        write_fst(dictionary, output_path + ".opt");
    }

    // Number of states in FST
    Decoder::logger.Log(LogLevel::INFO, "Number of states in FST are ", dictionary->NumStates());

    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    Decoder::logger.Log(LogLevel::INFO, "Total time taken: ", duration, " seconds");

    // delete the FST
    delete dictionary;
}