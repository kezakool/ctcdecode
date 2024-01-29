#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <map>

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "[UNK]";
const std::string END_TOKEN = "</s>";

const float NUM_FLT_INF = std::numeric_limits<float>::max();
const float NUM_FLT_MIN = std::numeric_limits<float>::min();
const int NUM_INT_INF = std::numeric_limits<int>::max();
const float NUM_FLT_LOGE = 0.4342944819;

const std::string LOG_FILENAME = "CTCBeamDecoderLogger.txt";

enum TokenizerType { CHAR = 0, BPE = 1, WORD = 2 };

static std::map<std::string, TokenizerType> StringToTokenizerType
    = { { "character", TokenizerType::CHAR },
        { "bpe", TokenizerType::BPE },
        { "word", TokenizerType::WORD } };

#endif // CONSTANTS_H
