#pragma once
#include <atomic>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include "constants.h"

enum LogLevel { NONE = -1, ERROR = 0, INFO = 1, DEBUG = 2 };

static std::map<LogLevel, std::string> Level = { { LogLevel::INFO, "[INFO]" },
                                                 { LogLevel::DEBUG, "[DEBUG]" },
                                                 { LogLevel::ERROR, "[ERROR]" },
                                                 { LogLevel::NONE, "[NONE]" } };

static std::map<std::string, LogLevel> StringToLogLevel = { { "none", LogLevel::NONE },
                                                            { "info", LogLevel::INFO },
                                                            { "debug", LogLevel::DEBUG },
                                                            { "error", LogLevel::ERROR } };

class Logger {
private:
    std::atomic<bool> rotateFile;
    std::atomic<LogLevel> level;
    std::string logFileName;
    std::stringstream message;
    std::mutex mtxLogger;
    std::ofstream logFile;

public:
    Logger()
        : rotateFile(true)
        , level(LogLevel::NONE)
        , logFileName(LOG_FILENAME)
    {
    }

    ~Logger() { Stop(); }

    /**
     * @brief Customize logger file and logger level settings
     * If file is user defined then not rotating the file.
     */
    void SetFile(std::string fileName, bool rotateFile = false)
    {
        std::lock_guard<std::mutex> lock(mtxLogger);
        this->rotateFile = rotateFile;
        logFileName = fileName;
        if (this->level > LogLevel::NONE) {
            Restart();
        }
    }

    /** @brief set the loglevel of the logger object to the provided string
     * @param level loglevel string
     */
    void SetLevelString(std::string level)
    {
        if (StringToLogLevel.count(level) == 0)
            return;
        this->level = StringToLogLevel[level];
        if (this->level > LogLevel::NONE)
            Start();
    }

    /** @brief gets the loglevel of the logger object */
    inline LogLevel GetLevel() { return this->level; }

    /** @brief gets the logger filename */
    inline std::string GetFilename() { return this->logFileName; }

    /**
     * @brief Print statement with variable types.
     */
    template <typename T, typename... Args>
    void Log(LogLevel level, const T& log, Args... args)
    {
        std::lock_guard<std::mutex> lock(mtxLogger);
        if (this->level == LogLevel::NONE || level > this->level)
            return;
        CheckIfFileSizeLimitReached();
        logFile << GetDateAndTime() << " " << Level[level] << ":";
        _Log(log, args...);
        logFile << std::endl;
    }

private:
    template <typename T>
    void _Log(const T& log)
    {
        logFile << log;
    }

    template <typename T, typename... Args>
    void _Log(const T& log, Args... args)
    {
        logFile << log << " ";
        _Log(args...);
    }

    std::string GetDateAndTime()
    {
        time_t now;
        time(&now);
        struct tm tstruct = *localtime(&now);
        std::stringstream currTime;
        currTime << tstruct.tm_year + 1900 << "-" << tstruct.tm_mon << "-" << tstruct.tm_mday
                 << " ";
        currTime << tstruct.tm_hour << ":" << tstruct.tm_min << ":" << tstruct.tm_sec;
        return currTime.str();
    }

    /**
     *  @brief clears contents in logfile if it reaches above 10mb.
     */
    void CheckIfFileSizeLimitReached()
    {
        if (rotateFile && logFile.tellp() >= 1000000) {
            Stop();
            logFile.open(logFileName, std::ios_base::out);
        }
    }

    inline void Start()
    {
        if (logFile.is_open())
            return;
        logFile.open(logFileName, std::ios_base::app);
        CheckIfFileSizeLimitReached();
    }

    inline void Stop()
    {
        if (logFile.is_open())
            logFile.close();
    }

    inline void Restart()
    {
        Stop();
        Start();
    }
};

/*Usage of Logger*/
// int main()
//{
//	Logger* logger = new Logger();
//	logger->SetLevelString("debug");
//	logger->Log(LogLevel::INFO, "Initializing Logger");
//	logger->Log(LogLevel::DEBUG, "Testing print values");
//	logger->Log(LogLevel::INFO, "Terminating Logger");
//	delete logger;
//	return 0;
// }
