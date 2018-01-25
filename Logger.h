#ifndef LOGGER_H
#define LOGGER_H

#include "external/spdlog/spdlog.h"

#define SPDLOG_TRACE_ON

auto logger = spdlog::stdout_color_mt("console");

#endif // LOGGER_H