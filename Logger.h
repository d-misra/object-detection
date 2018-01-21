#ifndef LOGGER_H
#define LOGGER_H

#include "external/spdlog/spdlog.h"

auto logger = spdlog::stdout_color_mt("console");

#endif // LOGGER_H