/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef THREAD_SELECTION_H_INCLUDED
#define THREAD_SELECTION_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <vector>

#include "types.h"

namespace Stockfish {

struct RootObservation {
    Move   move;
    double value;
    double variance;
    double margin;
    double marginError;
    usize  pvLength;
};

struct MoveSummary {
    Move   move;
    double precision;
    double mean;
    double variance;
    double evidence;
};

enum class ThreadSelection {
    Vote,
    Competence,
};

inline double competence_log_odds(double margin, double marginError) {
    const double separation  = margin / std::max(marginError, 1e-9);
    const double probability = std::clamp(0.5 * (1.0 + std::erf(separation / std::sqrt(2.0))),
                                         1e-3, 1.0 - 1e-3);
    return std::log(probability / (1.0 - probability));
}

inline std::vector<MoveSummary> summarize_votes(const std::vector<RootObservation>& observations) {
    std::vector<MoveSummary> summaries;

    for (const RootObservation& observation : observations)
    {
        usize slot = 0;

        while (slot < summaries.size() && summaries[slot].move != observation.move)
            ++slot;

        if (slot == summaries.size())
            summaries.push_back({observation.move, 0.0, 0.0, 0.0, 0.0});

        const double observationPrecision = 1.0 / std::max(observation.variance, 1.0);

        summaries[slot].precision += observationPrecision;
        summaries[slot].mean += observation.value * observationPrecision;
        summaries[slot].evidence += competence_log_odds(observation.margin, observation.marginError);
    }

    for (MoveSummary& summary : summaries)
    {
        summary.mean     = summary.mean / summary.precision;
        summary.variance = 1.0 / summary.precision;
    }

    return summaries;
}

inline usize find_summary(const std::vector<MoveSummary>& summaries, Move move) {
    for (usize i = 0; i < summaries.size(); ++i)
        if (summaries[i].move == move)
            return i;

    return summaries.size();
}

inline double between_move_variance(const std::vector<MoveSummary>& summaries) {
    double grandMean = 0.0;

    for (const MoveSummary& summary : summaries)
        grandMean += summary.mean;

    grandMean /= double(summaries.size());

    double between = 0.0, within = 0.0;

    for (const MoveSummary& summary : summaries)
    {
        between += (summary.mean - grandMean) * (summary.mean - grandMean);
        within += summary.variance;
    }

    between /= double(std::max<usize>(summaries.size(), 2) - 1);
    within /= double(summaries.size());

    return std::max(0.0, between - within);
}

inline double shrunken_score(const std::vector<MoveSummary>& summaries, usize summaryIndex) {
    double grandMean = 0.0;

    for (const MoveSummary& summary : summaries)
        grandMean += summary.mean;

    grandMean /= double(summaries.size());

    const double tauSquared = between_move_variance(summaries);
    const double shrinkage  = summaries[summaryIndex].variance
                           / (summaries[summaryIndex].variance + tauSquared);

    return grandMean + (1.0 - shrinkage) * (summaries[summaryIndex].mean - grandMean);
}

inline std::vector<double> weighted_votes(const std::vector<RootObservation>& observations,
                                          ThreadSelection                     selection) {
    const std::vector<MoveSummary> summaries = summarize_votes(observations);
    std::vector<double>            weights(observations.size());

    double lowestValue = observations.empty() ? 0.0 : observations.front().value;

    for (const RootObservation& observation : observations)
        lowestValue = std::min(lowestValue, observation.value);

    for (usize i = 0; i < observations.size(); ++i)
    {
        if (selection == ThreadSelection::Competence)
        {
            weights[i] = summaries[find_summary(summaries, observations[i].move)].evidence;
            continue;
        }

        double vote = 0.0;

        for (const RootObservation& other : observations)
            if (other.move == observations[i].move)
                vote += other.value - lowestValue + 14.0;

        weights[i] = vote;
    }

    return weights;
}

}  // namespace Stockfish

#endif  // #ifndef THREAD_SELECTION_H_INCLUDED
