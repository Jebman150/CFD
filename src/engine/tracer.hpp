#pragma once

#include <unordered_map>
#include <string>
#include <chrono>
#include <vector>

class Tracer {
    std::unordered_map<std::string, std::pair<std::chrono::_V2::system_clock::time_point, std::chrono::_V2::system_clock::time_point>> runningJobs;
    std::unordered_map<std::string, std::vector<std::chrono::duration<float, std::milli>>> jobTimes;

public:
    void startJob(std::string name) { runningJobs[name].first = std::chrono::high_resolution_clock::now(); }
    void endJob(std::string name) {
        runningJobs[name].second = std::chrono::high_resolution_clock::now();
        jobTimes[name].push_back(runningJobs[name].second - runningJobs[name].first);
    }

    float getAvgTime(std::string name) {
        int i = 0;
        std::chrono::duration<float, std::milli> total;
        for(auto time : jobTimes[name]) {
            total += time;
            i++;
        }
        return total.count() / float(i);
    }
};