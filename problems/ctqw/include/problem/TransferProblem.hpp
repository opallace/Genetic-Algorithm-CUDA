#pragma once

namespace ctqw::problem {

    struct TransferProblem {
        int source = 0;
        int target = 0;

        float final_time = 1.0f;
        int steps        = 1000;
    };

    struct TransferResult {
        float target_probability = 0.0f;
        float total_probability  = 0.0f;
        float norm_error         = 0.0f;
    };

}