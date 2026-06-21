#pragma once

namespace ga::utils
{
    __device__
    float standard_normal_cdf(float x)
    {
        constexpr float inv_sqrt_2 = 0.7071067811865475f;

        return 0.5f * (1.0f + erff(x * inv_sqrt_2));
    }

    __device__
    float standard_normal_inverse_cdf(float p)
    {
        constexpr float sqrt_2 = 1.4142135623730951f;
        constexpr float eps = 1.0e-7f;

        p = fminf(fmaxf(p, eps), 1.0f - eps);

        return sqrt_2 * erfinvf(2.0f * p - 1.0f);
    }

};