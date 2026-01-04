#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        // 1. mass.
        for (int i = 0; i < num_nodes; i++)
        {
            float t = (float)i / ((float)num_nodes - 1.0f);
            const auto pos = (1 - t) * start + t * end;
            auto mass = new Mass(pos, node_mass, false);
            masses.push_back(mass);
        }

        for (auto &i : pinned_nodes) {
            masses[i]->pinned = true;
        }

        // 2. spring.
        for (int i = 0; i < num_nodes - 1; i++)
        {
            auto mass0 = masses[i];
            auto mass1 = masses[i + 1];
            auto spring = new Spring(mass0, mass1, k);
            springs.push_back(spring);
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // F = k · (|b - a| - l_rest) · (b - a)/|b - a|.
            const auto diff = s->m2->position - s->m1->position;
            const auto length = diff.norm();
            const auto offset = length - s->rest_length;
            const auto dir = diff / length;
            const auto f = s->k * offset * dir;

            s->m1->forces += f;
            s->m2->forces -= f;
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                m->forces += gravity * m->mass;

                // v_{t+1} = v_t + F_t/m · Δt.
                // x_{t+1} = x_t + v_t · Δt.
                m->velocity = m->velocity + m->forces / m->mass * delta_t;
                m->position = m->position + m->velocity * delta_t;
            }

            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // x_{t+1} = x_t + (1 - damp) · (x_t - x_{t-1}) + a · Δt^2.
                const auto temp_position = m->position;
                const auto damp = 0.001f;
                const auto new_position = m->position + (1 - damp) * (m->position - m->last_position) + gravity * delta_t * delta_t;
                m->position = new_position;
                m->last_position = temp_position;
            }
        }

        for (auto &s : springs)
        {
            const auto diff = s->m2->position - s->m1->position;
            const auto length = diff.norm();
            const auto dir = diff / length;
            const auto offset = length - s->rest_length;
            const auto correction = offset / 2.0f * dir;
            if (!s->m1->pinned && !s->m2->pinned) {
                s->m1->position += correction; // m1 被拉向 m2.
                s->m2->position -= correction; // m2 被拉向 m1.
            }
            else if (s->m1->pinned && !s->m2->pinned) { // m2 要负责把两份距离都跑完.
                s->m2->position -= correction * 2.0f;
            }
            else if (!s->m1->pinned && s->m2->pinned) { // 同理.
                s->m1->position += correction * 2.0f;
            }
        }
    }
}
