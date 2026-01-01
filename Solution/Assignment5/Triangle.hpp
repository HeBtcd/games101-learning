#pragma once

#include "Object.hpp"

#include <cstring>

const auto EPSILON = 0.00001f;

// Möller-Trumbore 算法.
bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    const auto edge1 = v1 - v0;
    const auto edge2 = v2 - v0;
    const auto t_vec = orig - v0;
    const auto neg_dir = -dir; 

    // orig + t_vec·dir = v0 + u·(v1 - v0) + v·(v2 - v0)
    //   => -t_vec·dir + u·edge0 + v·edge1 = t_vec

    // 克莱姆法则.

    // [ -D, edge1, edge2 ].
    const auto det = dotProduct(crossProduct(neg_dir, edge1), edge2);
    if (det < EPSILON) return false;

    // 将第一列换成 t_vec => [ t_vec, edge1, edge2 ].
    const auto det_t = dotProduct(crossProduct(t_vec, edge1), edge2);
    const auto t_val = det_t / det;    
    if (t_val < EPSILON) return false;

    // 将第二列换成 t_vec => [ -D, t_vec, edge2 ].
    const auto det_u = dotProduct(crossProduct(neg_dir, t_vec), edge2);
    const auto u_val = det_u / det;
    if (u_val < 0.f || u_val > 1.f) return false;

    // 将第三列换成 t_vec => [ -D, edge1, t_vec ].
    const auto det_v = dotProduct(crossProduct(neg_dir, edge1), t_vec);
    const auto v_val = det_v / det;
    if (v_val < 0.f || (u_val + v_val) > 1.f) return false;

    tnear = t_val;
    u = u_val;
    v = v_val;

    return true;
}

class MeshTriangle : public Object
{
public:
    MeshTriangle(const Vector3f* verts, const uint32_t* vertsIndex, const uint32_t& numTris, const Vector2f* st)
    {
        uint32_t maxIndex = 0;
        for (uint32_t i = 0; i < numTris * 3; ++i)
            if (vertsIndex[i] > maxIndex)
                maxIndex = vertsIndex[i];
        maxIndex += 1;
        vertices = std::unique_ptr<Vector3f[]>(new Vector3f[maxIndex]);
        memcpy(vertices.get(), verts, sizeof(Vector3f) * maxIndex);
        vertexIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        memcpy(vertexIndex.get(), vertsIndex, sizeof(uint32_t) * numTris * 3);
        numTriangles = numTris;
        stCoordinates = std::unique_ptr<Vector2f[]>(new Vector2f[maxIndex]);
        memcpy(stCoordinates.get(), st, sizeof(Vector2f) * maxIndex);
    }

    bool intersect(const Vector3f& orig, const Vector3f& dir, float& tnear, uint32_t& index,
                   Vector2f& uv) const override
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k)
        {
            const Vector3f& v0 = vertices[vertexIndex[k * 3]];
            const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) && t < tnear)
            {
                tnear = t;
                uv.x = u;
                uv.y = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t& index, const Vector2f& uv, Vector3f& N,
                              Vector2f& st) const override
    {
        const Vector3f& v0 = vertices[vertexIndex[index * 3]];
        const Vector3f& v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[index * 3 + 2]];
        Vector3f e0 = normalize(v1 - v0);
        Vector3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vector2f& st0 = stCoordinates[vertexIndex[index * 3]];
        const Vector2f& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vector2f& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    }

    Vector3f evalDiffuseColor(const Vector2f& st) const override
    {
        float scale = 5;
        float pattern = (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031), Vector3f(0.937, 0.937, 0.231), pattern);
    }

    std::unique_ptr<Vector3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vector2f[]> stCoordinates;
};
