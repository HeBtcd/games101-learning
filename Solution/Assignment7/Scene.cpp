//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// L_o = L_e + L_dir + L_indir (L_e 自发光项, L_dir 直接光照, L_indir 间接光照).
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    if (depth > maxDepth) return Vector3f(0.f, 0.f, 0.f);

    // 1. L_e
    Intersection isect = intersect(ray); // BVH 求交.
    if (!isect.happened) return Vector3f(0.f, 0.f, 0.f);
    auto L_e = Vector3f(0.f, 0.f, 0.f);
    if (isect.m && isect.m->hasEmission()) {
        if (depth == 0) {
            L_e = isect.m->getEmission();
        } else {
            return Vector3f(0.f, 0.f, 0.f);
        }
    }

    // 2. L_dir
    // ∫( L_i · f_r · cosθ · cosθ'/dist^2 · 1/pdf )dA
    auto L_dir = Vector3f(0.f, 0.f, 0.f);
    
    Intersection isect_dir;
    float pdf_dir; // 选中某点的概率. 因为这个点代表了整个光源的表面积, 所以后面要除以 pdf_dir.
    sampleLight(isect_dir, pdf_dir); // 随机选择发光物体, 随机选择其上的点, 然后填进出参.

    const auto L_i = isect_dir.emit;
    const auto p_dir = isect_dir.coords;
    const auto p = isect.coords;
    const auto diff = p_dir - p;
    const auto wi0 = normalize(diff);
    const auto wo0 = -ray.direction;
    const auto n = isect.normal;
    const auto n_dir = isect_dir.normal;
    const auto f_r0 = isect.m->eval(wo0, wi0, n);
    const auto cos0 = std::max(0.0f, static_cast<float>(dotProduct(n, wi0)));
    const auto cos_dir = std::max(0.0f, static_cast<float>(dotProduct(n_dir, -wi0)));
    const auto dist2 = dotProduct(diff, diff);
    const auto dist = std::sqrt(dist2);

    // 遮挡.
    const auto isect_block = intersect(Ray(p + n * EPSILON, wi0));
    if (isect_block.happened && isect_block.obj && isect_block.obj->hasEmit() && std::fabs(static_cast<float>(isect_block.distance) - static_cast<float>(dist)) < 1e-2f) {
        if (pdf_dir > EPSILON) {
            L_dir = L_i * f_r0 * cos0 * cos_dir / dist2 / pdf_dir;
        }
    }

    // 3. L_indir
    // L_i(recur) · f_r · cosθ · 1/pdf / RussianRoulette
    auto L_indir = Vector3f(0.f, 0.f, 0.f);

    if (get_random_float() < RussianRoulette)
    {
        const auto wi1 = normalize(isect.m->sample(wo0, n));
        const auto pdf_indir = isect.m->pdf(wo0, wi1, n);
        const auto L_recur = castRay(Ray(p + n * EPSILON, wi1), depth + 1);
        const auto f_r1 = isect.m->eval(wo0, wi1, n);
        const auto cos1 = std::max(0.0f, static_cast<float>(dotProduct(n, wi1)));

        if (pdf_indir > EPSILON) {
            L_indir = L_recur * f_r1 * cos1 / pdf_indir / RussianRoulette;
        }
    }

    return L_e + L_dir + L_indir;
}
