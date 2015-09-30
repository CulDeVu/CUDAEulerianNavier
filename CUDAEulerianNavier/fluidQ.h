#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>

using namespace std;

class fluidQ;

__device__ __host__ void eulerIntegrator(float*, float*, float, fluidQ*, fluidQ*);
__device__ __host__ void RK2Integrator(float*, float*, float, fluidQ*, fluidQ*);

class fluidQ
{
	__device__ __host__ float lerp(float a, float b, float t)
	{
		return a * (1 - t) + b * t;
	}

	__device__ __host__ float cerp(float a, float b, float c, float d, float t)
	{
		float t_2 = t * t;
		float t_3 = t_2 * t;

		float minV = min(min(min(a, b), b), d);
		float maxV = max(max(max(a, b), b), d);

		float ret =
			a*(0.0 - 0.5*t + 1.0*t_2 - 0.5*t_3) +
			b*(1.0 + 0.0*t - 2.5*t_2 + 1.5*t_3) +
			c*(0.0 + 0.5*t + 2.0*t_2 - 1.5*t_3) +
			d*(0.0 + 0.0*t - 0.5*t_2 + 0.5*t_3);

		return min(maxV, max(minV, ret));
	}

public:
	int w, h;
	float* cur;
	float* old;

	float ox, oy;
	float delta_x;

	void createCUDA(int width, int height, float oX, float oY, float delta_X)
	{
		w = width;
		h = height;
		ox = oX;
		oy = oY;
		delta_x = delta_X;

		cudaMalloc((void**)&cur, w * h * sizeof(float));
		cudaMalloc((void**)&old, w * h * sizeof(float));

		cudaMemset(cur, 0, w * h * sizeof(float));
	}
	void create(int width, int height, float oX, float oY, float delta_X)
	{
		w = width;
		h = height;
		ox = oX;
		oy = oY;
		delta_x = delta_X;

		cur = new float[w * h];
		old = new float[w * h];

		memset(cur, 0, w * h * sizeof(float));
	}

	void cleanupCUDA()
	{
		cudaFree(cur);
		cudaFree(old);
	}
	void cleanup()
	{
		delete[] cur;
		delete[] old;
	}

	void flip()
	{
		float* lulz = cur;
		cur = old;
		old = lulz;
	}

	__device__ __host__ float at(int x, int y) const
	{
		return cur[y * w + x];
	}
	__device__ __host__ float &at(int x, int y)
	{
		return cur[y * w + x];
	}

	__device__ __host__ float lerp(float x, float y)
	{
		x = min(max(x - ox, 0.f), w - 1.001f);
		y = min(max(y - oy, 0.f), h - 1.001f);
		int ix = (int)x;
		int iy = (int)y;

		x -= ix;
		y -= iy;

		float x00 = at(ix + 0, iy + 0);
		float x10 = at(ix + 1, iy + 0);
		float x01 = at(ix + 0, iy + 1);
		float x11 = at(ix + 1, iy + 1);

		return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
	}

	__device__ __host__ float cerp(float x, float y)
	{
		x = min(max(x - ox, 0.f), w - 1.001f);
		y = min(max(y - oy, 0.f), h - 1.001f);
		int ix = (int)x;
		int iy = (int)y;

		x -= ix;
		y -= iy;

		int x0 = max(ix - 1, 0), x1 = ix, x2 = ix + 1, x3 = min(ix + 2, w - 1);
		int y0 = max(iy - 1, 0), y1 = iy, y2 = iy + 1, y3 = min(iy + 2, h - 1);

		double q0 = cerp(at(x0, y0), at(x1, y0), at(x2, y0), at(x3, y0), x);
		double q1 = cerp(at(x0, y1), at(x1, y1), at(x2, y1), at(x3, y1), x);
		double q2 = cerp(at(x0, y2), at(x1, y2), at(x2, y2), at(x3, y2), x);
		double q3 = cerp(at(x0, y3), at(x1, y3), at(x2, y3), at(x3, y3), x);

		return cerp(q0, q1, q2, q3, y);
	}

	void advect(float delta_t, fluidQ* u, fluidQ* v)
	{
		for (int iy = 0; iy < h; ++iy)
		{
			for (int ix = 0; ix < w; ++ix)
			{
				float x = ix + ox;
				float y = iy + oy;

				RK2Integrator(&x, &y, -delta_t, u, v);
				x /= delta_x;
				y /= delta_x;

				old[iy * w + ix] = lerp(x, y);
			}
		}
	}

	__device__ __host__ void set(float x, float y, float v)
	{
		int ix = x;
		int iy = y;

		cur[iy * w + ix] = v;
	}
};

__device__ __host__ void eulerIntegrator(float* x, float* y, float delta_t, fluidQ* u, fluidQ* v)
{
	float uVel = u->lerp(*x, *y);
	float vVel = v->lerp(*x, *y);

	*x += uVel * delta_t;
	*y += vVel * delta_t;
}

__device__ __host__ void RK2Integrator(float* x, float* y, float delta_t, fluidQ* u, fluidQ* v)
{
	float uVel = u->lerp(*x, *y);
	float vVel = v->lerp(*x, *y);

	float x1 = *x + uVel * delta_t / 2;
	float y1 = *y + vVel * delta_t / 2;

	float u2 = u->lerp(x1, y1);
	float v2 = v->lerp(x1, y1);

	*x = *x + delta_t * u2;
	*y = *y + delta_t * v2;
}