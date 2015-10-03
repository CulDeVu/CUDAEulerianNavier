#pragma once

#include <cusp\array1d.h>
#include <cusp\coo_matrix.h>
#include <cusp\krylov\bicgstab.h>
#include <cusp\krylov\cg.h>
#include <cusp\monitor.h>
#include <cusp\precond\diagonal.h>

#include <Windows.h>

#include "fluidQ.h"

#define dt (1.f / 60.f)

#define mapW 128
#define mapH 128

float p[mapW][mapH];

enum cellType
{
	WATER, AIR, SOLID
};

#define PROJECT_ITERCOUNT 600
#define PROJECT_TOLERANCE 0.0001

struct cuspTriple
{
	int row, col;
	float amount;
};
void projectGPU(fluidQ* u, fluidQ* v, cellType* type)
{
	int countBuffer[mapW][mapH];
	int counter = 0;
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			if (type[y * mapW + x] == WATER)
			{
				countBuffer[x][y] = counter;
				++counter;
			}
			else
				countBuffer[x][y] = -1;
		}
	}
	cusp::array1d<float, cusp::host_memory> pressure(counter);

	cusp::array1d<float, cusp::host_memory> b(counter);
	{
		float scale = 1;
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				int index = countBuffer[x][y];
				if (index == -1)
					continue;

				b[index] = scale * (u->at(x + 1, y) - u->at(x, y) +
					v->at(x, y + 1) - v->at(x, y));
			}
		}
	}

	vector<cuspTriple> data;
	{
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				float scale = 1;
				int n = 0;

				int index = countBuffer[x][y];
				if (index == -1)
					continue;

				if (x > 0)
				{
					if (type[y * mapW + x - 1] != SOLID)
					{
						++n;

						int indexOther = countBuffer[x - 1][y];
						if (indexOther != -1)
						{
							cuspTriple t;
							t.row = index;
							t.col = indexOther;
							t.amount = 1;
							data.push_back(t);
						}
					}
				}
				if (y > 0) {
					if (type[(y - 1) * mapW + x] != SOLID)
					{
						++n;

						int indexOther = countBuffer[x][y - 1];
						if (indexOther != -1)
						{
							cuspTriple t;
							t.row = index;
							t.col = indexOther;
							t.amount = 1;
							data.push_back(t);
						}
					}
				}
				if (x < mapW - 1) {
					if (type[y * mapW + x + 1] != SOLID)
					{
						++n;

						int indexOther = countBuffer[x + 1][y];
						if (indexOther != -1)
						{
							cuspTriple t;
							t.row = index;
							t.col = indexOther;
							t.amount = 1;
							data.push_back(t);
						}
					}
				}
				if (y < mapH - 1) {
					if (type[(y + 1) * mapW + x] != SOLID)
					{
						++n;

						int indexOther = countBuffer[x][y + 1];
						if (indexOther != -1)
						{
							cuspTriple t;
							t.row = index;
							t.col = indexOther;
							t.amount = 1;
							data.push_back(t);
						}
					}
				}

				if (n == 0)
					continue; // this is safe because, if n == 0, the above steps can't have added anything to the matrix on that row either

				cuspTriple t;
				t.row = index;
				t.col = index;
				t.amount = -n;
				data.push_back(t);
			}
		}

	}
	cusp::coo_matrix<int, float, cusp::host_memory> A(counter, counter, data.size());
	{
		for (int i = 0; i < data.size(); ++i)
		{
			A.row_indices[i] = data[i].row;
			A.column_indices[i] = data[i].col;
			A.values[i] = data[i].amount;
		}
	}

	cusp::default_monitor<float> monitor(b, PROJECT_ITERCOUNT, PROJECT_TOLERANCE, 0);
	cusp::precond::diagonal<float, cusp::host_memory> M(A);

	cusp::krylov::bicgstab(A, pressure, b, monitor, M);

	if (!monitor.converged())
	{
		std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
		std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
	}

	{
		float scale = 1;

		for (int y = 0; y < mapH; y++)
		{
			for (int x = 0; x < mapW; x++)
			{
				if (type[y * mapW + x] != WATER)
					continue;

				float p = pressure[countBuffer[x][y]];

				u->at(x, y) -= scale * p;
				u->at(x + 1, y) += scale * p;
				v->at(x, y) -= scale * p;
				v->at(x, y + 1) += scale * p;
			}
		}
	}
}

void projectCPU(fluidQ* u, fluidQ* v, cellType* type)
{
	float r[mapW][mapH];
	{
		float scale = 1;
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				r[x][y] = scale * (u->at(x + 1, y) - u->at(x, y) +
					v->at(x, y + 1) - v->at(x, y));
			}
		}
	}

	{
		float scale = 1;
		int N = PROJECT_ITERCOUNT;

		float maxDelta = 0;
		for (int iter = 0; iter < N; ++iter)
		{
			maxDelta = 0;

			for (int y = 0; y < mapH; ++y)
			{
				for (int x = 0; x < mapW; ++x)
				{
					if (type[y * mapW + x] == AIR)
						continue;

					float sigma = 0;
					float n = 0;

					if (x > 0) {
						if (type[y * mapW + x - 1] != SOLID)
						{
							if (type[y * mapW + x - 1] == WATER)
								sigma += 1 * p[x - 1][y];
							++n;
						}
					}
					if (y > 0) {
						if (type[(y - 1) * mapW + x] != SOLID)
						{
							if (type[(y - 1) * mapW + x] == WATER)
								sigma += 1 * p[x][y - 1];
							++n;
						}
					}
					if (x < mapW - 1) {
						if (type[y * mapW + x + 1] != SOLID)
						{
							if (type[y * mapW + x + 1] == WATER)
								sigma += 1 * p[x + 1][y];
							++n;
						}
					}
					if (y < mapH - 1) {
						if (type[(y + 1) * mapW + x] != SOLID)
						{
							if (type[(y + 1) * mapW + x] == WATER)
								sigma += 1 * p[x][y + 1];
							++n;
						}
					}

					float newP = (r[x][y] - sigma) / -n;
					maxDelta = max(maxDelta, fabs(p[x][y] - newP));

					p[x][y] = newP;
				}
			}

			//enforceBoundary();

			if (maxDelta < PROJECT_TOLERANCE)
			{
				//printf("maxdelta good enough after %d\n", iter);
				goto next;
			}
		}
	}
next:

	{
		float scale = 1;

		for (int y = 0; y < mapH; y++)
		{
			for (int x = 0; x < mapW; x++)
			{
				if (type[y * mapW + x] != WATER)
					continue;
				u->at(x, y) -= scale * p[x][y];
				u->at(x + 1, y) += scale * p[x][y];
				v->at(x, y) -= scale * p[x][y];
				v->at(x, y + 1) += scale * p[x][y];
			}
		}
	}
}

void project(fluidQ* u, fluidQ* v, cellType* type)
{
	static bool gpuProject = false;
	static bool isKeyDown = false;
	static int counter = 0;

	if (counter == 1)
		gpuProject = true;

	if (GetAsyncKeyState('P'))
	{
		isKeyDown = true;
	}
	else if (isKeyDown)
	{
		isKeyDown = false;
		gpuProject = !gpuProject;
	}

	if (gpuProject)
		projectGPU(u, v, type);
	else
		projectCPU(u, v, type);

	++counter;
}