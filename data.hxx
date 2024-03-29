#pragma once
#include <vector>
#include "isolation_forest.hxx"

Matrix<float> example_data({
{
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
17.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
17.0, 87.0, 24.0
}),
MatrixRow<float>({
0.0, 87.0, 30.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
37.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({ 40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 69.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 21.0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
8.0, 86.0, 9.0
}),
MatrixRow<float>({ 40.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
40.0, 86.0, 100.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 58.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 46.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 55.0
}),
MatrixRow<float>({ 40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 32.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
17.0, 86.0, 9.0
}),
MatrixRow<float>({
2.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
2.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
17.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
3.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({ 15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 85.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 44.0
}),
MatrixRow<float>({
22.0, 87.0, 80.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({ 22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
2.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
36.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 13.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
36.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 70.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
36.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 68.0
}),
MatrixRow<float>({
40.0, 91.0, 78.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 10.0
}),
MatrixRow<float>({
15.0, 90.0, 112.0
}),
MatrixRow<float>({
22.0, 90.0, 26.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 84.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({ 40.0, 91.0, 64.0
}),
MatrixRow<float>({
40.0, 91.0, 68.0
}),
MatrixRow<float>({
40.0, 91.0, 68.0
}),
MatrixRow<float>({
17.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
37.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
17.0, 91.0, 14.0
}),
MatrixRow<float>({
31.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
8.0, 91.0, 29.0
}),
MatrixRow<float>({
8.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 95.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({ 17.0, 91.0, 30.0
}),
MatrixRow<float>({
17.0, 91.0, 22.0
}),
MatrixRow<float>({
8.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
29.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
8.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 48.0
}),
MatrixRow<float>({
8.0, 88.0, 11.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
2.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({ 22.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
36.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 62.0
}),
MatrixRow<float>({
21.0, 90.0, 50.0
}),
MatrixRow<float>({
31.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
36.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 32.0
}),
MatrixRow<float>({
37.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
21.0, 90.0, 80.0
}),
MatrixRow<float>({
22.0, 90.0, 57.0
}),
MatrixRow<float>({
31.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
19.0, 90.0, 36.0
}),
MatrixRow<float>({ 40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
0.0, 90.0, 22.0
}),
MatrixRow<float>({
36.0, 90.0, 25.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
2.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 49.0
}),
MatrixRow<float>({
40.0, 85.0, 85.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
2.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 36.0
}),
MatrixRow<float>({
15.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
37.0, 85.0, 13.0
}),
MatrixRow<float>({ 0.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 49.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
36.0, 89.0, 18.0
}),
MatrixRow<float>({
36.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
0.0, 89.0, 37.0
}),
MatrixRow<float>({
40.0, 89.0, 39.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
36.0, 89.0, 19.0
}),
MatrixRow<float>({
36.0, 89.0, 40.0
}),
MatrixRow<float>({
0.0, 89.0, 14.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 36.0
}),
MatrixRow<float>({
3.0, 89.0, 17.0
}),
MatrixRow<float>({ 22.0, 89.0, 25.0
}),
MatrixRow<float>({
18.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 26.0
}),
MatrixRow<float>({
0.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 13.0
}),
MatrixRow<float>({
0.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 52.0
}),
MatrixRow<float>({
3.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
22.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 52.0
}),
MatrixRow<float>({
0.0, 92.0, 215.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
37.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 125.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
2.0, 92.0, 70.0
}),
MatrixRow<float>({
0.0, 92.0, 30.0
}),
MatrixRow<float>({ 40.0, 87.0, 35.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 56.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
32.0, 87.0, 7.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
18.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({ 22.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 35.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 42.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 83.0, 35.0
}),
MatrixRow<float>({
15.0, 83.0, 16.0
}),
MatrixRow<float>({
15.0, 82.0, 11.0
}),
MatrixRow<float>({
37.0, 82.0, 13.0
}),
MatrixRow<float>({
15.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 48.0
}),
MatrixRow<float>({
37.0, 82.0, 11.0
}),
MatrixRow<float>({
8.0, 81.0, 20.0
}),
MatrixRow<float>({
8.0, 80.0, 19.0
}),
MatrixRow<float>({ 2.0, 100.0, 350.0
}),
MatrixRow<float>({
2.0, 98.0, 350.0
}),
MatrixRow<float>({
17.0, 97.0, 775.0
}),
MatrixRow<float>({
2.0, 97.0, 100.0
}),
MatrixRow<float>({
2.0, 97.0, 225.0
}),
MatrixRow<float>({
22.0, 97.0, 150.0
}),
MatrixRow<float>({
19.0, 96.0, 320.0
}),
MatrixRow<float>({
40.0, 96.0, 68.0
}),
MatrixRow<float>({
15.0, 96.0, 630.0
}),
MatrixRow<float>({
17.0, 96.0, 365.0
}),
MatrixRow<float>({
40.0, 96.0, 68.0
}),
MatrixRow<float>({
2.0, 95.0, 85.0
}),
MatrixRow<float>({
15.0, 95.0, 350.0
}),
MatrixRow<float>({
17.0, 95.0, 66.0
}),
MatrixRow<float>({
15.0, 95.0, 110.0
}),
MatrixRow<float>({
2.0, 95.0, 125.0
}),
MatrixRow<float>({
22.0, 95.0, 60.0
}),
MatrixRow<float>({
40.0, 95.0, 200.0
}),
MatrixRow<float>({
15.0, 95.0, 380.0
}),
MatrixRow<float>({
40.0, 95.0, 48.0
}),
MatrixRow<float>({
2.0, 95.0, 60.0
}),
MatrixRow<float>({
22.0, 88.0, 21.0
}),
MatrixRow<float>({
8.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 11.0
}),
MatrixRow<float>({
22.0, 88.0, 21.0
}),
MatrixRow<float>({ 40.0, 88.0, 160.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
18.0, 88.0, 0
}),
MatrixRow<float>({
26.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 11.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 9.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({ 22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 30.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
3.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({ 15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
3.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 89.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 54.0
}),
MatrixRow<float>({
37.0, 89.0, 15.0
}),
MatrixRow<float>({
37.0, 89.0, 42.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
3.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
3.0, 88.0, 12.0
}),
MatrixRow<float>({ 22.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 92.0, 39.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 69.0
}),
MatrixRow<float>({
7.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 66.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
17.0, 92.0, 115.0
}),
MatrixRow<float>({
22.0, 92.0, 69.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
17.0, 92.0, 42.0
}),
MatrixRow<float>({
17.0, 92.0, 27.0
}),
MatrixRow<float>({ 15.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 73.0
}),
MatrixRow<float>({
15.0, 92.0, 72.0
}),
MatrixRow<float>({
31.0, 92.0, 34.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 130.0
}),
MatrixRow<float>({
40.0, 92.0, 88.0
}),
MatrixRow<float>({
15.0, 92.0, 65.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
37.0, 87.0, 19.0
}),
MatrixRow<float>({
3.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({ 40.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
3.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 70.0
}),
MatrixRow<float>({
15.0, 91.0, 59.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
8.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({ 40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
17.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
8.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
17.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 110.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
22.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 54.0
}),
MatrixRow<float>({ 40.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 150.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
22.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 46.0
}),
MatrixRow<float>({
15.0, 93.0, 20.0
}),
MatrixRow<float>({
0.0, 93.0, 42.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
37.0, 93.0, 88.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 62.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 53.0
}),
MatrixRow<float>({
0.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({ 40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 11.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 37.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
3.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 18.0
}),
MatrixRow<float>({
3.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
3.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 41.0
}),
MatrixRow<float>({
3.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({ 40.0, 89.0, 30.0
}),
MatrixRow<float>({
18.0, 89.0, 13.0
}),
MatrixRow<float>({
3.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
26.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 69.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
37.0, 87.0, 35.0
}),
MatrixRow<float>({
0.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 54.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
0.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({ 22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
2.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
15.0, 92.0, 54.0
}),
MatrixRow<float>({
15.0, 92.0, 34.0
}),
MatrixRow<float>({
2.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
37.0, 92.0, 28.0
}),
MatrixRow<float>({
2.0, 92.0, 125.0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 36.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
31.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({ 40.0, 92.0, 50.0
}),
MatrixRow<float>({
31.0, 92.0, 23.0
}),
MatrixRow<float>({
31.0, 92.0, 80.0
}),
MatrixRow<float>({
17.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 24.0
}),
MatrixRow<float>({
15.0, 92.0, 39.0
}),
MatrixRow<float>({
2.0, 92.0, 21.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
36.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
37.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({ 40.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
22.0, 90.0, 53.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
3.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
36.0, 91.0, 50.0
}),
MatrixRow<float>({ 22.0, 91.0, 43.0
}),
MatrixRow<float>({
29.0, 91.0, 28.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
36.0, 91.0, 40.0
}),
MatrixRow<float>({
0.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 65.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
29.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
29.0, 91.0, 120.0
}),
MatrixRow<float>({
40.0, 91.0, 23.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
29.0, 91.0, 50.0
}),
MatrixRow<float>({
0.0, 91.0, 28.0
}),
MatrixRow<float>({
29.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({ 31.0, 85.0, 22.0
}),
MatrixRow<float>({
0.0, 85.0, 14.0
}),
MatrixRow<float>({
0.0, 85.0, 25.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 80.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
15.0, 85.0, 40.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 102.0
}),
MatrixRow<float>({
40.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 33.0
}),
MatrixRow<float>({
22.0, 85.0, 48.0
}),
MatrixRow<float>({ 31.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
40.0, 87.0, 70.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
8.0, 87.0, 19.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
2.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
2.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({ 15.0, 92.0, 62.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 66.0
}),
MatrixRow<float>({
31.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
15.0, 92.0, 72.0
}),
MatrixRow<float>({
22.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 105.0
}),
MatrixRow<float>({
15.0, 92.0, 101.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
31.0, 90.0, 32.0
}),
MatrixRow<float>({
0.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 125.0
}),
MatrixRow<float>({
0.0, 90.0, 46.0
}),
MatrixRow<float>({
17.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({ 15.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
2.0, 90.0, 66.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
37.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 125.0
}),
MatrixRow<float>({
17.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
2.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({ 37.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 70.0
}),
MatrixRow<float>({
22.0, 86.0, 28.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 23.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({ 22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 70.0
}),
MatrixRow<float>({
15.0, 90.0, 117.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
17.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 150.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
17.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
2.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
8.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({ 40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
19.0, 90.0, 16.0
}),
MatrixRow<float>({
31.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
37.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 72.0
}),
MatrixRow<float>({
40.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 58.0
}),
MatrixRow<float>({
2.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 29.0
}),
MatrixRow<float>({ 22.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
37.0, 90.0, 48.0
}),
MatrixRow<float>({
31.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
17.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 39.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
0.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
-1.0, 87.0, 30.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0 }),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
2.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 9.0
}),
MatrixRow<float>({
0.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
2.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
0.0, 87.0, 175.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
15.0, 87.0, 38.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0 }),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 40.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 48.0
}),
MatrixRow<float>({
29.0, 85.0, 9.0 }),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 41.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
32.0, 85.0, 12.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
2.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 53.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 0 }),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
2.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 41.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
15.0, 88.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 21.0
}),
MatrixRow<float>({
2.0, 88.0, 14.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0 }),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 125.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
21.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
37.0, 88.0, 19.0
}),
MatrixRow<float>({
37.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
37.0, 88.0, 28.0
}),
MatrixRow<float>({
29.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
29.0, 87.0, 70.0 }),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
3.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
3.0, 85.0, 24.0
}),
MatrixRow<float>({
3.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
36.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 38.0 }),
MatrixRow<float>({
40.0, 94.0, 34.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 62.0
}),
MatrixRow<float>({
22.0, 94.0, 57.0
}),
MatrixRow<float>({
31.0, 94.0, 150.0
}),
MatrixRow<float>({
31.0, 94.0, 78.0
}),
MatrixRow<float>({
2.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 52.0
}),
MatrixRow<float>({
15.0, 94.0, 112.0
}),
MatrixRow<float>({
22.0, 94.0, 180.0
}),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 32.0
}),
MatrixRow<float>({
40.0, 94.0, 32.0
}),
MatrixRow<float>({
40.0, 94.0, 120.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
15.0, 94.0, 120.0
}),
MatrixRow<float>({
40.0, 94.0, 38.0
}),
MatrixRow<float>({
40.0, 94.0, 32.0
}),
MatrixRow<float>({
40.0, 94.0, 58.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
31.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0 }),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
15.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
8.0, 90.0, 35.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
37.0, 90.0, 60.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
37.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 10.0 }),
MatrixRow<float>({
40.0, 90.0, 56.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 32.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 18.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 65.0
}),
MatrixRow<float>({
29.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 27.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 100.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
8.0, 84.0, 18.0 }),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 17.0
}),
MatrixRow<float>({
22.0, 83.0, 14.0
}),
MatrixRow<float>({
15.0, 83.0, 30.0
}),
MatrixRow<float>({
8.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
3.0, 87.0, 15.0
}),
MatrixRow<float>({
2.0, 87.0, 19.0
}),
MatrixRow<float>({
18.0, 87.0, 15.0
}),
MatrixRow<float>({
3.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
3.0, 87.0, 16.0
}),
MatrixRow<float>({
3.0, 87.0, 21.0
}),
MatrixRow<float>({
18.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0 }),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
0.0, 86.0, 17.0
}),
MatrixRow<float>({
17.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
0.0, 86.0, 12.0
}),
MatrixRow<float>({
3.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 88.0, 14.0
}),
MatrixRow<float>({
0.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
2.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 49.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
39.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 70.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0 }),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
0.0, 88.0, 16.0
}),
MatrixRow<float>({
0.0, 87.0, 40.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 90.0
}),
MatrixRow<float>({
2.0, 87.0, 25.0
}),
MatrixRow<float>({
2.0, 87.0, 13.0
}),
MatrixRow<float>({
17.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 75.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 80.0
}),
MatrixRow<float>({
31.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({
40.0, 93.0, 115.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0 }),
MatrixRow<float>({
22.0, 93.0, 33.0
}),
MatrixRow<float>({
22.0, 93.0, 57.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
31.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 49.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 9.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0 }),
MatrixRow<float>({
36.0, 88.0, 17.0
}),
MatrixRow<float>({
17.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
36.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
36.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
17.0, 88.0, 22.0
}),
MatrixRow<float>({
17.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 7.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
36.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
37.0, 88.0, 21.0
}),
MatrixRow<float>({
22.0, 88.0, 90.0
}),
MatrixRow<float>({
37.0, 85.0, 7.0
}),
MatrixRow<float>({
22.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
21.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0 }),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 70.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
37.0, 84.0, 8.0
}),
MatrixRow<float>({
37.0, 84.0, 27.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
29.0, 84.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
17.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 48.0
}),
MatrixRow<float>({
22.0, 89.0, 26.0 }),
MatrixRow<float>({
22.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 75.0
}),
MatrixRow<float>({
37.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
22.0, 89.0, 49.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 21.0
}),
MatrixRow<float>({
17.0, 89.0, 28.0
}),
MatrixRow<float>({
17.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
31.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0 }),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 98.0
}),
MatrixRow<float>({
29.0, 90.0, 25.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
3.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 56.0
}),
MatrixRow<float>({
37.0, 90.0, 33.0
}),
MatrixRow<float>({
29.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 13.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 21.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0 }),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
0.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 35.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
15.0, 86.0, 75.0
}),
MatrixRow<float>({
37.0, 86.0, 47.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
22.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 27.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 9.0 }),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
17.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
36.0, 88.0, 16.0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 20.0 }),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
31.0, 88.0, 21.0
}),
MatrixRow<float>({
17.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
31.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
17.0, 88.0, 9.0
}),
MatrixRow<float>({
31.0, 88.0, 40.0
}),
MatrixRow<float>({
31.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 31.0
}),
MatrixRow<float>({
15.0, 93.0, 30.0 }),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 24.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
3.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 86.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 120.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
36.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
3.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
3.0, 92.0, 0 }),
MatrixRow<float>({
3.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 120.0
}),
MatrixRow<float>({
15.0, 94.0, 25.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
31.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 28.0
}),
MatrixRow<float>({
15.0, 94.0, 180.0
}),
MatrixRow<float>({
15.0, 94.0, 85.0
}),
MatrixRow<float>({
15.0, 94.0, 135.0
}),
MatrixRow<float>({
22.0, 94.0, 125.0
}),
MatrixRow<float>({
15.0, 94.0, 90.0
}),
MatrixRow<float>({
22.0, 94.0, 154.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 150.0
}),
MatrixRow<float>({
17.0, 94.0, 56.0
}),
MatrixRow<float>({
15.0, 94.0, 49.0
}),
MatrixRow<float>({
40.0, 94.0, 72.0
}),
MatrixRow<float>({
22.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 70.0
}),
MatrixRow<float>({
15.0, 93.0, 132.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
31.0, 93.0, 20.0
}),
MatrixRow<float>({
40.0, 93.0, 175.0 }),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 110.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
22.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
2.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 19.0
}),
MatrixRow<float>({
29.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 75.0
}),
MatrixRow<float>({
19.0, 85.0, 48.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 35.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
29.0, 85.0, 23.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0 }),
MatrixRow<float>({
29.0, 85.0, 16.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
40.0, 85.0, 65.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
0.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 71.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 15.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
8.0, 92.0, 92.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 16.0 }),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 26.0
}),
MatrixRow<float>({
8.0, 92.0, 38.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
8.0, 92.0, 29.0
}),
MatrixRow<float>({
22.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
8.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
22.0, 86.0, 21.0
}),
MatrixRow<float>({
31.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 34.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0 }),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
32.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
37.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 7.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 99.0, 125.0
}),
MatrixRow<float>({
40.0, 99.0, 94.0
}),
MatrixRow<float>({
15.0, 98.0, 1900.0
}),
MatrixRow<float>({
15.0, 98.0, 380.0
}),
MatrixRow<float>({
40.0, 98.0, 50.0
}),
MatrixRow<float>({
22.0, 98.0, 102.0
}),
MatrixRow<float>({
15.0, 97.0, 170.0
}),
MatrixRow<float>({
22.0, 97.0, 220.0
}),
MatrixRow<float>({
22.0, 97.0, 215.0
}),
MatrixRow<float>({
40.0, 97.0, 150.0
}),
MatrixRow<float>({
15.0, 97.0, 1100.0 }),
MatrixRow<float>({
40.0, 97.0, 82.0
}),
MatrixRow<float>({
15.0, 96.0, 280.0
}),
MatrixRow<float>({
15.0, 96.0, 200.0
}),
MatrixRow<float>({
22.0, 96.0, 75.0
}),
MatrixRow<float>({
15.0, 96.0, 1200.0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 195.0
}),
MatrixRow<float>({
40.0, 96.0, 125.0
}),
MatrixRow<float>({
15.0, 96.0, 1300.0
}),
MatrixRow<float>({
15.0, 96.0, 400.0
}),
MatrixRow<float>({
40.0, 96.0, 78.0
}),
MatrixRow<float>({
15.0, 96.0, 75.0
}),
MatrixRow<float>({
40.0, 95.0, 56.0
}),
MatrixRow<float>({
15.0, 95.0, 89.0
}),
MatrixRow<float>({
15.0, 95.0, 163.0
}),
MatrixRow<float>({
15.0, 95.0, 110.0
}),
MatrixRow<float>({
40.0, 95.0, 60.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0 }),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
15.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0 }),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
31.0, 85.0, 6.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
31.0, 85.0, 7.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
31.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0 }),
MatrixRow<float>({
22.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 27.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
2.0, 85.0, 16.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
0.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 9.0 }),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 126.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
0.0, 90.0, 25.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
3.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 115.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 63.0 }),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
3.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
0.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 50.0
}),
MatrixRow<float>({
21.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 85.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
21.0, 86.0, 35.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
2.0, 86.0, 14.0
}),
MatrixRow<float>({
2.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 65.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
2.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 29.0
}),
MatrixRow<float>({
31.0, 85.0, 20.0
}),
MatrixRow<float>({
21.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0 }),
MatrixRow<float>({
31.0, 85.0, 11.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
2.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
37.0, 92.0, 47.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 90.0
}),
MatrixRow<float>({
15.0, 92.0, 135.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
22.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 56.0 }),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 49.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 25.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 19.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 37.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
37.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0 }),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 23.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 33.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
37.0, 91.0, 33.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
0.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0 }),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 29.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 11.0
}),
MatrixRow<float>({
37.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
31.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 20.0
}),
MatrixRow<float>({
37.0, 85.0, 9.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0 }),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 23.0
}),
MatrixRow<float>({
37.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 46.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
37.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
15.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 47.0
}),
MatrixRow<float>({
40.0, 84.0, 60.0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0 }),
MatrixRow<float>({
40.0, 83.0, 16.0
}),
MatrixRow<float>({
37.0, 83.0, 25.0
}),
MatrixRow<float>({
0.0, 83.0, 0
}),
MatrixRow<float>({
37.0, 83.0, 20.0
}),
MatrixRow<float>({
37.0, 83.0, 30.0
}),
MatrixRow<float>({
37.0, 83.0, 12.0
}),
MatrixRow<float>({
37.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 38.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
15.0, 83.0, 21.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
0.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 83.0, 55.0
}),
MatrixRow<float>({
40.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 29.0
}),
MatrixRow<float>({
22.0, 83.0, 30.0
}),
MatrixRow<float>({
15.0, 94.0, 180.0
}),
MatrixRow<float>({
15.0, 94.0, 74.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 300.0
}),
MatrixRow<float>({
22.0, 94.0, 130.0
}),
MatrixRow<float>({
22.0, 93.0, 120.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0 }),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
22.0, 93.0, 75.0
}),
MatrixRow<float>({
31.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
15.0, 93.0, 75.0
}),
MatrixRow<float>({
22.0, 93.0, 80.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 47.0
}),
MatrixRow<float>({
15.0, 93.0, 270.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 70.0
}),
MatrixRow<float>({
22.0, 93.0, 60.0
}),
MatrixRow<float>({
2.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 23.0
}),
MatrixRow<float>({
2.0, 89.0, 20.0
}),
MatrixRow<float>({
2.0, 89.0, 25.0 }),
MatrixRow<float>({
22.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 46.0
}),
MatrixRow<float>({
2.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
2.0, 89.0, 12.0
}),
MatrixRow<float>({
37.0, 89.0, 150.0
}),
MatrixRow<float>({
0.0, 89.0, 10.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 28.0
}),
MatrixRow<float>({
0.0, 89.0, 12.0
}),
MatrixRow<float>({
2.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
37.0, 89.0, 53.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
2.0, 88.0, 14.0
}),
MatrixRow<float>({
2.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
3.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
8.0, 87.0, 30.0
}),
MatrixRow<float>({
8.0, 87.0, 16.0 }),
MatrixRow<float>({
8.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
3.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
2.0, 87.0, 30.0
}),
MatrixRow<float>({
21.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 80.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
2.0, 87.0, 53.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
8.0, 87.0, 30.0
}),
MatrixRow<float>({
8.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
3.0, 87.0, 14.0 }),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
21.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0 }),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
37.0, 85.0, 4.0
}),
MatrixRow<float>({
37.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
0.0, 84.0, 15.0 }),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 19.0
}),
MatrixRow<float>({
22.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
31.0, 84.0, 20.0
}),
MatrixRow<float>({
31.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 19.0
}),
MatrixRow<float>({
15.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 17.0 }),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 39.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 70.0 }),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
22.0, 92.0, 82.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 85.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
3.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 48.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 76.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 17.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0 }),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 49.0
}),
MatrixRow<float>({
15.0, 90.0, 31.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 41.0
}),
MatrixRow<float>({
15.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
0.0, 90.0, 39.0
}),
MatrixRow<float>({
8.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
15.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0 }),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 36.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
18.0, 88.0, 79.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0 }),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 53.0
}),
MatrixRow<float>({
15.0, 88.0, 42.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 98.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 44.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
2.0, 84.0, 21.0
}),
MatrixRow<float>({
37.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 84.0, 8.0
}),
MatrixRow<float>({
37.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0 }),
MatrixRow<float>({
15.0, 84.0, 22.0
}),
MatrixRow<float>({
15.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
37.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
31.0, 84.0, 33.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
37.0, 84.0, 18.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 21.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
37.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
18.0, 90.0, 15.0 }),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 107.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 58.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
37.0, 90.0, 31.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
3.0, 90.0, 16.0
}),
MatrixRow<float>({
37.0, 90.0, 31.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
37.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
18.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 70.0
}),
MatrixRow<float>({
18.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0 }),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 62.0
}),
MatrixRow<float>({
40.0, 88.0, 85.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 53.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
29.0, 88.0, 17.0 }),
MatrixRow<float>({
15.0, 88.0, 21.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 94.0, 92.0
}),
MatrixRow<float>({
22.0, 94.0, 85.0
}),
MatrixRow<float>({
22.0, 94.0, 90.0
}),
MatrixRow<float>({
15.0, 94.0, 85.0
}),
MatrixRow<float>({
22.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 36.0
}),
MatrixRow<float>({
22.0, 94.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 64.0
}),
MatrixRow<float>({
22.0, 94.0, 120.0
}),
MatrixRow<float>({
22.0, 94.0, 73.0
}),
MatrixRow<float>({
22.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
22.0, 94.0, 45.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 46.0
}),
MatrixRow<float>({
22.0, 94.0, 106.0
}),
MatrixRow<float>({
22.0, 94.0, 72.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0 }),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
22.0, 94.0, 75.0
}),
MatrixRow<float>({
15.0, 94.0, 225.0
}),
MatrixRow<float>({
15.0, 94.0, 260.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
22.0, 94.0, 55.0
}),
MatrixRow<float>({
22.0, 94.0, 65.0
}),
MatrixRow<float>({
22.0, 94.0, 150.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
0.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
2.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
3.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0 }),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
12.0, 85.0, 25.0
}),
MatrixRow<float>({
31.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
0.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 34.0
}),
MatrixRow<float>({
37.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 32.0
}),
MatrixRow<float>({
12.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
0.0, 84.0, 17.0 }),
MatrixRow<float>({
35.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
17.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 66.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
17.0, 90.0, 47.0
}),
MatrixRow<float>({
17.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
22.0, 90.0, 58.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 53.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
37.0, 90.0, 54.0
}),
MatrixRow<float>({
17.0, 90.0, 0
}),
MatrixRow<float>({
17.0, 90.0, 27.0
}),
MatrixRow<float>({
2.0, 90.0, 33.0
}),
MatrixRow<float>({
22.0, 90.0, 37.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0 }),
MatrixRow<float>({
22.0, 90.0, 34.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 6.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0 }),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
37.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 28.0 }),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 85.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 47.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 26.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0 }),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 80.0
}),
MatrixRow<float>({
31.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
15.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0 }),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 40.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
8.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
3.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0 }),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
22.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 65.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 0 }),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 75.0
}),
MatrixRow<float>({
37.0, 90.0, 16.0
}),
MatrixRow<float>({
37.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
32.0, 88.0, 22.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0 }),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 43.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
32.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 44.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
37.0, 95.0, 300.0
}),
MatrixRow<float>({
22.0, 95.0, 65.0
}),
MatrixRow<float>({
22.0, 95.0, 60.0
}),
MatrixRow<float>({
22.0, 95.0, 96.0
}),
MatrixRow<float>({
22.0, 95.0, 59.0
}),
MatrixRow<float>({
40.0, 95.0, 32.0
}),
MatrixRow<float>({
22.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 43.0
}),
MatrixRow<float>({
22.0, 94.0, 42.0 }),
MatrixRow<float>({
2.0, 94.0, 225.0
}),
MatrixRow<float>({
22.0, 94.0, 48.0
}),
MatrixRow<float>({
22.0, 94.0, 70.0
}),
MatrixRow<float>({
22.0, 94.0, 80.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 46.0
}),
MatrixRow<float>({
22.0, 94.0, 80.0
}),
MatrixRow<float>({
22.0, 94.0, 64.0
}),
MatrixRow<float>({
22.0, 94.0, 75.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
3.0, 89.0, 10.0
}),
MatrixRow<float>({
3.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 70.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
29.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0 }),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
29.0, 89.0, 14.0
}),
MatrixRow<float>({
3.0, 89.0, 13.0
}),
MatrixRow<float>({
37.0, 89.0, 15.0
}),
MatrixRow<float>({
37.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
29.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 27.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0 }),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 33.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
2.0, 90.0, 35.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
17.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 92.0 }),
MatrixRow<float>({
22.0, 90.0, 82.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
21.0, 90.0, 39.0
}),
MatrixRow<float>({
17.0, 90.0, 29.0
}),
MatrixRow<float>({
2.0, 90.0, 20.0
}),
MatrixRow<float>({
17.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
8.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
8.0, 90.0, 23.0
}),
MatrixRow<float>({
8.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
2.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
17.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 11.0
}),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({
7.0, 83.0, 12.0 }),
MatrixRow<float>({
2.0, 83.0, 8.0
}),
MatrixRow<float>({
0.0, 83.0, 6.0
}),
MatrixRow<float>({
22.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 13.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 55.0
}),
MatrixRow<float>({
40.0, 82.0, 27.0
}),
MatrixRow<float>({
40.0, 82.0, 13.0
}),
MatrixRow<float>({
40.0, 82.0, 13.0
}),
MatrixRow<float>({
40.0, 82.0, 24.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
2.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 57.0
}),
MatrixRow<float>({
31.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
2.0, 90.0, 22.0
}),
MatrixRow<float>({
8.0, 90.0, 30.0 }),
MatrixRow<float>({
2.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 49.0
}),
MatrixRow<float>({
40.0, 89.0, 44.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
31.0, 89.0, 50.0
}),
MatrixRow<float>({
31.0, 89.0, 23.0
}),
MatrixRow<float>({
31.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 70.0
}),
MatrixRow<float>({
15.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 150.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 26.0 }),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
0.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 125.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 150.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0 }),
MatrixRow<float>({
26.0, 87.0, 18.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 95.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 9.0
}),
MatrixRow<float>({
8.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 11.0 }),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 7.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
22.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 70.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
8.0, 86.0, 23.0 }),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
32.0, 86.0, 9.0
}),
MatrixRow<float>({
32.0, 86.0, 9.0
}),
MatrixRow<float>({
32.0, 86.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 9.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 8.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0 }),
MatrixRow<float>({
37.0, 84.0, 52.0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
8.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 19.0
}),
MatrixRow<float>({
8.0, 83.0, 16.0
}),
MatrixRow<float>({
31.0, 84.0, 5.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0 }),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 60.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0 }),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
29.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0 }),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
37.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 15.0 }),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 37.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 91.0, 23.0
}),
MatrixRow<float>({
22.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0 }),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 48.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 26.0
}),
MatrixRow<float>({
22.0, 86.0, 65.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 47.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 41.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 30.0
}),
MatrixRow<float>({
17.0, 86.0, 42.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
37.0, 86.0, 21.0 }),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
31.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 35.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 47.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
8.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 75.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0 }),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
0.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 26.0
}),
MatrixRow<float>({
2.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 37.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
2.0, 86.0, 19.0
}),
MatrixRow<float>({
0.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
2.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
2.0, 86.0, 37.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 42.0 }),
MatrixRow<float>({
40.0, 85.0, 75.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
37.0, 85.0, 29.0
}),
MatrixRow<float>({
2.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 23.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 37.0
}),
MatrixRow<float>({
15.0, 91.0, 33.0
}),
MatrixRow<float>({
2.0, 91.0, 50.0
}),
MatrixRow<float>({
2.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
17.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0 }),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
2.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 66.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
17.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 43.0
}),
MatrixRow<float>({
37.0, 91.0, 55.0
}),
MatrixRow<float>({
37.0, 91.0, 21.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 36.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
37.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0 }),
MatrixRow<float>({
3.0, 92.0, 13.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 33.0
}),
MatrixRow<float>({
40.0, 92.0, 49.0
}),
MatrixRow<float>({
37.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
3.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 33.0
}),
MatrixRow<float>({
37.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
31.0, 88.0, 17.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 88.0, 9.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0 }),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
0.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
17.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
0.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 95.0, 60.0
}),
MatrixRow<float>({
15.0, 95.0, 164.0
}),
MatrixRow<float>({
40.0, 95.0, 36.0
}),
MatrixRow<float>({
22.0, 94.0, 49.0
}),
MatrixRow<float>({
40.0, 94.0, 25.0 }),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
15.0, 94.0, 190.0
}),
MatrixRow<float>({
15.0, 94.0, 205.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 145.0
}),
MatrixRow<float>({
15.0, 94.0, 89.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 36.0
}),
MatrixRow<float>({
2.0, 94.0, 105.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 55.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0 }),
MatrixRow<float>({
2.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 42.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 44.0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
2.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 145.0
}),
MatrixRow<float>({
8.0, 88.0, 40.0
}),
MatrixRow<float>({
8.0, 88.0, 24.0
}),
MatrixRow<float>({
8.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
8.0, 88.0, 49.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 52.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0 }),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
0.0, 84.0, 22.0
}),
MatrixRow<float>({
22.0, 84.0, 18.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 32.0
}),
MatrixRow<float>({
40.0, 83.0, 8.0
}),
MatrixRow<float>({
-1.0, 83.0, 0
}),
MatrixRow<float>({
37.0, 83.0, 11.0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
2.0, 89.0, 17.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 44.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 41.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
3.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
0.0, 89.0, 12.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
3.0, 89.0, 60.0
}),
MatrixRow<float>({ 40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
37.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
29.0, 86.0, 12.0
}),
MatrixRow<float>({
0.0, 86.0, 11.0
}),
MatrixRow<float>({
0.0, 86.0, 17.0
}),
MatrixRow<float>({
0.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 85.0
}),
MatrixRow<float>({
0.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 5.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 28.0
}),
MatrixRow<float>({ 15.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
0.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
3.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 90.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
3.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 21.0
}),
MatrixRow<float>({
22.0, 90.0, 400.0
}),
MatrixRow<float>({ 3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 45.0
}),
MatrixRow<float>({
37.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
37.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
37.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 60.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({ 40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
22.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
3.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
8.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 56.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({ 15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 31.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
3.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
29.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 45.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 50.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({ 2.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
0.0, 87.0, 46.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
2.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 42.0
}),
MatrixRow<float>({
36.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({ 37.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 80.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 37.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
21.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
21.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 45.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 47.0
}),
MatrixRow<float>({ 22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 29.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
15.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 29.0
}),
MatrixRow<float>({
39.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 40.0
}),
MatrixRow<float>({
15.0, 84.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({ 15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 44.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 92.0, 50.0
}),
MatrixRow<float>({
17.0, 92.0, 31.0
}),
MatrixRow<float>({
40.0, 92.0, 69.0
}),
MatrixRow<float>({
40.0, 92.0, 260.0
}),
MatrixRow<float>({
2.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
31.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
15.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({ 40.0, 92.0, 65.0
}),
MatrixRow<float>({
31.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
17.0, 92.0, 33.0
}),
MatrixRow<float>({
17.0, 92.0, 92.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
37.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
31.0, 92.0, 60.0
}),
MatrixRow<float>({
31.0, 92.0, 31.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
37.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
0.0, 84.0, 15.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({ 31.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 25.0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 40.0
}),
MatrixRow<float>({
37.0, 83.0, 60.0
}),
MatrixRow<float>({
31.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 55.0
}),
MatrixRow<float>({
31.0, 83.0, 8.0
}),
MatrixRow<float>({
22.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 9.0
}),
MatrixRow<float>({
37.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({ 31.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
37.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 72.0
}),
MatrixRow<float>({
31.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 88.0, 39.0
}),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 70.0
}),
MatrixRow<float>({
22.0, 88.0, 31.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
2.0, 88.0, 15.0
}),
MatrixRow<float>({ 0.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
2.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
2.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
18.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
22.0, 88.0, 29.0
}),
MatrixRow<float>({
37.0, 88.0, 60.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 88.0, 10.0
}),
MatrixRow<float>({
21.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({ 15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 67.0
}),
MatrixRow<float>({
31.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 33.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
17.0, 87.0, 42.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({ 40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 59.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
37.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 62.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 55.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({ 22.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
17.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 92.0, 92.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
2.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 18.0
}),
MatrixRow<float>({
2.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 39.0
}),
MatrixRow<float>({
3.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 120.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 56.0
}),
MatrixRow<float>({
2.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 17.0
}),
MatrixRow<float>({ 17.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
2.0, 92.0, 50.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
3.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
19.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
31.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
17.0, 89.0, 12.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
31.0, 89.0, 13.0
}),
MatrixRow<float>({
15.0, 89.0, 23.0
}),
MatrixRow<float>({
24.0, 89.0, 16.0
}),
MatrixRow<float>({ 40.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 33.0
}),
MatrixRow<float>({
0.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
17.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
31.0, 89.0, 26.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 29.0
}),
MatrixRow<float>({
21.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 50.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
2.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({ 15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 33.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
8.0, 82.0, 15.0
}),
MatrixRow<float>({
8.0, 82.0, 13.0
}),
MatrixRow<float>({
31.0, 82.0, 24.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 19.0
}),
MatrixRow<float>({
40.0, 82.0, 42.0
}),
MatrixRow<float>({
40.0, 82.0, 50.0
}),
MatrixRow<float>({ 8.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 22.0
}),
MatrixRow<float>({
8.0, 82.0, 20.0
}),
MatrixRow<float>({
8.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 81.0, 11.0
}),
MatrixRow<float>({
8.0, 81.0, 14.0
}),
MatrixRow<float>({
40.0, 81.0, 37.0
}),
MatrixRow<float>({
8.0, 81.0, 12.0
}),
MatrixRow<float>({
40.0, 81.0, 16.0
}),
MatrixRow<float>({
8.0, 81.0, 10.0
}),
MatrixRow<float>({
31.0, 80.0, 8.0
}),
MatrixRow<float>({
8.0, 80.0, 15.0
}),
MatrixRow<float>({
15.0, 97.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 170.0
}),
MatrixRow<float>({
22.0, 93.0, 52.0
}),
MatrixRow<float>({
3.0, 93.0, 67.0
}),
MatrixRow<float>({
3.0, 93.0, 56.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({ 3.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 85.0
}),
MatrixRow<float>({
0.0, 92.0, 64.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
8.0, 84.0, 14.0
}),
MatrixRow<float>({
29.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({ 40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 9.0
}),
MatrixRow<float>({
2.0, 84.0, 9.0
}),
MatrixRow<float>({
36.0, 84.0, 28.0
}),
MatrixRow<float>({
8.0, 84.0, 9.0
}),
MatrixRow<float>({
36.0, 84.0, 11.0
}),
MatrixRow<float>({
36.0, 84.0, 19.0
}),
MatrixRow<float>({
36.0, 84.0, 10.0
}),
MatrixRow<float>({
29.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 90.0
}),
MatrixRow<float>({
40.0, 91.0, 72.0
}),
MatrixRow<float>({
31.0, 91.0, 56.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 65.0
}),
MatrixRow<float>({
37.0, 91.0, 27.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
37.0, 91.0, 52.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 90.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
0.0, 91.0, 45.0
}),
MatrixRow<float>({ 17.0, 91.0, 95.0
}),
MatrixRow<float>({
0.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
17.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 27.0
}),
MatrixRow<float>({
0.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
17.0, 88.0, 55.0
}),
MatrixRow<float>({
17.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
37.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
31.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({ 22.0, 86.0, 22.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 21.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
18.0, 86.0, 23.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 90.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
8.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 7.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({ 15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 29.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
29.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 49.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 54.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({ 40.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
29.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
3.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 33.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
8.0, 88.0, 10.0
}),
MatrixRow<float>({ 40.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
37.0, 88.0, 44.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 93.0, 65.0
}),
MatrixRow<float>({
31.0, 93.0, 0
}),
MatrixRow<float>({
31.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
37.0, 93.0, 95.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 84.0
}),
MatrixRow<float>({
15.0, 93.0, 130.0
}),
MatrixRow<float>({
22.0, 93.0, 36.0
}),
MatrixRow<float>({ 40.0, 93.0, 39.0
}),
MatrixRow<float>({
40.0, 93.0, 79.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 93.0, 23.0
}),
MatrixRow<float>({
40.0, 93.0, 106.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 32.0
}),
MatrixRow<float>({
15.0, 93.0, 30.0
}),
MatrixRow<float>({
22.0, 93.0, 22.0
}),
MatrixRow<float>({
31.0, 93.0, 100.0
}),
MatrixRow<float>({
22.0, 93.0, 25.0
}),
MatrixRow<float>({
15.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 59.0
}),
MatrixRow<float>({
31.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 20.0
}),
MatrixRow<float>({
31.0, 93.0, 18.0
}),
MatrixRow<float>({
22.0, 93.0, 16.0
}),
MatrixRow<float>({
15.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 42.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
17.0, 92.0, 31.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({ 15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
8.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
17.0, 91.0, 84.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({ 15.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
2.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
2.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({ 22.0, 87.0, 16.0
}),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 31.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
0.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 47.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({ 22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
0.0, 83.0, 7.0
}),
MatrixRow<float>({
0.0, 83.0, 8.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 38.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 11.0
}),
MatrixRow<float>({
37.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
31.0, 83.0, 8.0
}),
MatrixRow<float>({
40.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 25.0
}),
MatrixRow<float>({
0.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 33.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 83.0, 12.0
}),
MatrixRow<float>({
31.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
2.0, 83.0, 15.0
}),
MatrixRow<float>({
31.0, 83.0, 6.0
}),
MatrixRow<float>({
2.0, 83.0, 10.0
}),
MatrixRow<float>({
31.0, 83.0, 5.0
}),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({ 2.0, 83.0, 9.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 105.0
}),
MatrixRow<float>({
8.0, 90.0, 32.0
}),
MatrixRow<float>({
8.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
0.0, 90.0, 17.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 140.0
}),
MatrixRow<float>({
15.0, 90.0, 95.0
}),
MatrixRow<float>({
40.0, 90.0, 195.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 47.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 98.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({ 22.0, 90.0, 37.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
29.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
29.0, 85.0, 25.0
}),
MatrixRow<float>({
3.0, 85.0, 48.0
}),
MatrixRow<float>({
22.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 26.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 21.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 96.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
37.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({ 0.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 33.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
10.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 14.0
}),
MatrixRow<float>({
36.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
29.0, 88.0, 18.0
}),
MatrixRow<float>({
37.0, 88.0, 24.0
}),
MatrixRow<float>({
29.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
10.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 11.0
}),
MatrixRow<float>({
36.0, 88.0, 15.0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({ 40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
16.0, 88.0, 12.0
}),
MatrixRow<float>({
29.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
42.0, 88.0, 43.0
}),
MatrixRow<float>({ 40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 60.0
}),
MatrixRow<float>({
0.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
14.0, 90.0, 46.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({ 40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 53.0
}),
MatrixRow<float>({
40.0, 90.0, 31.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
8.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
2.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 54.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({ 40.0, 86.0, 33.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
42.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 24.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 88.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 62.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
17.0, 90.0, 17.0
}),
MatrixRow<float>({
17.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
17.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 49.0
}),
MatrixRow<float>({
15.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 31.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
22.0, 90.0, 49.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({ 15.0, 90.0, 60.0
}),
MatrixRow<float>({
15.0, 90.0, 53.0
}),
MatrixRow<float>({
15.0, 90.0, 63.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
15.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 65.0
}),
MatrixRow<float>({
31.0, 90.0, 60.0
}),
MatrixRow<float>({
2.0, 90.0, 29.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({
31.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 66.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
22.0, 92.0, 35.0
}),
MatrixRow<float>({
31.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 95.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({ 40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 100.0
}),
MatrixRow<float>({
15.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 27.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
31.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 66.0
}),
MatrixRow<float>({
31.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
17.0, 92.0, 210.0
}),
MatrixRow<float>({
40.0, 87.0, 58.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({ 40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0
}),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 79.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
8.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 52.0
}),
MatrixRow<float>({ 22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
36.0, 87.0, 19.0
}),
MatrixRow<float>({
0.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
17.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
0.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
0.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({ 15.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
0.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
29.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
15.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 100.0
}),
MatrixRow<float>({
8.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 115.0
}),
MatrixRow<float>({
-1.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
8.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0 }),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
37.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
8.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
37.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0 }),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 49.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 31.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
18.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 21.0 }),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 43.0
}),
MatrixRow<float>({
29.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 90.0
}),
MatrixRow<float>({
8.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
31.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
8.0, 89.0, 18.0
}),
MatrixRow<float>({
37.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0 }),
MatrixRow<float>({
2.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
2.0, 88.0, 24.0
}),
MatrixRow<float>({
2.0, 88.0, 60.0
}),
MatrixRow<float>({
37.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 130.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0 }),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
22.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 60.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 52.0
}),
MatrixRow<float>({
15.0, 91.0, 37.0
}),
MatrixRow<float>({
22.0, 91.0, 47.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
0.0, 91.0, 120.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
0.0, 91.0, 16.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 75.0 }),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 32.0
}),
MatrixRow<float>({
37.0, 91.0, 85.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 69.0
}),
MatrixRow<float>({
15.0, 91.0, 69.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 42.0
}),
MatrixRow<float>({
15.0, 88.0, 42.0 }),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 31.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 88.0, 47.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
37.0, 89.0, 38.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 49.0
}),
MatrixRow<float>({
40.0, 89.0, 52.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
37.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
37.0, 89.0, 19.0 }),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 49.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 27.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 46.0
}),
MatrixRow<float>({
15.0, 89.0, 161.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 37.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 28.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0 }),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 48.0
}),
MatrixRow<float>({
37.0, 85.0, 45.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 90.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 65.0
}),
MatrixRow<float>({
40.0, 85.0, 85.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0 }),
MatrixRow<float>({
37.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 64.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
29.0, 91.0, 50.0
}),
MatrixRow<float>({
29.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 67.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 48.0
}),
MatrixRow<float>({
36.0, 89.0, 0 }),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 98.0, 155.0
}),
MatrixRow<float>({
22.0, 97.0, 60.0
}),
MatrixRow<float>({
40.0, 97.0, 50.0
}),
MatrixRow<float>({
40.0, 97.0, 115.0
}),
MatrixRow<float>({
40.0, 96.0, 70.0
}),
MatrixRow<float>({
40.0, 96.0, 115.0
}),
MatrixRow<float>({
40.0, 96.0, 90.0
}),
MatrixRow<float>({
40.0, 96.0, 115.0
}),
MatrixRow<float>({
40.0, 96.0, 58.0
}),
MatrixRow<float>({
40.0, 96.0, 69.0
}),
MatrixRow<float>({
40.0, 96.0, 88.0
}),
MatrixRow<float>({
22.0, 96.0, 146.0
}),
MatrixRow<float>({
40.0, 95.0, 72.0
}),
MatrixRow<float>({
22.0, 95.0, 96.0
}),
MatrixRow<float>({
40.0, 95.0, 44.0
}),
MatrixRow<float>({
40.0, 95.0, 45.0
}),
MatrixRow<float>({
40.0, 95.0, 47.0
}),
MatrixRow<float>({
15.0, 95.0, 60.0
}),
MatrixRow<float>({
40.0, 95.0, 90.0
}),
MatrixRow<float>({
40.0, 95.0, 48.0
}),
MatrixRow<float>({
22.0, 95.0, 225.0
}),
MatrixRow<float>({
22.0, 95.0, 30.0 }),
MatrixRow<float>({
22.0, 95.0, 40.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
15.0, 95.0, 40.0
}),
MatrixRow<float>({
22.0, 95.0, 30.0
}),
MatrixRow<float>({
8.0, 82.0, 9.0
}),
MatrixRow<float>({
40.0, 82.0, 32.0
}),
MatrixRow<float>({
40.0, 82.0, 32.0
}),
MatrixRow<float>({
40.0, 82.0, 36.0
}),
MatrixRow<float>({
37.0, 82.0, 11.0
}),
MatrixRow<float>({
40.0, 82.0, 11.0
}),
MatrixRow<float>({
22.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 50.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
22.0, 82.0, 21.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 17.0
}),
MatrixRow<float>({
15.0, 82.0, 0
}),
MatrixRow<float>({
22.0, 82.0, 21.0
}),
MatrixRow<float>({
40.0, 81.0, 25.0
}),
MatrixRow<float>({
40.0, 81.0, 14.0
}),
MatrixRow<float>({
40.0, 81.0, 32.0
}),
MatrixRow<float>({
15.0, 81.0, 12.0
}),
MatrixRow<float>({
8.0, 81.0, 10.0 }),
MatrixRow<float>({
40.0, 81.0, 11.0
}),
MatrixRow<float>({
40.0, 81.0, 11.0
}),
MatrixRow<float>({
40.0, 81.0, 14.0
}),
MatrixRow<float>({
37.0, 81.0, 7.0
}),
MatrixRow<float>({
40.0, 81.0, 14.0
}),
MatrixRow<float>({
8.0, 81.0, 13.0
}),
MatrixRow<float>({
37.0, 81.0, 12.0
}),
MatrixRow<float>({
22.0, 80.0, 12.0
}),
MatrixRow<float>({
37.0, 80.0, 14.0
}),
MatrixRow<float>({
37.0, 85.0, 7.0
}),
MatrixRow<float>({
31.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
31.0, 85.0, 21.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 60.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 25.0 }),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
37.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 26.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
37.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0 }),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 115.0
}),
MatrixRow<float>({
37.0, 87.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 17.0
}),
MatrixRow<float>({
37.0, 87.0, 35.0
}),
MatrixRow<float>({
37.0, 87.0, 41.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
8.0, 87.0, 8.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 94.0, 66.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 62.0
}),
MatrixRow<float>({
40.0, 94.0, 41.0
}),
MatrixRow<float>({
29.0, 94.0, 85.0 }),
MatrixRow<float>({
0.0, 94.0, 55.0
}),
MatrixRow<float>({
22.0, 94.0, 50.0
}),
MatrixRow<float>({
22.0, 94.0, 50.0
}),
MatrixRow<float>({
22.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 62.0
}),
MatrixRow<float>({
22.0, 94.0, 57.0
}),
MatrixRow<float>({
22.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 30.0
}),
MatrixRow<float>({
0.0, 94.0, 68.0
}),
MatrixRow<float>({
40.0, 94.0, 37.0
}),
MatrixRow<float>({
15.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
0.0, 94.0, 125.0
}),
MatrixRow<float>({
40.0, 94.0, 47.0
}),
MatrixRow<float>({
40.0, 94.0, 44.0
}),
MatrixRow<float>({
40.0, 94.0, 37.0
}),
MatrixRow<float>({
31.0, 94.0, 50.0
}),
MatrixRow<float>({
31.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 39.0
}),
MatrixRow<float>({
15.0, 94.0, 50.0
}),
MatrixRow<float>({
31.0, 94.0, 65.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0 }),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 17.0
}),
MatrixRow<float>({
0.0, 89.0, 16.0
}),
MatrixRow<float>({
36.0, 89.0, 27.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
2.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
17.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 115.0 }),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
17.0, 91.0, 29.0
}),
MatrixRow<float>({
17.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 59.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
8.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
37.0, 91.0, 17.0 }),
MatrixRow<float>({
37.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
17.0, 87.0, 16.0
}),
MatrixRow<float>({
0.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 90.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
17.0, 87.0, 16.0
}),
MatrixRow<float>({
17.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
17.0, 87.0, 10.0
}),
MatrixRow<float>({
2.0, 87.0, 14.0 }),
MatrixRow<float>({
21.0, 87.0, 15.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0 }),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 85.0
}),
MatrixRow<float>({
3.0, 90.0, 89.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
3.0, 90.0, 21.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
0.0, 90.0, 20.0 }),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
0.0, 90.0, 17.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
3.0, 90.0, 41.0
}),
MatrixRow<float>({
3.0, 90.0, 26.0
}),
MatrixRow<float>({
0.0, 90.0, 95.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
18.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
18.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
18.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0 }),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
18.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
0.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
3.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 48.0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
36.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 48.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0 }),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
8.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 10.0 }),
MatrixRow<float>({
17.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
17.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0 }),
MatrixRow<float>({
18.0, 87.0, 25.0
}),
MatrixRow<float>({
17.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 37.0
}),
MatrixRow<float>({
31.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
17.0, 87.0, 20.0
}),
MatrixRow<float>({
18.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
23.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 36.0 }),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 23.0
}),
MatrixRow<float>({
31.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 62.0
}),
MatrixRow<float>({
17.0, 91.0, 20.0
}),
MatrixRow<float>({
0.0, 91.0, 62.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 24.0
}),
MatrixRow<float>({
17.0, 91.0, 37.0
}),
MatrixRow<float>({
0.0, 91.0, 100.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 28.0
}),
MatrixRow<float>({
31.0, 91.0, 60.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
22.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 25.0 }),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
0.0, 86.0, 13.0
}),
MatrixRow<float>({
36.0, 86.0, 30.0
}),
MatrixRow<float>({
2.0, 86.0, 28.0
}),
MatrixRow<float>({
36.0, 86.0, 12.0
}),
MatrixRow<float>({
36.0, 86.0, 11.0
}),
MatrixRow<float>({
37.0, 86.0, 49.0
}),
MatrixRow<float>({
0.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
2.0, 86.0, 10.0
}),
MatrixRow<float>({
0.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
31.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0 }),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 69.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 7.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
37.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
37.0, 90.0, 35.0
}),
MatrixRow<float>({
0.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
8.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0 }),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
17.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 39.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 11.0
}),
MatrixRow<float>({
29.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
36.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 54.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
2.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
29.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0 }),
MatrixRow<float>({
8.0, 88.0, 19.0
}),
MatrixRow<float>({
17.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
36.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 33.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 95.0, 55.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 59.0
}),
MatrixRow<float>({
40.0, 95.0, 20.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 130.0
}),
MatrixRow<float>({
40.0, 94.0, 145.0
}),
MatrixRow<float>({
40.0, 94.0, 210.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 150.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0 }),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 230.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 130.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
31.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 27.0
}),
MatrixRow<float>({
15.0, 91.0, 53.0
}),
MatrixRow<float>({
15.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 54.0
}),
MatrixRow<float>({
8.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0 }),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 14.0
}),
MatrixRow<float>({
23.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
31.0, 91.0, 12.0
}),
MatrixRow<float>({
31.0, 91.0, 47.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 80.0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
0.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0 }),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
37.0, 87.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
33.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 90.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 14.0 }),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
3.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
8.0, 86.0, 41.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
3.0, 86.0, 13.0
}),
MatrixRow<float>({
3.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 45.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
3.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
3.0, 92.0, 62.0 }),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
3.0, 92.0, 47.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
7.0, 92.0, 32.0
}),
MatrixRow<float>({
15.0, 92.0, 43.0
}),
MatrixRow<float>({
8.0, 92.0, 120.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
3.0, 92.0, 19.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
8.0, 92.0, 26.0
}),
MatrixRow<float>({
3.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
3.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 8.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 70.0
}),
MatrixRow<float>({
15.0, 90.0, 51.0 }),
MatrixRow<float>({
2.0, 89.0, 33.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 6.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
2.0, 87.0, 45.0
}),
MatrixRow<float>({
2.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 85.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 24.0 }),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
3.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 51.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
0.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 54.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
0.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
17.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 33.0 }),
MatrixRow<float>({
31.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
21.0, 90.0, 27.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
17.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
17.0, 92.0, 26.0
}),
MatrixRow<float>({
37.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
17.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 135.0
}),
MatrixRow<float>({
40.0, 92.0, 125.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 150.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 33.0 }),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
29.0, 92.0, 61.0
}),
MatrixRow<float>({
21.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
31.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 104.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
15.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
37.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 57.0
}),
MatrixRow<float>({
15.0, 91.0, 23.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 31.0 }),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
37.0, 91.0, 24.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
7.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 72.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
18.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0 }),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
3.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
36.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
36.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
2.0, 88.0, 13.0
}),
MatrixRow<float>({
2.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
2.0, 88.0, 43.0
}),
MatrixRow<float>({
2.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
0.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0 }),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
2.0, 88.0, 10.0
}),
MatrixRow<float>({
36.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
0.0, 88.0, 12.0
}),
MatrixRow<float>({
2.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 36.0
}),
MatrixRow<float>({
36.0, 88.0, 13.0
}),
MatrixRow<float>({
0.0, 88.0, 29.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
0.0, 88.0, 50.0
}),
MatrixRow<float>({
0.0, 88.0, 17.0
}),
MatrixRow<float>({
37.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
3.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0 }),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
3.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
3.0, 89.0, 19.0 }),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
3.0, 89.0, 21.0
}),
MatrixRow<float>({
29.0, 89.0, 25.0
}),
MatrixRow<float>({
29.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
3.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
29.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
29.0, 89.0, 90.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 55.0
}),
MatrixRow<float>({
29.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0 }),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 23.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 48.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
17.0, 85.0, 27.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 23.0
}),
MatrixRow<float>({
37.0, 85.0, 13.0
}),
MatrixRow<float>({
31.0, 85.0, 18.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0 }),
MatrixRow<float>({
40.0, 85.0, 37.0
}),
MatrixRow<float>({
22.0, 85.0, 29.0
}),
MatrixRow<float>({
37.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 125.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
37.0, 85.0, 25.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
19.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
0.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
31.0, 88.0, 13.0
}),
MatrixRow<float>({
0.0, 88.0, 17.0
}),
MatrixRow<float>({
31.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 90.0
}),
MatrixRow<float>({
15.0, 88.0, 21.0
}),
MatrixRow<float>({
16.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
2.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 43.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0 }),
MatrixRow<float>({
22.0, 88.0, 44.0
}),
MatrixRow<float>({
0.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
0.0, 88.0, 64.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
0.0, 88.0, 10.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 52.0
}),
MatrixRow<float>({
0.0, 88.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 60.0 }),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
31.0, 88.0, 58.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 29.0
}),
MatrixRow<float>({
22.0, 88.0, 65.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 65.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
37.0, 86.0, 24.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0 }),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
8.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
15.0, 86.0, 27.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 148.0
}),
MatrixRow<float>({
3.0, 93.0, 0
}),
MatrixRow<float>({
3.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 62.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
3.0, 93.0, 0
}),
MatrixRow<float>({
37.0, 93.0, 90.0 }),
MatrixRow<float>({
15.0, 93.0, 22.0
}),
MatrixRow<float>({
40.0, 93.0, 125.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 90.0
}),
MatrixRow<float>({
3.0, 93.0, 28.0
}),
MatrixRow<float>({
3.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
3.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
3.0, 93.0, 17.0
}),
MatrixRow<float>({
3.0, 93.0, 32.0
}),
MatrixRow<float>({
22.0, 93.0, 209.0
}),
MatrixRow<float>({
0.0, 93.0, 60.0
}),
MatrixRow<float>({
3.0, 93.0, 35.0
}),
MatrixRow<float>({
3.0, 93.0, 47.0
}),
MatrixRow<float>({
8.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
31.0, 83.0, 10.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
31.0, 83.0, 14.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 8.0
}),
MatrixRow<float>({
37.0, 83.0, 17.0 }),
MatrixRow<float>({
22.0, 83.0, 11.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 22.0
}),
MatrixRow<float>({
40.0, 82.0, 7.0
}),
MatrixRow<float>({
37.0, 82.0, 16.0
}),
MatrixRow<float>({
8.0, 82.0, 17.0
}),
MatrixRow<float>({
37.0, 82.0, 18.0
}),
MatrixRow<float>({
8.0, 82.0, 9.0
}),
MatrixRow<float>({
8.0, 82.0, 13.0
}),
MatrixRow<float>({
37.0, 82.0, 17.0
}),
MatrixRow<float>({
8.0, 82.0, 16.0
}),
MatrixRow<float>({
40.0, 82.0, 25.0
}),
MatrixRow<float>({
40.0, 82.0, 11.0
}),
MatrixRow<float>({
8.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 13.0
}),
MatrixRow<float>({
37.0, 82.0, 12.0
}),
MatrixRow<float>({
37.0, 82.0, 9.0
}),
MatrixRow<float>({
8.0, 82.0, 10.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
37.0, 82.0, 14.0
}),
MatrixRow<float>({
17.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0 }),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 33.0
}),
MatrixRow<float>({
0.0, 90.0, 21.0
}),
MatrixRow<float>({
15.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
36.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
3.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 37.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
0.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
17.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
0.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 24.0 }),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 19.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
7.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
0.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
0.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
17.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0 }),
MatrixRow<float>({
17.0, 88.0, 67.0
}),
MatrixRow<float>({
40.0, 88.0, 52.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 9.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
22.0, 88.0, 61.0
}),
MatrixRow<float>({
22.0, 88.0, 43.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
8.0, 88.0, 13.0
}),
MatrixRow<float>({
17.0, 88.0, 58.0
}),
MatrixRow<float>({
22.0, 88.0, 47.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({
2.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 58.0
}),
MatrixRow<float>({
37.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 27.0 }),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
3.0, 90.0, 52.0
}),
MatrixRow<float>({
37.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
37.0, 90.0, 145.0
}),
MatrixRow<float>({
22.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
3.0, 90.0, 35.0 }),
MatrixRow<float>({
22.0, 90.0, 300.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
3.0, 90.0, 21.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
7.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 58.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
3.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
3.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 43.0 }),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
3.0, 88.0, 23.0
}),
MatrixRow<float>({
3.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 57.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 29.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0 }),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
3.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
0.0, 91.0, 95.0
}),
MatrixRow<float>({
3.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 78.0
}),
MatrixRow<float>({
22.0, 91.0, 160.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0 }),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 200.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 100.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 52.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
3.0, 90.0, 13.0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0 }),
MatrixRow<float>({
29.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 160.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
32.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 49.0
}),
MatrixRow<float>({
8.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0 }),
MatrixRow<float>({
22.0, 92.0, 99.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 79.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 64.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
15.0, 83.0, 9.0
}),
MatrixRow<float>({
15.0, 83.0, 16.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 20.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
15.0, 83.0, 5.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
37.0, 83.0, 13.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
15.0, 83.0, 9.0
}),
MatrixRow<float>({
15.0, 83.0, 6.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 26.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 83.0, 19.0 }),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
0.0, 83.0, 8.0
}),
MatrixRow<float>({
15.0, 83.0, 5.0
}),
MatrixRow<float>({
15.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 50.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 22.0
}),
MatrixRow<float>({
15.0, 83.0, 9.0
}),
MatrixRow<float>({
15.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 26.0
}),
MatrixRow<float>({
31.0, 84.0, 11.0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
31.0, 84.0, 36.0
}),
MatrixRow<float>({
15.0, 84.0, 22.0
}),
MatrixRow<float>({
37.0, 84.0, 7.0
}),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({
31.0, 84.0, 19.0
}),
MatrixRow<float>({
37.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 9.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
31.0, 84.0, 11.0
}),
MatrixRow<float>({
37.0, 84.0, 23.0 }),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 24.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
31.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 8.0
}),
MatrixRow<float>({
2.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 18.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 70.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0 }),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
5.0, 88.0, 36.0
}),
MatrixRow<float>({
15.0, 88.0, 9.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 33.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
2.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 49.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0 }),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 47.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
17.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 95.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
37.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 37.0
}),
MatrixRow<float>({
40.0, 88.0, 65.0
}),
MatrixRow<float>({
17.0, 87.0, 12.0
}),
MatrixRow<float>({
0.0, 82.0, 15.0
}),
MatrixRow<float>({
0.0, 82.0, 11.0
}),
MatrixRow<float>({
40.0, 82.0, 40.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
0.0, 82.0, 10.0
}),
MatrixRow<float>({
0.0, 82.0, 6.0
}),
MatrixRow<float>({
40.0, 82.0, 26.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0 }),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
0.0, 81.0, 12.0
}),
MatrixRow<float>({
0.0, 81.0, 15.0
}),
MatrixRow<float>({
0.0, 81.0, 8.0
}),
MatrixRow<float>({
0.0, 81.0, 13.0
}),
MatrixRow<float>({
31.0, 81.0, 16.0
}),
MatrixRow<float>({
0.0, 80.0, 17.0
}),
MatrixRow<float>({
0.0, 80.0, 20.0
}),
MatrixRow<float>({
0.0, 80.0, 15.0
}),
MatrixRow<float>({
0.0, 80.0, 12.0
}),
MatrixRow<float>({
0.0, 80.0, 12.0
}),
MatrixRow<float>({
15.0, 97.0, 430.0
}),
MatrixRow<float>({
15.0, 97.0, 550.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
29.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 80.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0 }),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 33.0
}),
MatrixRow<float>({
15.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 0 }),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 7.0
}),
MatrixRow<float>({
8.0, 84.0, 17.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
18.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 22.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
22.0, 83.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 27.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 13.0 }),
MatrixRow<float>({
22.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
2.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 37.0
}),
MatrixRow<float>({
35.0, 87.0, 7.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
27.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0 }),
MatrixRow<float>({
22.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0
}),
MatrixRow<float>({
42.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 190.0
}),
MatrixRow<float>({
31.0, 90.0, 26.0
}),
MatrixRow<float>({
28.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
31.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
8.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0 }),
MatrixRow<float>({
31.0, 90.0, 10.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 120.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 48.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
2.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0 }),
MatrixRow<float>({
2.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 44.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
21.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
37.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0 }),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 11.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
3.0, 89.0, 12.0
}),
MatrixRow<float>({
37.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
21.0, 89.0, 15.0
}),
MatrixRow<float>({
21.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
0.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 48.0
}),
MatrixRow<float>({
0.0, 87.0, 10.0
}),
MatrixRow<float>({
0.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
0.0, 87.0, 18.0
}),
MatrixRow<float>({
0.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0 }),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
31.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 37.0
}),
MatrixRow<float>({
22.0, 87.0, 39.0
}),
MatrixRow<float>({
40.0, 87.0, 80.0
}),
MatrixRow<float>({
12.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
37.0, 92.0, 35.0
}),
MatrixRow<float>({
37.0, 92.0, 41.0
}),
MatrixRow<float>({
22.0, 92.0, 110.0
}),
MatrixRow<float>({
21.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0 }),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 64.0
}),
MatrixRow<float>({
40.0, 92.0, 56.0
}),
MatrixRow<float>({
15.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
31.0, 92.0, 28.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
22.0, 92.0, 53.0
}),
MatrixRow<float>({
40.0, 92.0, 27.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0 }),
MatrixRow<float>({
15.0, 89.0, 70.0
}),
MatrixRow<float>({
29.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
29.0, 89.0, 23.0
}),
MatrixRow<float>({
2.0, 89.0, 32.0
}),
MatrixRow<float>({
8.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 31.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 21.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 63.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0 }),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
37.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0 }),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 25.0
}),
MatrixRow<float>({
0.0, 85.0, 14.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 37.0
}),
MatrixRow<float>({
15.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
31.0, 85.0, 22.0 }),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 21.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
31.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
17.0, 90.0, 47.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
17.0, 90.0, 42.0
}),
MatrixRow<float>({
8.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0 }),
MatrixRow<float>({
31.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
31.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
31.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 95.0, 89.0
}),
MatrixRow<float>({
40.0, 95.0, 53.0
}),
MatrixRow<float>({
40.0, 95.0, 140.0
}),
MatrixRow<float>({
40.0, 95.0, 55.0
}),
MatrixRow<float>({
40.0, 95.0, 80.0
}),
MatrixRow<float>({
40.0, 95.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 18.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 150.0
}),
MatrixRow<float>({
40.0, 94.0, 120.0
}),
MatrixRow<float>({
17.0, 94.0, 58.0
}),
MatrixRow<float>({
40.0, 94.0, 90.0 }),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 140.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
0.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
31.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0 }),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
31.0, 83.0, 17.0
}),
MatrixRow<float>({
15.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 33.0
}),
MatrixRow<float>({
22.0, 83.0, 22.0
}),
MatrixRow<float>({
22.0, 83.0, 22.0
}),
MatrixRow<float>({
0.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 26.0
}),
MatrixRow<float>({
22.0, 83.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 18.0
}),
MatrixRow<float>({
15.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
22.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
29.0, 91.0, 50.0 }),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0
}),
MatrixRow<float>({
36.0, 91.0, 50.0
}),
MatrixRow<float>({
36.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 34.0
}),
MatrixRow<float>({
29.0, 91.0, 25.0
}),
MatrixRow<float>({
29.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 69.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 23.0
}),
MatrixRow<float>({
15.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 12.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 95.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 21.0
}),
MatrixRow<float>({
17.0, 93.0, 38.0
}),
MatrixRow<float>({
31.0, 93.0, 99.0 }),
MatrixRow<float>({
40.0, 93.0, 95.0
}),
MatrixRow<float>({
40.0, 93.0, 69.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
22.0, 93.0, 23.0
}),
MatrixRow<float>({
22.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
2.0, 93.0, 30.0
}),
MatrixRow<float>({
22.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
0.0, 93.0, 65.0
}),
MatrixRow<float>({
22.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
2.0, 93.0, 40.0 }),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
36.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
0.0, 87.0, 23.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
0.0, 86.0, 13.0
}),
MatrixRow<float>({
36.0, 86.0, 12.0
}),
MatrixRow<float>({
0.0, 86.0, 39.0
}),
MatrixRow<float>({
22.0, 86.0, 28.0
}),
MatrixRow<float>({
0.0, 86.0, 12.0 }),
MatrixRow<float>({
31.0, 86.0, 7.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
36.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
3.0, 85.0, 22.0
}),
MatrixRow<float>({
3.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
29.0, 85.0, 17.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
29.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 60.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 96.0, 475.0 }),
MatrixRow<float>({
22.0, 96.0, 73.0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 155.0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 132.0
}),
MatrixRow<float>({
15.0, 95.0, 235.0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
40.0, 95.0, 145.0
}),
MatrixRow<float>({
40.0, 95.0, 165.0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
22.0, 95.0, 115.0
}),
MatrixRow<float>({
37.0, 95.0, 200.0
}),
MatrixRow<float>({
15.0, 95.0, 90.0
}),
MatrixRow<float>({
22.0, 95.0, 150.0
}),
MatrixRow<float>({
15.0, 95.0, 375.0
}),
MatrixRow<float>({
15.0, 95.0, 160.0
}),
MatrixRow<float>({
15.0, 95.0, 398.0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 130.0
}),
MatrixRow<float>({
22.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 200.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0 }),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 95.0
}),
MatrixRow<float>({
15.0, 91.0, 54.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 94.0
}),
MatrixRow<float>({
15.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 80.0
}),
MatrixRow<float>({
37.0, 91.0, 50.0
}),
MatrixRow<float>({
7.0, 91.0, 80.0
}),
MatrixRow<float>({
22.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 56.0
}),
MatrixRow<float>({
22.0, 91.0, 54.0
}),
MatrixRow<float>({
15.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
22.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0 }),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
31.0, 83.0, 10.0
}),
MatrixRow<float>({
31.0, 83.0, 25.0
}),
MatrixRow<float>({
0.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
22.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
31.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 35.0
}),
MatrixRow<float>({
40.0, 82.0, 40.0
}),
MatrixRow<float>({
37.0, 82.0, 20.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
0.0, 82.0, 11.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
22.0, 82.0, 37.0
}),
MatrixRow<float>({
37.0, 82.0, 23.0 }),
MatrixRow<float>({
22.0, 82.0, 15.0
}),
MatrixRow<float>({
0.0, 82.0, 11.0
}),
MatrixRow<float>({
31.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 35.0
}),
MatrixRow<float>({
31.0, 82.0, 30.0
}),
MatrixRow<float>({
0.0, 82.0, 15.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
37.0, 81.0, 10.0
}),
MatrixRow<float>({
37.0, 81.0, 15.0
}),
MatrixRow<float>({
0.0, 81.0, 15.0
}),
MatrixRow<float>({
37.0, 81.0, 29.0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 13.0
}),
MatrixRow<float>({
15.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
40.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
15.0, 83.0, 21.0
}),
MatrixRow<float>({
15.0, 83.0, 18.0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 19.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0 }),
MatrixRow<float>({
40.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 35.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 83.0, 36.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 83.0, 13.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 8.0
}),
MatrixRow<float>({
40.0, 83.0, 8.0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
15.0, 83.0, 23.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0 }),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
8.0, 85.0, 18.0
}),
MatrixRow<float>({
30.0, 85.0, 14.0
}),
MatrixRow<float>({
29.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 44.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 94.0, 67.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 150.0
}),
MatrixRow<float>({
15.0, 94.0, 99.0
}),
MatrixRow<float>({
15.0, 94.0, 273.0 }),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
19.0, 94.0, 175.0
}),
MatrixRow<float>({
15.0, 94.0, 117.0
}),
MatrixRow<float>({
15.0, 94.0, 152.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 84.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
15.0, 93.0, 115.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
36.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 0 }),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 31.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
29.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 26.0
}),
MatrixRow<float>({
36.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
29.0, 86.0, 15.0
}),
MatrixRow<float>({
36.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
15.0, 88.0, 46.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 0 }),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 50.0
}),
MatrixRow<float>({
37.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 24.0
}),
MatrixRow<float>({
26.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 21.0
}),
MatrixRow<float>({
22.0, 90.0, 19.0
}),
MatrixRow<float>({
3.0, 90.0, 44.0
}),
MatrixRow<float>({
2.0, 90.0, 19.0 }),
MatrixRow<float>({
29.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 125.0
}),
MatrixRow<float>({
0.0, 90.0, 20.0
}),
MatrixRow<float>({
0.0, 90.0, 24.0
}),
MatrixRow<float>({
2.0, 90.0, 21.0
}),
MatrixRow<float>({
29.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 63.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
20.0, 93.0, 20.0 }),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
31.0, 93.0, 55.0
}),
MatrixRow<float>({
15.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 500.0
}),
MatrixRow<float>({
40.0, 93.0, 145.0
}),
MatrixRow<float>({
40.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
22.0, 93.0, 105.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
15.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 39.0
}),
MatrixRow<float>({
40.0, 93.0, 200.0
}),
MatrixRow<float>({
22.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 45.0
}),
MatrixRow<float>({
22.0, 93.0, 49.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
22.0, 93.0, 58.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 53.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0 }),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
14.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 41.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
8.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 18.0
}),
MatrixRow<float>({
15.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
22.0, 91.0, 53.0
}),
MatrixRow<float>({
15.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0 }),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
37.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 53.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 160.0
}),
MatrixRow<float>({
15.0, 94.0, 285.0
}),
MatrixRow<float>({
15.0, 94.0, 280.0
}),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
3.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 36.0
}),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
22.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0 }),
MatrixRow<float>({
40.0, 89.0, 21.0
}),
MatrixRow<float>({
22.0, 89.0, 48.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
0.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 10.0
}),
MatrixRow<float>({
22.0, 89.0, 80.0
}),
MatrixRow<float>({
3.0, 89.0, 38.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
0.0, 89.0, 65.0
}),
MatrixRow<float>({
0.0, 89.0, 42.0
}),
MatrixRow<float>({
37.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 70.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0 }),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 29.0
}),
MatrixRow<float>({
15.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
36.0, 89.0, 12.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
36.0, 89.0, 24.0
}),
MatrixRow<float>({
36.0, 89.0, 36.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 61.0
}),
MatrixRow<float>({
40.0, 89.0, 8.0
}),
MatrixRow<float>({
37.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
36.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 24.0
}),
MatrixRow<float>({
36.0, 89.0, 15.0
}),
MatrixRow<float>({
36.0, 89.0, 24.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 74.0
}),
MatrixRow<float>({
37.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
22.0, 89.0, 190.0 }),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 52.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
36.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 39.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
0.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
37.0, 88.0, 55.0
}),
MatrixRow<float>({
37.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 120.0 }),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
37.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
0.0, 91.0, 25.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 80.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
37.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 160.0
}),
MatrixRow<float>({
36.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
0.0, 91.0, 23.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 55.0 }),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 99.0
}),
MatrixRow<float>({
40.0, 91.0, 33.0
}),
MatrixRow<float>({
15.0, 89.0, 65.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 26.0
}),
MatrixRow<float>({
0.0, 89.0, 27.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
7.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
17.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 95.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 28.0
}),
MatrixRow<float>({
29.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 67.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0 }),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
17.0, 89.0, 13.0
}),
MatrixRow<float>({
29.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
17.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
17.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 33.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0 }),
MatrixRow<float>({
0.0, 87.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 250.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
17.0, 87.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 17.0
}),
MatrixRow<float>({
0.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 93.0, 70.0
}),
MatrixRow<float>({
31.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 23.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
37.0, 93.0, 46.0
}),
MatrixRow<float>({
22.0, 93.0, 29.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
31.0, 93.0, 30.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 64.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0 }),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 175.0
}),
MatrixRow<float>({
31.0, 93.0, 20.0
}),
MatrixRow<float>({
31.0, 93.0, 23.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
17.0, 93.0, 40.0
}),
MatrixRow<float>({
31.0, 93.0, 18.0
}),
MatrixRow<float>({
31.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 135.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
22.0, 93.0, 26.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
31.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 49.0
}),
MatrixRow<float>({
31.0, 92.0, 40.0
}),
MatrixRow<float>({
2.0, 89.0, 30.0
}),
MatrixRow<float>({
36.0, 89.0, 11.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
2.0, 89.0, 35.0
}),
MatrixRow<float>({
17.0, 89.0, 38.0
}),
MatrixRow<float>({
17.0, 89.0, 17.0
}),
MatrixRow<float>({
17.0, 89.0, 30.0
}),
MatrixRow<float>({
36.0, 89.0, 15.0 }),
MatrixRow<float>({
40.0, 89.0, 37.0
}),
MatrixRow<float>({
22.0, 89.0, 55.0
}),
MatrixRow<float>({
22.0, 89.0, 47.0
}),
MatrixRow<float>({
17.0, 89.0, 24.0
}),
MatrixRow<float>({
37.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
17.0, 89.0, 45.0
}),
MatrixRow<float>({
17.0, 89.0, 43.0
}),
MatrixRow<float>({
17.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
42.0, 86.0, 23.0
}),
MatrixRow<float>({
6.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0 }),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 28.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
2.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
2.0, 86.0, 8.0
}),
MatrixRow<float>({
32.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
35.0, 89.0, 35.0
}),
MatrixRow<float>({
35.0, 89.0, 35.0 }),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 47.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 59.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
8.0, 89.0, 23.0
}),
MatrixRow<float>({
31.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
15.0, 89.0, 54.0
}),
MatrixRow<float>({
15.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 52.0 }),
MatrixRow<float>({
22.0, 92.0, 95.0
}),
MatrixRow<float>({
29.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
29.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 95.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
37.0, 92.0, 22.0
}),
MatrixRow<float>({
29.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
17.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
31.0, 84.0, 6.0
}),
MatrixRow<float>({
31.0, 84.0, 7.0
}),
MatrixRow<float>({
0.0, 84.0, 14.0
}),
MatrixRow<float>({
0.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0 }),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 64.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 20.0
}),
MatrixRow<float>({
37.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
31.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
2.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0 }),
MatrixRow<float>({
40.0, 83.0, 28.0
}),
MatrixRow<float>({
0.0, 83.0, 14.0
}),
MatrixRow<float>({
0.0, 83.0, 29.0
}),
MatrixRow<float>({
40.0, 82.0, 9.0
}),
MatrixRow<float>({
40.0, 82.0, 45.0
}),
MatrixRow<float>({
15.0, 82.0, 0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
2.0, 82.0, 41.0
}),
MatrixRow<float>({
2.0, 82.0, 30.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 27.0
}),
MatrixRow<float>({
8.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 75.0
}),
MatrixRow<float>({
40.0, 82.0, 22.0
}),
MatrixRow<float>({
0.0, 82.0, 15.0
}),
MatrixRow<float>({
8.0, 81.0, 13.0
}),
MatrixRow<float>({
8.0, 81.0, 12.0
}),
MatrixRow<float>({
0.0, 81.0, 7.0
}),
MatrixRow<float>({
0.0, 80.0, 18.0
}),
MatrixRow<float>({
0.0, 80.0, 12.0
}),
MatrixRow<float>({
8.0, 80.0, 12.0
}),
MatrixRow<float>({
40.0, 97.0, 110.0
}),
MatrixRow<float>({
31.0, 87.0, 25.0 }),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 31.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
15.0, 87.0, 57.0
}),
MatrixRow<float>({
31.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 125.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
36.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 75.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0 }),
MatrixRow<float>({
37.0, 87.0, 17.0
}),
MatrixRow<float>({
7.0, 87.0, 49.0
}),
MatrixRow<float>({
0.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
31.0, 88.0, 16.0
}),
MatrixRow<float>({
8.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 48.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
7.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 75.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 55.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0 }),
MatrixRow<float>({
23.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 16.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 19.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 83.0, 47.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
22.0, 83.0, 81.0
}),
MatrixRow<float>({
22.0, 83.0, 27.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
0.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 37.0
}),
MatrixRow<float>({
29.0, 90.0, 14.0
}),
MatrixRow<float>({
21.0, 90.0, 20.0
}),
MatrixRow<float>({
21.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
22.0, 90.0, 42.0 }),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
36.0, 90.0, 36.0
}),
MatrixRow<float>({
36.0, 90.0, 35.0
}),
MatrixRow<float>({
36.0, 90.0, 80.0
}),
MatrixRow<float>({
37.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 49.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
29.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
29.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 80.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0 }),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
0.0, 90.0, 42.0
}),
MatrixRow<float>({
31.0, 90.0, 25.0
}),
MatrixRow<float>({
17.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 41.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
31.0, 90.0, 66.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0 }),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 110.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 32.0
}),
MatrixRow<float>({
36.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 19.0 }),
MatrixRow<float>({
37.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
19.0, 89.0, 53.0
}),
MatrixRow<float>({
19.0, 89.0, 27.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 9.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
32.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
8.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
12.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0 }),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 10.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
19.0, 89.0, 42.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 75.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 36.0
}),
MatrixRow<float>({
17.0, 90.0, 12.0
}),
MatrixRow<float>({
17.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 58.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 56.0 }),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 10.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 82.0, 27.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 16.0
}),
MatrixRow<float>({
31.0, 82.0, 7.0
}),
MatrixRow<float>({
37.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 13.0
}),
MatrixRow<float>({
40.0, 82.0, 35.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
22.0, 81.0, 10.0
}),
MatrixRow<float>({
31.0, 81.0, 8.0
}),
MatrixRow<float>({
21.0, 81.0, 35.0 }),
MatrixRow<float>({
15.0, 81.0, 13.0
}),
MatrixRow<float>({
21.0, 81.0, 22.0
}),
MatrixRow<float>({
37.0, 81.0, 13.0
}),
MatrixRow<float>({
40.0, 81.0, 25.0
}),
MatrixRow<float>({
40.0, 81.0, 15.0
}),
MatrixRow<float>({
40.0, 81.0, 16.0
}),
MatrixRow<float>({
40.0, 81.0, 44.0
}),
MatrixRow<float>({
40.0, 81.0, 10.0
}),
MatrixRow<float>({
37.0, 80.0, 19.0
}),
MatrixRow<float>({
40.0, 80.0, 10.0
}),
MatrixRow<float>({
40.0, 80.0, 19.0
}),
MatrixRow<float>({
22.0, 100.0, 210.0
}),
MatrixRow<float>({
40.0, 97.0, 70.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
3.0, 88.0, 18.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 70.0
}),
MatrixRow<float>({
37.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0 }),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
0.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
22.0, 88.0, 60.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 49.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
15.0, 92.0, 70.0
}),
MatrixRow<float>({
37.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
36.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 21.0 }),
MatrixRow<float>({
3.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 66.0
}),
MatrixRow<float>({
40.0, 92.0, 68.0
}),
MatrixRow<float>({
3.0, 92.0, 34.0
}),
MatrixRow<float>({
36.0, 92.0, 55.0
}),
MatrixRow<float>({
37.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 14.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
2.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
37.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 30.0 }),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 7.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 27.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
36.0, 86.0, 22.0 }),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
36.0, 86.0, 33.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
36.0, 86.0, 22.0
}),
MatrixRow<float>({
36.0, 86.0, 34.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
2.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
36.0, 86.0, 10.0
}),
MatrixRow<float>({
36.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
36.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
29.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
2.0, 85.0, 20.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0 }),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 36.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 35.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 78.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
22.0, 86.0, 60.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0 }),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 48.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 32.0
}),
MatrixRow<float>({
40.0, 83.0, 40.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 38.0
}),
MatrixRow<float>({
2.0, 83.0, 7.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 19.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 12.0
}),
MatrixRow<float>({
37.0, 83.0, 19.0
}),
MatrixRow<float>({
40.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 28.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
0.0, 83.0, 6.0 }),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
7.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
0.0, 92.0, 62.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
0.0, 92.0, 135.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
7.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 150.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
2.0, 92.0, 42.0
}),
MatrixRow<float>({
15.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 63.0
}),
MatrixRow<float>({
15.0, 92.0, 54.0
}),
MatrixRow<float>({
31.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0 }),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 95.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0 }),
MatrixRow<float>({
22.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
31.0, 91.0, 12.0
}),
MatrixRow<float>({
37.0, 91.0, 23.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 17.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 27.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
32.0, 85.0, 12.0
}),
MatrixRow<float>({
11.0, 85.0, 17.0
}),
MatrixRow<float>({
32.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0 }),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 19.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 41.0
}),
MatrixRow<float>({
18.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
32.0, 85.0, 8.0
}),
MatrixRow<float>({
0.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
18.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
8.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
8.0, 83.0, 7.0
}),
MatrixRow<float>({
8.0, 83.0, 6.0
}),
MatrixRow<float>({
8.0, 83.0, 6.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
22.0, 83.0, 12.0 }),
MatrixRow<float>({
40.0, 83.0, 9.0
}),
MatrixRow<float>({
22.0, 83.0, 10.0
}),
MatrixRow<float>({
8.0, 83.0, 15.0
}),
MatrixRow<float>({
8.0, 83.0, 8.0
}),
MatrixRow<float>({
8.0, 83.0, 9.0
}),
MatrixRow<float>({
8.0, 83.0, 9.0
}),
MatrixRow<float>({
8.0, 83.0, 5.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
31.0, 83.0, 7.0
}),
MatrixRow<float>({
26.0, 83.0, 8.0
}),
MatrixRow<float>({
22.0, 82.0, 10.0
}),
MatrixRow<float>({
8.0, 82.0, 14.0
}),
MatrixRow<float>({
31.0, 82.0, 6.0
}),
MatrixRow<float>({
8.0, 82.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 70.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
7.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0 }),
MatrixRow<float>({
3.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
7.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 54.0
}),
MatrixRow<float>({
40.0, 87.0, 70.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 62.0
}),
MatrixRow<float>({
15.0, 87.0, 46.0
}),
MatrixRow<float>({
15.0, 87.0, 47.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
0.0, 87.0, 14.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
15.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
22.0, 85.0, 0 }),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
37.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 39.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 0
}),
MatrixRow<float>({
2.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
37.0, 85.0, 6.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0 }),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
18.0, 90.0, 28.0
}),
MatrixRow<float>({
37.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
18.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 60.0
}),
MatrixRow<float>({
37.0, 90.0, 9.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
39.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
15.0, 90.0, 32.0
}),
MatrixRow<float>({
37.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
36.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0 }),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 85.0
}),
MatrixRow<float>({
36.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
37.0, 86.0, 56.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
36.0, 86.0, 35.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
15.0, 86.0, 60.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
36.0, 86.0, 13.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
17.0, 86.0, 25.0
}),
MatrixRow<float>({
36.0, 86.0, 8.0
}),
MatrixRow<float>({
36.0, 86.0, 8.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
31.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0 }),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
31.0, 88.0, 25.0
}),
MatrixRow<float>({
31.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
17.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
2.0, 88.0, 56.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
31.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0 }),
MatrixRow<float>({
3.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 68.0
}),
MatrixRow<float>({
3.0, 91.0, 17.0
}),
MatrixRow<float>({
3.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 95.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 95.0
}),
MatrixRow<float>({
40.0, 91.0, 95.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 23.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
3.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
3.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
3.0, 91.0, 18.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 75.0 }),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
37.0, 91.0, 29.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
7.0, 92.0, 30.0
}),
MatrixRow<float>({
3.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 39.0
}),
MatrixRow<float>({
3.0, 92.0, 73.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
22.0, 92.0, 47.0
}),
MatrixRow<float>({
2.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 41.0
}),
MatrixRow<float>({
3.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 49.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 62.0
}),
MatrixRow<float>({
36.0, 92.0, 24.0
}),
MatrixRow<float>({
22.0, 92.0, 99.0
}),
MatrixRow<float>({
3.0, 92.0, 75.0
}),
MatrixRow<float>({
3.0, 92.0, 33.0
}),
MatrixRow<float>({
3.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
3.0, 92.0, 60.0 }),
MatrixRow<float>({
3.0, 92.0, 16.0
}),
MatrixRow<float>({
2.0, 92.0, 70.0
}),
MatrixRow<float>({
7.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
2.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
37.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
2.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
2.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0 }),
MatrixRow<float>({
37.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
2.0, 87.0, 39.0
}),
MatrixRow<float>({
2.0, 87.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
2.0, 87.0, 34.0
}),
MatrixRow<float>({
2.0, 87.0, 17.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 7.0
}),
MatrixRow<float>({
2.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 7.0
}),
MatrixRow<float>({
40.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 7.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0 }),
MatrixRow<float>({
2.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
0.0, 94.0, 48.0
}),
MatrixRow<float>({
0.0, 94.0, 48.0
}),
MatrixRow<float>({
40.0, 94.0, 58.0
}),
MatrixRow<float>({
40.0, 94.0, 68.0
}),
MatrixRow<float>({
15.0, 94.0, 85.0
}),
MatrixRow<float>({
22.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 96.0
}),
MatrixRow<float>({
15.0, 94.0, 150.0
}),
MatrixRow<float>({
15.0, 94.0, 286.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
22.0, 94.0, 67.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
3.0, 94.0, 30.0
}),
MatrixRow<float>({
15.0, 94.0, 262.0
}),
MatrixRow<float>({
0.0, 94.0, 90.0
}),
MatrixRow<float>({
0.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
36.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
36.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0 }),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 140.0
}),
MatrixRow<float>({
40.0, 94.0, 145.0
}),
MatrixRow<float>({
22.0, 94.0, 90.0
}),
MatrixRow<float>({
7.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 58.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
8.0, 84.0, 45.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
31.0, 84.0, 45.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 35.0 }),
MatrixRow<float>({
15.0, 84.0, 60.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 39.0
}),
MatrixRow<float>({
40.0, 84.0, 33.0
}),
MatrixRow<float>({
15.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
2.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
2.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0 }),
MatrixRow<float>({
31.0, 90.0, 34.0
}),
MatrixRow<float>({
31.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
0.0, 90.0, 13.0
}),
MatrixRow<float>({
0.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
31.0, 90.0, 13.0
}),
MatrixRow<float>({
2.0, 90.0, 19.0
}),
MatrixRow<float>({
6.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 85.0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
6.0, 90.0, 14.0
}),
MatrixRow<float>({
15.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
2.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0 }),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
2.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
29.0, 88.0, 60.0
}),
MatrixRow<float>({
3.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
29.0, 86.0, 55.0
}),
MatrixRow<float>({
22.0, 86.0, 7.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
3.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0 }),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
29.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 89.0
}),
MatrixRow<float>({
40.0, 86.0, 52.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 55.0
}),
MatrixRow<float>({
29.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 33.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 19.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0 }),
MatrixRow<float>({
0.0, 87.0, 9.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
15.0, 87.0, 50.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 32.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
17.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0 }),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0 }),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 31.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 14.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
8.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
8.0, 91.0, 71.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0 }),
MatrixRow<float>({
31.0, 91.0, 33.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
8.0, 91.0, 20.0
}),
MatrixRow<float>({
8.0, 91.0, 19.0
}),
MatrixRow<float>({
15.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
31.0, 91.0, 22.0
}),
MatrixRow<float>({
31.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 64.0
}),
MatrixRow<float>({
23.0, 91.0, 20.0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 58.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
29.0, 93.0, 25.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0 }),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
29.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 113.0
}),
MatrixRow<float>({
15.0, 93.0, 117.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
22.0, 93.0, 79.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
22.0, 93.0, 77.0
}),
MatrixRow<float>({
0.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
22.0, 93.0, 92.0
}),
MatrixRow<float>({
36.0, 93.0, 275.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 95.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
29.0, 93.0, 105.0
}),
MatrixRow<float>({
15.0, 93.0, 70.0
}),
MatrixRow<float>({
15.0, 93.0, 106.0
}),
MatrixRow<float>({
29.0, 93.0, 59.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0 }),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
29.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
15.0, 87.0, 53.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 43.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
29.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
0.0, 87.0, 17.0
}),
MatrixRow<float>({
7.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 25.0 }),
MatrixRow<float>({
29.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
1.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
2.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0 }),
MatrixRow<float>({
27.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 21.0
}),
MatrixRow<float>({
2.0, 87.0, 16.0
}),
MatrixRow<float>({
6.0, 87.0, 22.0
}),
MatrixRow<float>({
6.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 44.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 92.0, 95.0
}),
MatrixRow<float>({
3.0, 92.0, 50.0
}),
MatrixRow<float>({
8.0, 91.0, 90.0
}),
MatrixRow<float>({
8.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
22.0, 91.0, 19.0
}),
MatrixRow<float>({
15.0, 91.0, 47.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
29.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 23.0 }),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
3.0, 91.0, 60.0
}),
MatrixRow<float>({
3.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
31.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
17.0, 92.0, 57.0
}),
MatrixRow<float>({
17.0, 92.0, 71.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
31.0, 92.0, 18.0 }),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 275.0
}),
MatrixRow<float>({
40.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
7.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
8.0, 92.0, 75.0
}),
MatrixRow<float>({
7.0, 92.0, 77.0
}),
MatrixRow<float>({
40.0, 92.0, 135.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
8.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
36.0, 89.0, 19.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 32.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0 }),
MatrixRow<float>({
22.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
3.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 48.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 80.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0 }),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
10.0, 88.0, 24.0
}),
MatrixRow<float>({
8.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 75.0
}),
MatrixRow<float>({
18.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 76.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 91.0
}),
MatrixRow<float>({
29.0, 92.0, 35.0
}),
MatrixRow<float>({
3.0, 92.0, 45.0
}),
MatrixRow<float>({
29.0, 92.0, 79.0
}),
MatrixRow<float>({
40.0, 92.0, 49.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
3.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 18.0 }),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
3.0, 92.0, 28.0
}),
MatrixRow<float>({
3.0, 92.0, 47.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
3.0, 92.0, 49.0
}),
MatrixRow<float>({
3.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 58.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
3.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
3.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 115.0
}),
MatrixRow<float>({
3.0, 92.0, 25.0
}),
MatrixRow<float>({
22.0, 92.0, 160.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0 }),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 48.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 45.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 29.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
15.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0 }),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
17.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
17.0, 87.0, 10.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
37.0, 87.0, 75.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0 }),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 33.0
}),
MatrixRow<float>({
15.0, 92.0, 148.0
}),
MatrixRow<float>({
15.0, 92.0, 68.0
}),
MatrixRow<float>({
40.0, 92.0, 64.0
}),
MatrixRow<float>({
40.0, 92.0, 68.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
22.0, 92.0, 20.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
37.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
8.0, 92.0, 95.0
}),
MatrixRow<float>({
37.0, 92.0, 450.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
17.0, 92.0, 22.0 }),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 150.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
37.0, 92.0, 31.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
17.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
29.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 84.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
29.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
8.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
32.0, 85.0, 7.0 }),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
8.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 5.0
}),
MatrixRow<float>({
8.0, 85.0, 23.0
}),
MatrixRow<float>({
8.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
8.0, 85.0, 39.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
0.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
31.0, 92.0, 29.0 }),
MatrixRow<float>({
0.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
37.0, 92.0, 24.0
}),
MatrixRow<float>({
37.0, 92.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
22.0, 92.0, 80.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 57.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 17.0
}),
MatrixRow<float>({
37.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 36.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
17.0, 88.0, 67.0
}),
MatrixRow<float>({
8.0, 88.0, 24.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0 }),
MatrixRow<float>({
17.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
17.0, 88.0, 16.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
17.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 75.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
17.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 27.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0 }),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 49.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
29.0, 86.0, 55.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0 }),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 93.0, 27.0
}),
MatrixRow<float>({
3.0, 93.0, 44.0 }),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 64.0
}),
MatrixRow<float>({
3.0, 93.0, 51.0
}),
MatrixRow<float>({
40.0, 93.0, 46.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
15.0, 93.0, 24.0
}),
MatrixRow<float>({
3.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 26.0
}),
MatrixRow<float>({
40.0, 93.0, 29.0
}),
MatrixRow<float>({
15.0, 93.0, 32.0
}),
MatrixRow<float>({
3.0, 93.0, 34.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
3.0, 93.0, 29.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
3.0, 93.0, 27.0
}),
MatrixRow<float>({
22.0, 93.0, 120.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 46.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
22.0, 93.0, 125.0
}),
MatrixRow<float>({
3.0, 93.0, 65.0
}),
MatrixRow<float>({
3.0, 93.0, 47.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0 }),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
8.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
8.0, 89.0, 35.0
}),
MatrixRow<float>({
31.0, 89.0, 25.0
}),
MatrixRow<float>({
31.0, 89.0, 24.0
}),
MatrixRow<float>({
31.0, 89.0, 26.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
2.0, 89.0, 30.0
}),
MatrixRow<float>({
29.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
2.0, 89.0, 15.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
2.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 49.0
}),
MatrixRow<float>({
36.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 52.0
}),
MatrixRow<float>({
2.0, 89.0, 25.0
}),
MatrixRow<float>({
36.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0 }),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
29.0, 89.0, 30.0
}),
MatrixRow<float>({
36.0, 88.0, 13.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
8.0, 86.0, 13.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
15.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0 }),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 21.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
37.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 62.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0 }),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 27.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 43.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
3.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 54.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0 }),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
3.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
3.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 75.0
}),
MatrixRow<float>({
3.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 65.0
}),
MatrixRow<float>({
37.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 49.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
22.0, 88.0, 41.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
8.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 0 }),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 49.0
}),
MatrixRow<float>({
36.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 44.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
17.0, 88.0, 104.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
29.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0 }),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 85.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 61.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 21.0
}),
MatrixRow<float>({
22.0, 89.0, 12.0 }),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
8.0, 89.0, 45.0
}),
MatrixRow<float>({
3.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
8.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
37.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
29.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 11.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0 }),
MatrixRow<float>({
37.0, 89.0, 95.0
}),
MatrixRow<float>({
37.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
39.0, 89.0, 18.0
}),
MatrixRow<float>({
39.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
39.0, 89.0, 39.0
}),
MatrixRow<float>({
39.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
37.0, 89.0, 11.0
}),
MatrixRow<float>({
18.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 148.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 85.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
37.0, 89.0, 42.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
37.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 42.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 24.0 }),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 53.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 80.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 25.0 }),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
3.0, 87.0, 47.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 95.0
}),
MatrixRow<float>({
37.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 49.0
}),
MatrixRow<float>({
22.0, 93.0, 270.0
}),
MatrixRow<float>({
22.0, 93.0, 130.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 49.0
}),
MatrixRow<float>({
3.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
37.0, 93.0, 62.0
}),
MatrixRow<float>({
22.0, 93.0, 90.0
}),
MatrixRow<float>({
22.0, 93.0, 59.0
}),
MatrixRow<float>({
3.0, 93.0, 25.0 }),
MatrixRow<float>({
40.0, 93.0, 64.0
}),
MatrixRow<float>({
22.0, 93.0, 72.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 84.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
15.0, 95.0, 50.0
}),
MatrixRow<float>({
15.0, 95.0, 55.0
}),
MatrixRow<float>({
22.0, 95.0, 450.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 95.0, 85.0
}),
MatrixRow<float>({
31.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 95.0, 155.0
}),
MatrixRow<float>({
40.0, 95.0, 40.0
}),
MatrixRow<float>({
22.0, 95.0, 82.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
40.0, 95.0, 55.0
}),
MatrixRow<float>({
0.0, 95.0, 74.0
}),
MatrixRow<float>({
40.0, 95.0, 145.0
}),
MatrixRow<float>({
15.0, 95.0, 75.0
}),
MatrixRow<float>({
40.0, 95.0, 47.0
}),
MatrixRow<float>({
15.0, 95.0, 130.0 }),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
17.0, 95.0, 114.0
}),
MatrixRow<float>({
40.0, 95.0, 155.0
}),
MatrixRow<float>({
40.0, 95.0, 45.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
31.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 95.0, 65.0
}),
MatrixRow<float>({
22.0, 95.0, 55.0
}),
MatrixRow<float>({
40.0, 95.0, 150.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 165.0
}),
MatrixRow<float>({
22.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 120.0
}),
MatrixRow<float>({
15.0, 93.0, 100.0
}),
MatrixRow<float>({
22.0, 93.0, 74.0
}),
MatrixRow<float>({
15.0, 93.0, 250.0
}),
MatrixRow<float>({
15.0, 93.0, 146.0
}),
MatrixRow<float>({
15.0, 93.0, 224.0
}),
MatrixRow<float>({
15.0, 93.0, 138.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 299.0
}),
MatrixRow<float>({
15.0, 93.0, 359.0
}),
MatrixRow<float>({
22.0, 93.0, 50.0 }),
MatrixRow<float>({
40.0, 93.0, 130.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
22.0, 93.0, 115.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 55.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 61.0
}),
MatrixRow<float>({
15.0, 93.0, 185.0
}),
MatrixRow<float>({
22.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 56.0
}),
MatrixRow<float>({
15.0, 92.0, 20.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
31.0, 92.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0 }),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
2.0, 91.0, 50.0
}),
MatrixRow<float>({
2.0, 91.0, 21.0
}),
MatrixRow<float>({
8.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
2.0, 91.0, 75.0
}),
MatrixRow<float>({
29.0, 91.0, 27.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
8.0, 89.0, 39.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 41.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0 }),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 24.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 125.0
}),
MatrixRow<float>({
22.0, 89.0, 32.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 93.0, 115.0
}),
MatrixRow<float>({
40.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0 }),
MatrixRow<float>({
22.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
37.0, 93.0, 149.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
22.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 93.0, 100.0
}),
MatrixRow<float>({
15.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 25.0
}),
MatrixRow<float>({
22.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
15.0, 93.0, 30.0
}),
MatrixRow<float>({
22.0, 93.0, 120.0
}),
MatrixRow<float>({
22.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
15.0, 93.0, 35.0
}),
MatrixRow<float>({
37.0, 93.0, 190.0
}),
MatrixRow<float>({
37.0, 93.0, 110.0
}),
MatrixRow<float>({
40.0, 93.0, 43.0 }),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
2.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 21.0
}),
MatrixRow<float>({
31.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
2.0, 88.0, 29.0
}),
MatrixRow<float>({
37.0, 88.0, 70.0
}),
MatrixRow<float>({
22.0, 88.0, 9.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 24.0
}),
MatrixRow<float>({
2.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
2.0, 88.0, 23.0 }),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
3.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
8.0, 91.0, 100.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
3.0, 91.0, 25.0
}),
MatrixRow<float>({
8.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
17.0, 91.0, 35.0
}),
MatrixRow<float>({
8.0, 91.0, 60.0
}),
MatrixRow<float>({
8.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
31.0, 91.0, 16.0
}),
MatrixRow<float>({
22.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
8.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
17.0, 91.0, 75.0 }),
MatrixRow<float>({
17.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
22.0, 91.0, 26.0
}),
MatrixRow<float>({
17.0, 91.0, 72.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
3.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 77.0
}),
MatrixRow<float>({
8.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 75.0
}),
MatrixRow<float>({
22.0, 89.0, 41.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 27.0
}),
MatrixRow<float>({
15.0, 89.0, 59.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0 }),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
37.0, 89.0, 30.0
}),
MatrixRow<float>({
37.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 8.0
}),
MatrixRow<float>({
15.0, 89.0, 39.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 21.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 57.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 55.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 25.0 }),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
2.0, 88.0, 22.0
}),
MatrixRow<float>({
36.0, 88.0, 15.0
}),
MatrixRow<float>({
0.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 105.0
}),
MatrixRow<float>({
40.0, 90.0, 95.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 90.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 41.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 56.0
}),
MatrixRow<float>({
22.0, 90.0, 68.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 29.0 }),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 53.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 30.0
}),
MatrixRow<float>({
31.0, 90.0, 59.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
42.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 84.0, 8.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 18.0
}),
MatrixRow<float>({
6.0, 83.0, 9.0
}),
MatrixRow<float>({
10.0, 83.0, 21.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
22.0, 83.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 14.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0 }),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
31.0, 83.0, 0
}),
MatrixRow<float>({
5.0, 83.0, 31.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
8.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 12.0
}),
MatrixRow<float>({
8.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 52.0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 78.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
22.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 195.0
}),
MatrixRow<float>({
17.0, 92.0, 69.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0 }),
MatrixRow<float>({
40.0, 92.0, 47.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 99.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 54.0
}),
MatrixRow<float>({
22.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 33.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
8.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0 }),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 52.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 44.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0 }),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
17.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
36.0, 87.0, 28.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 70.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
2.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
2.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
22.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 105.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0 }),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
2.0, 91.0, 120.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
40.0, 91.0, 56.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 31.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 52.0
}),
MatrixRow<float>({
8.0, 86.0, 9.0
}),
MatrixRow<float>({
8.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 23.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0 }),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
17.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
8.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 78.0
}),
MatrixRow<float>({
8.0, 92.0, 33.0
}),
MatrixRow<float>({
22.0, 92.0, 53.0 }),
MatrixRow<float>({
15.0, 92.0, 49.0
}),
MatrixRow<float>({
37.0, 92.0, 40.0
}),
MatrixRow<float>({
8.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
3.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
15.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
22.0, 92.0, 27.0
}),
MatrixRow<float>({
8.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 109.0
}),
MatrixRow<float>({
3.0, 92.0, 50.0
}),
MatrixRow<float>({
3.0, 92.0, 42.0
}),
MatrixRow<float>({
22.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 49.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
37.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0 }),
MatrixRow<float>({
31.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 26.0
}),
MatrixRow<float>({
22.0, 89.0, 43.0
}),
MatrixRow<float>({
22.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 36.0
}),
MatrixRow<float>({
22.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
17.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
18.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0 }),
MatrixRow<float>({
37.0, 89.0, 23.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
36.0, 84.0, 11.0
}),
MatrixRow<float>({
36.0, 84.0, 9.0
}),
MatrixRow<float>({
36.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
37.0, 84.0, 17.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 21.0
}),
MatrixRow<float>({
36.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
2.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 21.0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
40.0, 83.0, 35.0
}),
MatrixRow<float>({
37.0, 83.0, 49.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
2.0, 83.0, 15.0 }),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 30.0
}),
MatrixRow<float>({
3.0, 93.0, 19.0
}),
MatrixRow<float>({
3.0, 93.0, 20.0
}),
MatrixRow<float>({
40.0, 93.0, 37.0
}),
MatrixRow<float>({
0.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 57.0
}),
MatrixRow<float>({
40.0, 93.0, 0
}),
MatrixRow<float>({
0.0, 93.0, 85.0
}),
MatrixRow<float>({
0.0, 93.0, 140.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
3.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0 }),
MatrixRow<float>({
15.0, 93.0, 46.0
}),
MatrixRow<float>({
3.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
37.0, 83.0, 11.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 39.0
}),
MatrixRow<float>({
0.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
15.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 20.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
37.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 36.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0 }),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 24.0
}),
MatrixRow<float>({
0.0, 82.0, 8.0
}),
MatrixRow<float>({
40.0, 82.0, 30.0
}),
MatrixRow<float>({
40.0, 82.0, 30.0
}),
MatrixRow<float>({
40.0, 82.0, 42.0
}),
MatrixRow<float>({
37.0, 82.0, 22.0
}),
MatrixRow<float>({
40.0, 82.0, 11.0
}),
MatrixRow<float>({
15.0, 82.0, 11.0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
3.0, 92.0, 20.0
}),
MatrixRow<float>({
3.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 72.0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
3.0, 92.0, 15.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 0 }),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
37.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 66.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 87.0
}),
MatrixRow<float>({
40.0, 92.0, 52.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
3.0, 92.0, 14.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
3.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
3.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
29.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 80.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0 }),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
2.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
18.0, 88.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
29.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
6.0, 87.0, 8.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0 }),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 78.0
}),
MatrixRow<float>({
32.0, 87.0, 9.0
}),
MatrixRow<float>({
32.0, 87.0, 7.0
}),
MatrixRow<float>({
8.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 79.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
-1.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
37.0, 92.0, 25.0
}),
MatrixRow<float>({
37.0, 92.0, 60.0
}),
MatrixRow<float>({
17.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 21.0
}),
MatrixRow<float>({
2.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 125.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 119.0
}),
MatrixRow<float>({
15.0, 92.0, 33.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 97.0
}),
MatrixRow<float>({ 40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
18.0, 92.0, 21.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 32.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 44.0
}),
MatrixRow<float>({
40.0, 94.0, 115.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
15.0, 93.0, 26.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
15.0, 93.0, 50.0
}),
MatrixRow<float>({
31.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 23.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
31.0, 93.0, 26.0
}),
MatrixRow<float>({ 40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 22.0
}),
MatrixRow<float>({
40.0, 93.0, 68.0
}),
MatrixRow<float>({
8.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 95.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
17.0, 86.0, 22.0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 55.0
}),
MatrixRow<float>({
31.0, 86.0, 24.0
}),
MatrixRow<float>({ 15.0, 85.0, 18.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
31.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 26.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
17.0, 85.0, 10.0
}),
MatrixRow<float>({
17.0, 85.0, 30.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
22.0, 85.0, 22.0
}),
MatrixRow<float>({
31.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 22.0
}),
MatrixRow<float>({
40.0, 93.0, 18.0
}),
MatrixRow<float>({
40.0, 93.0, 59.0
}),
MatrixRow<float>({
40.0, 93.0, 62.0
}),
MatrixRow<float>({
0.0, 93.0, 90.0
}),
MatrixRow<float>({
37.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 150.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({ 15.0, 93.0, 13.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 125.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
22.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 125.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 127.0
}),
MatrixRow<float>({
15.0, 93.0, 57.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 40.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 85.0
}),
MatrixRow<float>({
8.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({ 40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
36.0, 85.0, 14.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 64.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
37.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 105.0
}),
MatrixRow<float>({
22.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 33.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({ 40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
31.0, 90.0, 16.0
}),
MatrixRow<float>({
31.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 55.0
}),
MatrixRow<float>({
31.0, 90.0, 100.0
}),
MatrixRow<float>({
37.0, 90.0, 35.0
}),
MatrixRow<float>({
21.0, 90.0, 20.0
}),
MatrixRow<float>({
21.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
17.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
18.0, 87.0, 43.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({ 2.0, 87.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 17.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 12.0
}),
MatrixRow<float>({
0.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
2.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
37.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
0.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({ 22.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 47.0
}),
MatrixRow<float>({
27.0, 90.0, 38.0
}),
MatrixRow<float>({
27.0, 90.0, 38.0
}),
MatrixRow<float>({
31.0, 90.0, 39.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
0.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 9.0
}),
MatrixRow<float>({ 22.0, 89.0, 65.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
0.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 80.0
}),
MatrixRow<float>({
36.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
0.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
21.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 100.0
}),
MatrixRow<float>({
21.0, 89.0, 17.0
}),
MatrixRow<float>({
31.0, 89.0, 19.0
}),
MatrixRow<float>({
-1.0, 89.0, 28.0
}),
MatrixRow<float>({
0.0, 89.0, 58.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0 }),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
36.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
17.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
19.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
0.0, 92.0, 110.0
}),
MatrixRow<float>({
0.0, 92.0, 110.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
17.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
37.0, 92.0, 34.0
}),
MatrixRow<float>({
15.0, 92.0, 93.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({
3.0, 92.0, 99.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
17.0, 92.0, 28.0
}),
MatrixRow<float>({
0.0, 92.0, 22.0
}),
MatrixRow<float>({
0.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 42.0 }),
MatrixRow<float>({
15.0, 92.0, 43.0
}),
MatrixRow<float>({
15.0, 92.0, 94.0
}),
MatrixRow<float>({
17.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
3.0, 92.0, 37.0
}),
MatrixRow<float>({
0.0, 92.0, 112.0
}),
MatrixRow<float>({
3.0, 92.0, 46.0
}),
MatrixRow<float>({
17.0, 93.0, 38.0
}),
MatrixRow<float>({
17.0, 93.0, 34.0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
31.0, 93.0, 30.0
}),
MatrixRow<float>({
31.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 39.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0 }),
MatrixRow<float>({
40.0, 93.0, 56.0
}),
MatrixRow<float>({
31.0, 93.0, 95.0
}),
MatrixRow<float>({
2.0, 93.0, 100.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 46.0
}),
MatrixRow<float>({
17.0, 93.0, 32.0
}),
MatrixRow<float>({
18.0, 93.0, 16.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
22.0, 93.0, 39.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
31.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
2.0, 89.0, 24.0
}),
MatrixRow<float>({
19.0, 89.0, 39.0
}),
MatrixRow<float>({
40.0, 89.0, 75.0 }),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 36.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
15.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 38.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0 }),
MatrixRow<float>({
15.0, 86.0, 23.0
}),
MatrixRow<float>({
15.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 8.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 33.0
}),
MatrixRow<float>({
36.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 130.0
}),
MatrixRow<float>({
36.0, 88.0, 25.0 }),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
36.0, 88.0, 18.0
}),
MatrixRow<float>({
17.0, 88.0, 12.0
}),
MatrixRow<float>({
17.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
36.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
17.0, 88.0, 14.0
}),
MatrixRow<float>({
6.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 46.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
0.0, 88.0, 19.0
}),
MatrixRow<float>({
36.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 96.0, 160.0
}),
MatrixRow<float>({
40.0, 96.0, 52.0
}),
MatrixRow<float>({
15.0, 96.0, 78.0
}),
MatrixRow<float>({
15.0, 96.0, 60.0
}),
MatrixRow<float>({
15.0, 96.0, 29.0
}),
MatrixRow<float>({
15.0, 96.0, 60.0
}),
MatrixRow<float>({
15.0, 96.0, 45.0
}),
MatrixRow<float>({
15.0, 96.0, 42.0
}),
MatrixRow<float>({
15.0, 96.0, 100.0 }),
MatrixRow<float>({
15.0, 96.0, 120.0
}),
MatrixRow<float>({
15.0, 95.0, 25.0
}),
MatrixRow<float>({
15.0, 95.0, 80.0
}),
MatrixRow<float>({
14.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 95.0, 52.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
36.0, 95.0, 330.0
}),
MatrixRow<float>({
15.0, 95.0, 85.0
}),
MatrixRow<float>({
15.0, 95.0, 120.0
}),
MatrixRow<float>({
15.0, 95.0, 26.0
}),
MatrixRow<float>({
40.0, 95.0, 35.0
}),
MatrixRow<float>({
15.0, 95.0, 54.0
}),
MatrixRow<float>({
15.0, 95.0, 45.0
}),
MatrixRow<float>({
15.0, 95.0, 35.0
}),
MatrixRow<float>({
15.0, 95.0, 150.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
15.0, 95.0, 65.0
}),
MatrixRow<float>({
40.0, 95.0, 62.0
}),
MatrixRow<float>({
40.0, 95.0, 115.0
}),
MatrixRow<float>({
15.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
2.0, 90.0, 50.0
}),
MatrixRow<float>({
17.0, 90.0, 15.0 }),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
17.0, 90.0, 24.0
}),
MatrixRow<float>({
17.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
3.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
3.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
3.0, 90.0, 36.0
}),
MatrixRow<float>({
3.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
0.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0 }),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
31.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
0.0, 87.0, 18.0
}),
MatrixRow<float>({
17.0, 87.0, 33.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
15.0, 87.0, 34.0
}),
MatrixRow<float>({
22.0, 87.0, 29.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
8.0, 89.0, 24.0
}),
MatrixRow<float>({
8.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 0 }),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
29.0, 89.0, 15.0
}),
MatrixRow<float>({
29.0, 89.0, 24.0
}),
MatrixRow<float>({
2.0, 89.0, 26.0
}),
MatrixRow<float>({
8.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 37.0
}),
MatrixRow<float>({
2.0, 89.0, 37.0
}),
MatrixRow<float>({
29.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 46.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 43.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 52.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 15.0
}),
MatrixRow<float>({
22.0, 91.0, 29.0
}),
MatrixRow<float>({
7.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 57.0
}),
MatrixRow<float>({
40.0, 91.0, 110.0 }),
MatrixRow<float>({
15.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 193.0
}),
MatrixRow<float>({
31.0, 91.0, 32.0
}),
MatrixRow<float>({
8.0, 91.0, 61.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
17.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 54.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 54.0
}),
MatrixRow<float>({
40.0, 91.0, 155.0
}),
MatrixRow<float>({
15.0, 91.0, 42.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
6.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
37.0, 89.0, 22.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0 }),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 19.0
}),
MatrixRow<float>({
29.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
29.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
37.0, 89.0, 46.0
}),
MatrixRow<float>({
0.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
37.0, 89.0, 85.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0 }),
MatrixRow<float>({
40.0, 88.0, 125.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
31.0, 88.0, 40.0
}),
MatrixRow<float>({
21.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 107.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
0.0, 88.0, 25.0 }),
MatrixRow<float>({
17.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 56.0
}),
MatrixRow<float>({
36.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 44.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
36.0, 88.0, 22.0
}),
MatrixRow<float>({
36.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 70.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
2.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 44.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0 }),
MatrixRow<float>({
0.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
0.0, 87.0, 90.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
0.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
21.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
22.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 18.0 }),
MatrixRow<float>({
22.0, 86.0, 46.0
}),
MatrixRow<float>({
29.0, 86.0, 22.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
22.0, 86.0, 37.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 21.0
}),
MatrixRow<float>({
37.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
31.0, 87.0, 14.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 70.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
17.0, 87.0, 13.0
}),
MatrixRow<float>({
21.0, 87.0, 45.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 36.0 }),
MatrixRow<float>({
17.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 36.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
2.0, 87.0, 40.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
17.0, 87.0, 11.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
17.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0 }),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 110.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 31.0
}),
MatrixRow<float>({
19.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
37.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
31.0, 87.0, 7.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
15.0, 87.0, 83.0
}),
MatrixRow<float>({
15.0, 86.0, 55.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0 }),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 6.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
31.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 39.0
}),
MatrixRow<float>({
37.0, 85.0, 13.0
}),
MatrixRow<float>({
37.0, 85.0, 6.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
31.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 15.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0 }),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
8.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 88.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 14.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
17.0, 91.0, 30.0
}),
MatrixRow<float>({
17.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
8.0, 91.0, 200.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 14.0
}),
MatrixRow<float>({
40.0, 91.0, 54.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
8.0, 91.0, 35.0
}),
MatrixRow<float>({
0.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0 }),
MatrixRow<float>({
3.0, 89.0, 32.0
}),
MatrixRow<float>({
3.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 12.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
2.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 68.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 44.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 10.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 29.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0 }),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
35.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 52.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
10.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 13.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0 }),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
36.0, 91.0, 29.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
29.0, 91.0, 24.0
}),
MatrixRow<float>({
8.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 68.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 56.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({
8.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
17.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 80.0 }),
MatrixRow<float>({
29.0, 90.0, 22.0
}),
MatrixRow<float>({
37.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
31.0, 90.0, 90.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 67.0
}),
MatrixRow<float>({
22.0, 90.0, 48.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 34.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 54.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
37.0, 87.0, 23.0
}),
MatrixRow<float>({
37.0, 87.0, 17.0 }),
MatrixRow<float>({
15.0, 87.0, 26.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
37.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
3.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
3.0, 87.0, 14.0
}),
MatrixRow<float>({
3.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0 }),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 23.0
}),
MatrixRow<float>({
40.0, 91.0, 9.0
}),
MatrixRow<float>({
22.0, 91.0, 42.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
3.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
37.0, 91.0, 70.0
}),
MatrixRow<float>({
37.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 44.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0 }),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
37.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
8.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 93.0
}),
MatrixRow<float>({
40.0, 91.0, 62.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 165.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 19.0
}),
MatrixRow<float>({
17.0, 91.0, 89.0
}),
MatrixRow<float>({
40.0, 91.0, 135.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0 }),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
37.0, 91.0, 30.0
}),
MatrixRow<float>({
37.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 9.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
36.0, 85.0, 15.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 27.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0 }),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
0.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 58.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
15.0, 90.0, 85.0
}),
MatrixRow<float>({
37.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 12.0
}),
MatrixRow<float>({
17.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 51.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
18.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
31.0, 90.0, 12.0
}),
MatrixRow<float>({
40.0, 90.0, 33.0 }),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
8.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
18.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
36.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
3.0, 88.0, 28.0
}),
MatrixRow<float>({
36.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 18.0 }),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
36.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
36.0, 87.0, 10.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
36.0, 87.0, 10.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
37.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
42.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
8.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0 }),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
42.0, 90.0, 25.0
}),
MatrixRow<float>({
17.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 17.0
}),
MatrixRow<float>({
31.0, 90.0, 14.0
}),
MatrixRow<float>({
31.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
17.0, 90.0, 17.0
}),
MatrixRow<float>({
17.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
17.0, 90.0, 14.0
}),
MatrixRow<float>({
37.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
17.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
2.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 17.0
}),
MatrixRow<float>({
17.0, 89.0, 28.0 }),
MatrixRow<float>({
31.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 53.0
}),
MatrixRow<float>({
15.0, 84.0, 30.0
}),
MatrixRow<float>({
31.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 15.0
}),
MatrixRow<float>({
5.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 18.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
42.0, 84.0, 37.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 80.0
}),
MatrixRow<float>({
5.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 25.0 }),
MatrixRow<float>({
42.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 65.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
42.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 93.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
0.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 85.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
15.0, 91.0, 65.0
}),
MatrixRow<float>({
15.0, 91.0, 29.0 }),
MatrixRow<float>({
15.0, 91.0, 39.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
0.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
0.0, 91.0, 55.0
}),
MatrixRow<float>({
3.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 112.0
}),
MatrixRow<float>({
37.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
22.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0 }),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
21.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
21.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
21.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0 }),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 50.0
}),
MatrixRow<float>({
37.0, 84.0, 18.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
37.0, 84.0, 30.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 0 }),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
7.0, 84.0, 28.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 25.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
36.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 5.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
17.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0 }),
MatrixRow<float>({
29.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 18.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
7.0, 92.0, 60.0
}),
MatrixRow<float>({
3.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
3.0, 92.0, 39.0
}),
MatrixRow<float>({
3.0, 92.0, 75.0
}),
MatrixRow<float>({
3.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
3.0, 92.0, 43.0
}),
MatrixRow<float>({
3.0, 92.0, 29.0
}),
MatrixRow<float>({
3.0, 92.0, 66.0
}),
MatrixRow<float>({
3.0, 92.0, 78.0
}),
MatrixRow<float>({
7.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0 }),
MatrixRow<float>({
3.0, 91.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 66.0
}),
MatrixRow<float>({
2.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
3.0, 91.0, 52.0
}),
MatrixRow<float>({
3.0, 91.0, 62.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
7.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
8.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 95.0, 460.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
22.0, 95.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 235.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 33.0
}),
MatrixRow<float>({
22.0, 94.0, 80.0
}),
MatrixRow<float>({
31.0, 94.0, 35.0
}),
MatrixRow<float>({
31.0, 94.0, 49.0
}),
MatrixRow<float>({
22.0, 94.0, 58.0
}),
MatrixRow<float>({
22.0, 94.0, 80.0
}),
MatrixRow<float>({
31.0, 94.0, 38.0 }),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
22.0, 94.0, 150.0
}),
MatrixRow<float>({
22.0, 94.0, 130.0
}),
MatrixRow<float>({
40.0, 94.0, 110.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
15.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
31.0, 94.0, 23.0
}),
MatrixRow<float>({
22.0, 94.0, 120.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 67.0
}),
MatrixRow<float>({
17.0, 90.0, 27.0
}),
MatrixRow<float>({
17.0, 90.0, 43.0 }),
MatrixRow<float>({
15.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 42.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
7.0, 90.0, 70.0
}),
MatrixRow<float>({
7.0, 90.0, 70.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 78.0
}),
MatrixRow<float>({
22.0, 90.0, 95.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 112.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0 }),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
37.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
37.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
37.0, 85.0, 23.0
}),
MatrixRow<float>({
22.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 72.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 8.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 47.0 }),
MatrixRow<float>({
2.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 69.0
}),
MatrixRow<float>({
2.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
22.0, 90.0, 100.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 48.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
37.0, 90.0, 174.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 112.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0 }),
MatrixRow<float>({
22.0, 91.0, 88.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
37.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 39.0
}),
MatrixRow<float>({
37.0, 91.0, 120.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 125.0
}),
MatrixRow<float>({
40.0, 91.0, 125.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 56.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 56.0
}),
MatrixRow<float>({
22.0, 91.0, 58.0
}),
MatrixRow<float>({
40.0, 91.0, 150.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 140.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0 }),
MatrixRow<float>({
3.0, 90.0, 40.0
}),
MatrixRow<float>({
7.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
7.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
7.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 120.0
}),
MatrixRow<float>({
15.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 24.0
}),
MatrixRow<float>({
3.0, 90.0, 23.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 85.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
7.0, 90.0, 50.0 }),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 93.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 25.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 225.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 49.0
}),
MatrixRow<float>({
15.0, 93.0, 28.0
}),
MatrixRow<float>({
15.0, 93.0, 120.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
15.0, 93.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 75.0
}),
MatrixRow<float>({
0.0, 93.0, 115.0
}),
MatrixRow<float>({
15.0, 93.0, 56.0
}),
MatrixRow<float>({
22.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 72.0
}),
MatrixRow<float>({
3.0, 93.0, 34.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0 }),
MatrixRow<float>({
15.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 29.0
}),
MatrixRow<float>({
40.0, 93.0, 33.0
}),
MatrixRow<float>({
0.0, 93.0, 150.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 8.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 40.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 34.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0 }),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 14.0
}),
MatrixRow<float>({
0.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
0.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
0.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
0.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 57.0
}),
MatrixRow<float>({
22.0, 90.0, 47.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 68.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0 }),
MatrixRow<float>({
22.0, 90.0, 26.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 41.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
31.0, 90.0, 66.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
19.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
31.0, 90.0, 20.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
31.0, 90.0, 13.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
0.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 52.0
}),
MatrixRow<float>({
31.0, 90.0, 38.0 }),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
31.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 59.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
17.0, 87.0, 44.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
31.0, 87.0, 24.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0 }),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 52.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
22.0, 89.0, 48.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0 }),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
31.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 79.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
36.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 24.0
}),
MatrixRow<float>({
36.0, 82.0, 14.0
}),
MatrixRow<float>({
15.0, 82.0, 14.0
}),
MatrixRow<float>({
36.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0 }),
MatrixRow<float>({
15.0, 81.0, 11.0
}),
MatrixRow<float>({
15.0, 81.0, 9.0
}),
MatrixRow<float>({
40.0, 81.0, 25.0
}),
MatrixRow<float>({
40.0, 81.0, 18.0
}),
MatrixRow<float>({
15.0, 80.0, 8.0
}),
MatrixRow<float>({
15.0, 80.0, 11.0
}),
MatrixRow<float>({
40.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
8.0, 93.0, 87.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
2.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 18.0
}),
MatrixRow<float>({
22.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
8.0, 92.0, 60.0
}),
MatrixRow<float>({
2.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 63.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
8.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 25.0 }),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
42.0, 87.0, 17.0
}),
MatrixRow<float>({
42.0, 87.0, 22.0
}),
MatrixRow<float>({
15.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
16.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0 }),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
22.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 59.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 46.0
}),
MatrixRow<float>({
17.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
6.0, 89.0, 11.0
}),
MatrixRow<float>({
22.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 80.0
}),
MatrixRow<float>({
22.0, 89.0, 95.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
-1.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
19.0, 89.0, 37.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
17.0, 89.0, 23.0
}),
MatrixRow<float>({
17.0, 89.0, 16.0
}),
MatrixRow<float>({
21.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 42.0
}),
MatrixRow<float>({
0.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
17.0, 89.0, 34.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({ 0.0, 84.0, 9.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 39.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 95.0, 50.0
}),
MatrixRow<float>({
37.0, 95.0, 85.0
}),
MatrixRow<float>({
37.0, 95.0, 90.0
}),
MatrixRow<float>({
15.0, 95.0, 300.0
}),
MatrixRow<float>({
15.0, 95.0, 225.0
}),
MatrixRow<float>({ 15.0, 95.0, 180.0
}),
MatrixRow<float>({
40.0, 95.0, 65.0
}),
MatrixRow<float>({
15.0, 95.0, 187.0
}),
MatrixRow<float>({
15.0, 95.0, 170.0
}),
MatrixRow<float>({
15.0, 95.0, 120.0
}),
MatrixRow<float>({
40.0, 95.0, 65.0
}),
MatrixRow<float>({
15.0, 95.0, 520.0
}),
MatrixRow<float>({
15.0, 95.0, 200.0
}),
MatrixRow<float>({
37.0, 94.0, 350.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
15.0, 94.0, 175.0
}),
MatrixRow<float>({
15.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
15.0, 94.0, 140.0
}),
MatrixRow<float>({
15.0, 94.0, 140.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
15.0, 94.0, 166.0
}),
MatrixRow<float>({
15.0, 94.0, 257.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 54.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({ 40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 52.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 37.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({ 22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
21.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 37.0
}),
MatrixRow<float>({
22.0, 88.0, 38.0
}),
MatrixRow<float>({
8.0, 88.0, 70.0
}),
MatrixRow<float>({
8.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
18.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 19.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({ 22.0, 86.0, 11.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
17.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
37.0, 87.0, 19.0
}),
MatrixRow<float>({ 40.0, 87.0, 50.0
}),
MatrixRow<float>({
15.0, 87.0, 55.0
}),
MatrixRow<float>({
15.0, 87.0, 65.0
}),
MatrixRow<float>({
37.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 55.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
15.0, 87.0, 39.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
17.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({ 15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
3.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({
3.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
3.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
3.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 130.0
}),
MatrixRow<float>({
3.0, 93.0, 49.0
}),
MatrixRow<float>({
3.0, 93.0, 59.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
3.0, 93.0, 22.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
3.0, 93.0, 45.0
}),
MatrixRow<float>({
3.0, 93.0, 37.0
}),
MatrixRow<float>({
3.0, 93.0, 30.0
}),
MatrixRow<float>({
15.0, 93.0, 45.0
}),
MatrixRow<float>({
3.0, 93.0, 49.0
}),
MatrixRow<float>({
40.0, 93.0, 225.0
}),
MatrixRow<float>({
3.0, 93.0, 95.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({ 15.0, 93.0, 30.0
}),
MatrixRow<float>({
15.0, 93.0, 68.0
}),
MatrixRow<float>({
15.0, 93.0, 47.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
22.0, 93.0, 45.0
}),
MatrixRow<float>({
29.0, 93.0, 105.0
}),
MatrixRow<float>({
0.0, 82.0, 14.0
}),
MatrixRow<float>({
-1.0, 82.0, 0
}),
MatrixRow<float>({
40.0, 82.0, 29.0
}),
MatrixRow<float>({
0.0, 82.0, 13.0
}),
MatrixRow<float>({
37.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 17.0
}),
MatrixRow<float>({
0.0, 82.0, 11.0
}),
MatrixRow<float>({
15.0, 82.0, 15.0
}),
MatrixRow<float>({
15.0, 82.0, 19.0
}),
MatrixRow<float>({
15.0, 82.0, 0
}),
MatrixRow<float>({
15.0, 82.0, 18.0
}),
MatrixRow<float>({
15.0, 82.0, 0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 8.0 }),
MatrixRow<float>({
40.0, 82.0, 13.0
}),
MatrixRow<float>({
0.0, 82.0, 13.0
}),
MatrixRow<float>({
37.0, 81.0, 11.0
}),
MatrixRow<float>({
40.0, 81.0, 14.0
}),
MatrixRow<float>({
15.0, 81.0, 13.0
}),
MatrixRow<float>({
40.0, 81.0, 13.0
}),
MatrixRow<float>({
37.0, 81.0, 130.0
}),
MatrixRow<float>({
40.0, 81.0, 8.0
}),
MatrixRow<float>({
40.0, 81.0, 44.0
}),
MatrixRow<float>({
0.0, 87.0, 50.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 23.0
}),
MatrixRow<float>({
37.0, 87.0, 45.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 80.0
}),
MatrixRow<float>({
15.0, 87.0, 34.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 26.0 }),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
15.0, 87.0, 35.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
15.0, 91.0, 13.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 63.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
8.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0 }),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
31.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
31.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
31.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 33.0
}),
MatrixRow<float>({
37.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
37.0, 91.0, 100.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
8.0, 91.0, 24.0
}),
MatrixRow<float>({
37.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0 }),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 89.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 27.0
}),
MatrixRow<float>({
15.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
17.0, 91.0, 28.0
}),
MatrixRow<float>({
17.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 59.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
37.0, 91.0, 16.0
}),
MatrixRow<float>({
37.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 56.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 32.0 }),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
29.0, 87.0, 22.0
}),
MatrixRow<float>({
29.0, 87.0, 41.0
}),
MatrixRow<float>({
29.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
29.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
8.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0 }),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 69.0
}),
MatrixRow<float>({
37.0, 92.0, 46.0
}),
MatrixRow<float>({
15.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 95.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
31.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
2.0, 92.0, 75.0
}),
MatrixRow<float>({
2.0, 92.0, 75.0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
37.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
0.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0 }),
MatrixRow<float>({
15.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 49.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
3.0, 86.0, 22.0
}),
MatrixRow<float>({
8.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 32.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
2.0, 86.0, 29.0
}),
MatrixRow<float>({
36.0, 86.0, 11.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
0.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 0 }),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
22.0, 92.0, 78.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 126.0
}),
MatrixRow<float>({
31.0, 92.0, 37.0
}),
MatrixRow<float>({
22.0, 92.0, 62.0
}),
MatrixRow<float>({
22.0, 92.0, 84.0
}),
MatrixRow<float>({
22.0, 92.0, 100.0
}),
MatrixRow<float>({
22.0, 92.0, 210.0
}),
MatrixRow<float>({
22.0, 92.0, 80.0
}),
MatrixRow<float>({
31.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
2.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 88.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
22.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0 }),
MatrixRow<float>({
15.0, 91.0, 79.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
37.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 16.0
}),
MatrixRow<float>({
37.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 130.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
22.0, 83.0, 0 }),
MatrixRow<float>({
40.0, 83.0, 26.0
}),
MatrixRow<float>({
0.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
26.0, 83.0, 21.0
}),
MatrixRow<float>({
31.0, 83.0, 8.0
}),
MatrixRow<float>({
22.0, 83.0, 18.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
22.0, 83.0, 12.0
}),
MatrixRow<float>({
22.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 7.0
}),
MatrixRow<float>({
40.0, 83.0, 6.0
}),
MatrixRow<float>({
22.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
0.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 38.0
}),
MatrixRow<float>({
40.0, 83.0, 22.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
0.0, 83.0, 12.0
}),
MatrixRow<float>({
22.0, 83.0, 17.0
}),
MatrixRow<float>({
22.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 11.0
}),
MatrixRow<float>({
0.0, 83.0, 16.0
}),
MatrixRow<float>({
22.0, 83.0, 13.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0 }),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
31.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
31.0, 85.0, 8.0
}),
MatrixRow<float>({
31.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
2.0, 85.0, 23.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0 }),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 85.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 87.0, 70.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 90.0
}),
MatrixRow<float>({
17.0, 87.0, 51.0
}),
MatrixRow<float>({
8.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 59.0
}),
MatrixRow<float>({
15.0, 87.0, 85.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0 }),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 80.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
15.0, 87.0, 19.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
17.0, 87.0, 16.0
}),
MatrixRow<float>({
17.0, 87.0, 26.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 27.0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
37.0, 87.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0 }),
MatrixRow<float>({
15.0, 86.0, 25.0
}),
MatrixRow<float>({
37.0, 86.0, 22.0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
17.0, 90.0, 31.0 }),
MatrixRow<float>({
17.0, 90.0, 40.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
8.0, 90.0, 80.0
}),
MatrixRow<float>({
17.0, 90.0, 31.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
3.0, 90.0, 29.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
3.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 49.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0 }),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 23.0
}),
MatrixRow<float>({
37.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
2.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 15.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
37.0, 89.0, 95.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 11.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
2.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
37.0, 89.0, 19.0
}),
MatrixRow<float>({
29.0, 84.0, 29.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0 }),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
18.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 75.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
18.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 22.0
}),
MatrixRow<float>({
21.0, 84.0, 19.0
}),
MatrixRow<float>({
31.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 27.0
}),
MatrixRow<float>({
29.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 44.0
}),
MatrixRow<float>({
18.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 21.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0 }),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
36.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 7.0
}),
MatrixRow<float>({
8.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0 }),
MatrixRow<float>({
15.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 65.0
}),
MatrixRow<float>({
22.0, 88.0, 10.0
}),
MatrixRow<float>({
7.0, 88.0, 25.0
}),
MatrixRow<float>({
36.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
2.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 37.0
}),
MatrixRow<float>({
2.0, 88.0, 85.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
8.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 92.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 30.0 }),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 46.0
}),
MatrixRow<float>({
36.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 9.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
2.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
2.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 26.0
}),
MatrixRow<float>({
22.0, 84.0, 23.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 17.0 }),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
8.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 49.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
17.0, 90.0, 12.0
}),
MatrixRow<float>({
2.0, 90.0, 70.0
}),
MatrixRow<float>({
2.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
15.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
17.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
17.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0 }),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
2.0, 90.0, 22.0
}),
MatrixRow<float>({
8.0, 90.0, 30.0
}),
MatrixRow<float>({
8.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 58.0
}),
MatrixRow<float>({
37.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 15.0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
37.0, 92.0, 32.0
}),
MatrixRow<float>({
15.0, 92.0, 85.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
37.0, 92.0, 50.0 }),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 78.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 86.0
}),
MatrixRow<float>({
15.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 38.0
}),
MatrixRow<float>({
22.0, 92.0, 72.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 90.0
}),
MatrixRow<float>({
22.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 90.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 20.0 }),
MatrixRow<float>({
15.0, 85.0, 7.0
}),
MatrixRow<float>({
0.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
0.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
37.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 19.0
}),
MatrixRow<float>({
22.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
31.0, 91.0, 10.0
}),
MatrixRow<float>({
31.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 120.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
31.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0 }),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
29.0, 91.0, 100.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
31.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 34.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
31.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 27.0
}),
MatrixRow<float>({
31.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 36.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0 }),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
31.0, 85.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
37.0, 85.0, 16.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
3.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 92.0, 125.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 199.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
0.0, 92.0, 20.0
}),
MatrixRow<float>({
22.0, 92.0, 90.0 }),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 69.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
22.0, 92.0, 38.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 56.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
22.0, 92.0, 37.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
37.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 103.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0 }),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 88.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
2.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 9.0
}),
MatrixRow<float>({
40.0, 83.0, 28.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 29.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 25.0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
0.0, 82.0, 8.0
}),
MatrixRow<float>({
0.0, 82.0, 29.0
}),
MatrixRow<float>({
40.0, 82.0, 33.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
3.0, 82.0, 9.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0 }),
MatrixRow<float>({
40.0, 82.0, 25.0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 8.0
}),
MatrixRow<float>({
40.0, 82.0, 26.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 27.0
}),
MatrixRow<float>({
40.0, 82.0, 45.0
}),
MatrixRow<float>({
40.0, 82.0, 8.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 33.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
37.0, 82.0, 22.0
}),
MatrixRow<float>({
40.0, 82.0, 8.0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
29.0, 86.0, 23.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0 }),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
6.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
29.0, 86.0, 19.0
}),
MatrixRow<float>({
29.0, 86.0, 16.0
}),
MatrixRow<float>({
29.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 23.0
}),
MatrixRow<float>({
21.0, 86.0, 22.0
}),
MatrixRow<float>({
21.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
21.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
18.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 72.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 200.0 }),
MatrixRow<float>({
37.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
17.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 63.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
17.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
31.0, 90.0, 14.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0 }),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
7.0, 91.0, 120.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
37.0, 91.0, 35.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
37.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 160.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 28.0 }),
MatrixRow<float>({
37.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
37.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
32.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
40.0, 84.0, 80.0
}),
MatrixRow<float>({
40.0, 84.0, 60.0 }),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
22.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
15.0, 84.0, 24.0
}),
MatrixRow<float>({
18.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
22.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
3.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 21.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0 }),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
36.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
29.0, 86.0, 25.0
}),
MatrixRow<float>({
29.0, 86.0, 60.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
29.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
3.0, 86.0, 15.0
}),
MatrixRow<float>({
29.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0 }),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
2.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 21.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 68.0
}),
MatrixRow<float>({
22.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
15.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
2.0, 93.0, 90.0
}),
MatrixRow<float>({
36.0, 93.0, 15.0
}),
MatrixRow<float>({
36.0, 92.0, 17.0
}),
MatrixRow<float>({
2.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0 }),
MatrixRow<float>({
40.0, 91.0, 17.0
}),
MatrixRow<float>({
7.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
40.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({
36.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
36.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
36.0, 90.0, 15.0
}),
MatrixRow<float>({
2.0, 90.0, 19.0
}),
MatrixRow<float>({
7.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 13.0
}),
MatrixRow<float>({
22.0, 94.0, 58.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 67.0
}),
MatrixRow<float>({
40.0, 94.0, 120.0 }),
MatrixRow<float>({
40.0, 94.0, 41.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
15.0, 94.0, 400.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 100.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 88.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 65.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 105.0
}),
MatrixRow<float>({
15.0, 94.0, 200.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 38.0
}),
MatrixRow<float>({
40.0, 94.0, 82.0
}),
MatrixRow<float>({
40.0, 94.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
29.0, 87.0, 18.0 }),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
37.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
29.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 34.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0 }),
MatrixRow<float>({
10.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 58.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 11.0
}),
MatrixRow<float>({
36.0, 88.0, 15.0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
29.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
0.0, 88.0, 17.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
15.0, 85.0, 33.0
}),
MatrixRow<float>({
15.0, 85.0, 21.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0 }),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 7.0
}),
MatrixRow<float>({
8.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 55.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0 }),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
31.0, 87.0, 24.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 65.0
}),
MatrixRow<float>({
31.0, 87.0, 11.0
}),
MatrixRow<float>({
31.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0 }),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 59.0
}),
MatrixRow<float>({
2.0, 88.0, 46.0
}),
MatrixRow<float>({
2.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
35.0, 87.0, 30.0
}),
MatrixRow<float>({
10.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
8.0, 87.0, 48.0
}),
MatrixRow<float>({
6.0, 87.0, 9.0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0 }),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 52.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
31.0, 87.0, 28.0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
32.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
22.0, 84.0, 22.0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
22.0, 84.0, 27.0
}),
MatrixRow<float>({
22.0, 84.0, 24.0
}),
MatrixRow<float>({
22.0, 84.0, 17.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0 }),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 23.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
8.0, 86.0, 26.0
}),
MatrixRow<float>({
15.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0 }),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
8.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 22.0 }),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
17.0, 88.0, 101.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
31.0, 88.0, 20.0
}),
MatrixRow<float>({
17.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
17.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0 }),
MatrixRow<float>({
37.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
29.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
0.0, 88.0, 26.0
}),
MatrixRow<float>({
21.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
29.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 70.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0 }),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
29.0, 87.0, 17.0
}),
MatrixRow<float>({
8.0, 87.0, 14.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
3.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 0 }),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 69.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
0.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 38.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
26.0, 86.0, 39.0
}),
MatrixRow<float>({
37.0, 86.0, 45.0
}),
MatrixRow<float>({
37.0, 86.0, 21.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0 }),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
0.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
0.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
2.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
2.0, 88.0, 13.0
}),
MatrixRow<float>({
3.0, 88.0, 20.0
}),
MatrixRow<float>({
3.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 61.0
}),
MatrixRow<float>({
22.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 8.0
}),
MatrixRow<float>({
36.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0 }),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 34.0
}),
MatrixRow<float>({
3.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
3.0, 90.0, 15.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 46.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 16.0
}),
MatrixRow<float>({
29.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0 }),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
5.0, 83.0, 20.0
}),
MatrixRow<float>({
42.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
15.0, 83.0, 13.0
}),
MatrixRow<float>({
37.0, 83.0, 27.0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
40.0, 83.0, 35.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 27.0
}),
MatrixRow<float>({
15.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 33.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0 }),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
3.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 37.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 9.0
}),
MatrixRow<float>({
31.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
3.0, 84.0, 15.0
}),
MatrixRow<float>({
8.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 14.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0 }),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
42.0, 88.0, 20.0
}),
MatrixRow<float>({
42.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 33.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 49.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 43.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0 }),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 7.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
2.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
22.0, 85.0, 21.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 7.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
8.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 21.0 }),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 30.0
}),
MatrixRow<float>({
2.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
29.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
29.0, 90.0, 0
}),
MatrixRow<float>({
29.0, 90.0, 18.0
}),
MatrixRow<float>({
2.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
17.0, 90.0, 30.0
}),
MatrixRow<float>({
37.0, 90.0, 19.0
}),
MatrixRow<float>({
29.0, 90.0, 20.0
}),
MatrixRow<float>({
29.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0 }),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
37.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
0.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
3.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
3.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 65.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0 }),
MatrixRow<float>({
29.0, 88.0, 0
}),
MatrixRow<float>({
3.0, 88.0, 18.0
}),
MatrixRow<float>({
29.0, 88.0, 21.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
29.0, 88.0, 75.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
31.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 27.0
}),
MatrixRow<float>({
22.0, 88.0, 29.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 13.0
}),
MatrixRow<float>({
31.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 65.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0 }),
MatrixRow<float>({
22.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 75.0
}),
MatrixRow<float>({
40.0, 88.0, 41.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 70.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 43.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
2.0, 91.0, 20.0
}),
MatrixRow<float>({
37.0, 91.0, 28.0
}),
MatrixRow<float>({
2.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 125.0
}),
MatrixRow<float>({
37.0, 91.0, 19.0
}),
MatrixRow<float>({
31.0, 91.0, 0 }),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 47.0
}),
MatrixRow<float>({
2.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
31.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
17.0, 91.0, 46.0
}),
MatrixRow<float>({
37.0, 91.0, 62.0
}),
MatrixRow<float>({
40.0, 91.0, 100.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
2.0, 91.0, 75.0
}),
MatrixRow<float>({
36.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
21.0, 91.0, 35.0
}),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
37.0, 91.0, 90.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 32.0 }),
MatrixRow<float>({
22.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 120.0
}),
MatrixRow<float>({
40.0, 91.0, 125.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
37.0, 91.0, 27.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
29.0, 91.0, 53.0
}),
MatrixRow<float>({
22.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 0 }),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 56.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 140.0
}),
MatrixRow<float>({
40.0, 91.0, 140.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 10.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 75.0
}),
MatrixRow<float>({
22.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0 }),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 62.0
}),
MatrixRow<float>({
40.0, 94.0, 175.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
8.0, 94.0, 40.0
}),
MatrixRow<float>({
15.0, 94.0, 49.0
}),
MatrixRow<float>({
15.0, 94.0, 24.0
}),
MatrixRow<float>({
15.0, 94.0, 24.0
}),
MatrixRow<float>({
40.0, 94.0, 68.0
}),
MatrixRow<float>({
40.0, 94.0, 52.0
}),
MatrixRow<float>({
3.0, 94.0, 39.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 58.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
3.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
3.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
15.0, 94.0, 66.0
}),
MatrixRow<float>({
22.0, 94.0, 105.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 108.0 }),
MatrixRow<float>({
40.0, 94.0, 52.0
}),
MatrixRow<float>({
40.0, 94.0, 43.0
}),
MatrixRow<float>({
3.0, 94.0, 51.0
}),
MatrixRow<float>({
40.0, 94.0, 82.0
}),
MatrixRow<float>({
15.0, 94.0, 24.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
2.0, 84.0, 14.0
}),
MatrixRow<float>({
8.0, 84.0, 17.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 75.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
22.0, 84.0, 62.0 }),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 31.0
}),
MatrixRow<float>({
8.0, 83.0, 15.0
}),
MatrixRow<float>({
15.0, 83.0, 13.0
}),
MatrixRow<float>({
22.0, 83.0, 45.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 23.0
}),
MatrixRow<float>({
8.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 82.0, 43.0
}),
MatrixRow<float>({
40.0, 82.0, 39.0
}),
MatrixRow<float>({
0.0, 82.0, 8.0
}),
MatrixRow<float>({
0.0, 82.0, 8.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 32.0
}),
MatrixRow<float>({
40.0, 82.0, 25.0
}),
MatrixRow<float>({
0.0, 82.0, 13.0
}),
MatrixRow<float>({
0.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 12.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 16.0
}),
MatrixRow<float>({
40.0, 82.0, 25.0 }),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 21.0
}),
MatrixRow<float>({
0.0, 81.0, 12.0
}),
MatrixRow<float>({
40.0, 81.0, 28.0
}),
MatrixRow<float>({
40.0, 81.0, 28.0
}),
MatrixRow<float>({
40.0, 81.0, 16.0
}),
MatrixRow<float>({
40.0, 81.0, 16.0
}),
MatrixRow<float>({
40.0, 81.0, 35.0
}),
MatrixRow<float>({
40.0, 81.0, 16.0
}),
MatrixRow<float>({
0.0, 81.0, 11.0
}),
MatrixRow<float>({
40.0, 81.0, 35.0
}),
MatrixRow<float>({
40.0, 80.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 47.0
}),
MatrixRow<float>({
37.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
3.0, 92.0, 38.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 19.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0 }),
MatrixRow<float>({
37.0, 92.0, 58.0
}),
MatrixRow<float>({
37.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 36.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 80.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
15.0, 90.0, 29.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
0.0, 90.0, 30.0
}),
MatrixRow<float>({
37.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
36.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 150.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 53.0
}),
MatrixRow<float>({
37.0, 90.0, 80.0
}),
MatrixRow<float>({
17.0, 90.0, 25.0
}),
MatrixRow<float>({
29.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
7.0, 90.0, 34.0 }),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
29.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
37.0, 90.0, 46.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
37.0, 86.0, 11.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 7.0
}),
MatrixRow<float>({
37.0, 85.0, 18.0
}),
MatrixRow<float>({
36.0, 85.0, 13.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0 }),
MatrixRow<float>({
37.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
30.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 26.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 99.0
}),
MatrixRow<float>({
15.0, 89.0, 43.0
}),
MatrixRow<float>({
40.0, 89.0, 120.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 70.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
3.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 52.0
}),
MatrixRow<float>({
15.0, 89.0, 46.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 49.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0 }),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 43.0
}),
MatrixRow<float>({
0.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
29.0, 89.0, 75.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0 }),
MatrixRow<float>({
22.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
8.0, 84.0, 7.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 83.0, 0
}),
MatrixRow<float>({
15.0, 83.0, 17.0
}),
MatrixRow<float>({
22.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
22.0, 83.0, 16.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0 }),
MatrixRow<float>({
22.0, 83.0, 13.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
22.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
3.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 54.0
}),
MatrixRow<float>({
8.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0 }),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 27.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0 }),
MatrixRow<float>({
37.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
5.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 34.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
0.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 9.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0 }),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
8.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 125.0
}),
MatrixRow<float>({
3.0, 89.0, 22.0
}),
MatrixRow<float>({
21.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
8.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
2.0, 89.0, 21.0 }),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
22.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
21.0, 89.0, 15.0
}),
MatrixRow<float>({
3.0, 89.0, 28.0
}),
MatrixRow<float>({
3.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
17.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 62.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 55.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0 }),
MatrixRow<float>({
22.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
21.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
3.0, 90.0, 28.0
}),
MatrixRow<float>({
29.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
37.0, 90.0, 115.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0 }),
MatrixRow<float>({
40.0, 90.0, 100.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
8.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
37.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 34.0
}),
MatrixRow<float>({
22.0, 90.0, 51.0
}),
MatrixRow<float>({
40.0, 90.0, 54.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
17.0, 90.0, 49.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
8.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 47.0
}),
MatrixRow<float>({
31.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0 }),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
36.0, 86.0, 14.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
36.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 27.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 23.0
}),
MatrixRow<float>({
36.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 33.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 27.0
}),
MatrixRow<float>({
37.0, 85.0, 21.0 }),
MatrixRow<float>({
31.0, 92.0, 39.0
}),
MatrixRow<float>({
0.0, 92.0, 36.0
}),
MatrixRow<float>({
22.0, 92.0, 41.0
}),
MatrixRow<float>({
0.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
0.0, 92.0, 32.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
31.0, 92.0, 110.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
15.0, 92.0, 33.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
0.0, 92.0, 100.0
}),
MatrixRow<float>({
0.0, 92.0, 29.0
}),
MatrixRow<float>({
0.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
17.0, 92.0, 51.0
}),
MatrixRow<float>({
22.0, 92.0, 58.0
}),
MatrixRow<float>({
31.0, 92.0, 16.0
}),
MatrixRow<float>({
21.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0 }),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({
21.0, 92.0, 80.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
0.0, 92.0, 40.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 32.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
31.0, 86.0, 20.0
}),
MatrixRow<float>({
2.0, 86.0, 11.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0
}),
MatrixRow<float>({
8.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 27.0
}),
MatrixRow<float>({
8.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0 }),
MatrixRow<float>({
37.0, 86.0, 8.0
}),
MatrixRow<float>({
22.0, 86.0, 31.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
15.0, 90.0, 32.0
}),
MatrixRow<float>({
15.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 13.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
31.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 14.0
}),
MatrixRow<float>({
31.0, 90.0, 19.0
}),
MatrixRow<float>({
31.0, 90.0, 10.0
}),
MatrixRow<float>({
31.0, 90.0, 89.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
31.0, 90.0, 44.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
29.0, 90.0, 15.0
}),
MatrixRow<float>({
29.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 40.0 }),
MatrixRow<float>({
8.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
8.0, 90.0, 82.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 56.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 82.0
}),
MatrixRow<float>({
29.0, 90.0, 28.0
}),
MatrixRow<float>({
2.0, 90.0, 15.0
}),
MatrixRow<float>({
8.0, 90.0, 55.0
}),
MatrixRow<float>({
8.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
22.0, 93.0, 49.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0 }),
MatrixRow<float>({
37.0, 93.0, 43.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
22.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
22.0, 93.0, 80.0
}),
MatrixRow<float>({
22.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
37.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
3.0, 93.0, 20.0
}),
MatrixRow<float>({
40.0, 93.0, 150.0
}),
MatrixRow<float>({
40.0, 93.0, 49.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 18.0
}),
MatrixRow<float>({
40.0, 93.0, 96.0
}),
MatrixRow<float>({
15.0, 93.0, 0 }),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
22.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 39.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
21.0, 85.0, 18.0
}),
MatrixRow<float>({
21.0, 85.0, 15.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
29.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0 }),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 24.0
}),
MatrixRow<float>({
29.0, 85.0, 23.0
}),
MatrixRow<float>({
40.0, 85.0, 21.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 23.0
}),
MatrixRow<float>({
2.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
17.0, 92.0, 100.0
}),
MatrixRow<float>({
15.0, 92.0, 77.0
}),
MatrixRow<float>({
15.0, 92.0, 96.0
}),
MatrixRow<float>({
2.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
17.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
0.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0 }),
MatrixRow<float>({
37.0, 92.0, 57.0
}),
MatrixRow<float>({
40.0, 92.0, 125.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
2.0, 85.0, 14.0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
2.0, 85.0, 17.0
}),
MatrixRow<float>({
2.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
2.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
2.0, 84.0, 17.0 }),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
2.0, 84.0, 12.0
}),
MatrixRow<float>({
8.0, 84.0, 14.0
}),
MatrixRow<float>({
2.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
2.0, 84.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
2.0, 84.0, 28.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
29.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 29.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 72.0 }),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
29.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 35.0
}),
MatrixRow<float>({
3.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 28.0 }),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 22.0
}),
MatrixRow<float>({
3.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
8.0, 84.0, 7.0
}),
MatrixRow<float>({
8.0, 84.0, 14.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({
22.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 15.0
}),
MatrixRow<float>({
22.0, 91.0, 43.0
}),
MatrixRow<float>({
37.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
0.0, 91.0, 120.0
}),
MatrixRow<float>({
15.0, 91.0, 100.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
15.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 62.0 }),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
29.0, 91.0, 19.0
}),
MatrixRow<float>({
7.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 100.0
}),
MatrixRow<float>({
15.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
2.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 9.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0 }),
MatrixRow<float>({
40.0, 86.0, 85.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 26.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
36.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
37.0, 86.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 32.0 }),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 11.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
8.0, 90.0, 26.0
}),
MatrixRow<float>({
3.0, 90.0, 57.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
8.0, 90.0, 17.0 }),
MatrixRow<float>({
40.0, 90.0, 10.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 22.0
}),
MatrixRow<float>({
36.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
8.0, 90.0, 22.0
}),
MatrixRow<float>({
8.0, 90.0, 15.0
}),
MatrixRow<float>({
18.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
21.0, 88.0, 65.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
8.0, 88.0, 30.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0 }),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
17.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
8.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
22.0, 83.0, 16.0
}),
MatrixRow<float>({
8.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 19.0
}),
MatrixRow<float>({
2.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
22.0, 83.0, 16.0 }),
MatrixRow<float>({
8.0, 83.0, 15.0
}),
MatrixRow<float>({
8.0, 83.0, 20.0
}),
MatrixRow<float>({
8.0, 83.0, 43.0
}),
MatrixRow<float>({
8.0, 83.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
8.0, 82.0, 13.0
}),
MatrixRow<float>({
8.0, 82.0, 20.0
}),
MatrixRow<float>({
2.0, 82.0, 15.0
}),
MatrixRow<float>({
8.0, 82.0, 13.0
}),
MatrixRow<float>({
7.0, 82.0, 22.0
}),
MatrixRow<float>({
22.0, 82.0, 16.0
}),
MatrixRow<float>({
8.0, 82.0, 10.0
}),
MatrixRow<float>({
8.0, 82.0, 25.0
}),
MatrixRow<float>({
8.0, 82.0, 11.0
}),
MatrixRow<float>({
8.0, 82.0, 11.0
}),
MatrixRow<float>({
8.0, 81.0, 13.0
}),
MatrixRow<float>({
8.0, 81.0, 7.0
}),
MatrixRow<float>({
8.0, 81.0, 20.0
}),
MatrixRow<float>({
8.0, 80.0, 14.0
}),
MatrixRow<float>({
8.0, 80.0, 7.0
}),
MatrixRow<float>({
8.0, 80.0, 12.0
}),
MatrixRow<float>({
3.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0 }),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 38.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
29.0, 91.0, 45.0
}),
MatrixRow<float>({
3.0, 91.0, 39.0
}),
MatrixRow<float>({
3.0, 91.0, 33.0
}),
MatrixRow<float>({
3.0, 91.0, 25.0
}),
MatrixRow<float>({
3.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 64.0
}),
MatrixRow<float>({
3.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
3.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
3.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
3.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0 }),
MatrixRow<float>({
29.0, 90.0, 29.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 27.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
18.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 26.0
}),
MatrixRow<float>({
6.0, 84.0, 10.0
}),
MatrixRow<float>({
6.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
29.0, 84.0, 22.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 9.0
}),
MatrixRow<float>({
15.0, 86.0, 0 }),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
31.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 90.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 36.0
}),
MatrixRow<float>({
17.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 65.0
}),
MatrixRow<float>({
15.0, 86.0, 29.0
}),
MatrixRow<float>({
37.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0 }),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
2.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
2.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 9.0
}),
MatrixRow<float>({
2.0, 85.0, 13.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 10.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 11.0
}),
MatrixRow<float>({
0.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0 }),
MatrixRow<float>({
0.0, 84.0, 9.0
}),
MatrixRow<float>({
22.0, 84.0, 23.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 150.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 72.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
37.0, 91.0, 22.0
}),
MatrixRow<float>({
18.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
17.0, 91.0, 24.0
}),
MatrixRow<float>({
37.0, 91.0, 36.0 }),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 100.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
15.0, 91.0, 125.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 145.0
}),
MatrixRow<float>({
15.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 58.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 35.0
}),
MatrixRow<float>({
2.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
0.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 33.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
0.0, 89.0, 10.0 }),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
2.0, 89.0, 40.0
}),
MatrixRow<float>({
0.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
15.0, 89.0, 49.0
}),
MatrixRow<float>({
18.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0 }),
MatrixRow<float>({
37.0, 87.0, 21.0
}),
MatrixRow<float>({
37.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 37.0
}),
MatrixRow<float>({
22.0, 87.0, 32.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
18.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 55.0
}),
MatrixRow<float>({
22.0, 87.0, 37.0
}),
MatrixRow<float>({
37.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 94.0, 69.0
}),
MatrixRow<float>({
15.0, 94.0, 86.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
3.0, 94.0, 46.0
}),
MatrixRow<float>({
15.0, 94.0, 150.0
}),
MatrixRow<float>({
37.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 30.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
36.0, 94.0, 95.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 200.0 }),
MatrixRow<float>({
15.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 95.0
}),
MatrixRow<float>({
15.0, 93.0, 26.0
}),
MatrixRow<float>({
40.0, 93.0, 78.0
}),
MatrixRow<float>({
40.0, 93.0, 43.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
31.0, 93.0, 25.0
}),
MatrixRow<float>({
31.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 93.0, 138.0
}),
MatrixRow<float>({
15.0, 93.0, 135.0
}),
MatrixRow<float>({
40.0, 93.0, 150.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
31.0, 88.0, 18.0
}),
MatrixRow<float>({
0.0, 88.0, 26.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
21.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 100.0
}),
MatrixRow<float>({
15.0, 88.0, 36.0
}),
MatrixRow<float>({
15.0, 88.0, 37.0
}),
MatrixRow<float>({
2.0, 88.0, 18.0 }),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
16.0, 88.0, 28.0
}),
MatrixRow<float>({
15.0, 88.0, 37.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
0.0, 88.0, 0
}),
MatrixRow<float>({
31.0, 88.0, 18.0
}),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 65.0
}),
MatrixRow<float>({
22.0, 88.0, 135.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 59.0
}),
MatrixRow<float>({
16.0, 88.0, 11.0
}),
MatrixRow<float>({
16.0, 88.0, 13.0
}),
MatrixRow<float>({
-1.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 85.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 142.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({ 40.0, 87.0, 48.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 62.0
}),
MatrixRow<float>({
22.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 75.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
0.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 83.0, 25.0
}),
MatrixRow<float>({
2.0, 83.0, 9.0
}),
MatrixRow<float>({
0.0, 83.0, 9.0
}),
MatrixRow<float>({
0.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 83.0, 40.0
}),
MatrixRow<float>({
40.0, 83.0, 34.0
}),
MatrixRow<float>({
40.0, 82.0, 50.0
}),
MatrixRow<float>({ 0.0, 82.0, 6.0
}),
MatrixRow<float>({
37.0, 82.0, 9.0
}),
MatrixRow<float>({
40.0, 82.0, 11.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 30.0
}),
MatrixRow<float>({
37.0, 82.0, 9.0
}),
MatrixRow<float>({
31.0, 82.0, 0
}),
MatrixRow<float>({
0.0, 82.0, 6.0
}),
MatrixRow<float>({
0.0, 82.0, 7.0
}),
MatrixRow<float>({
0.0, 82.0, 8.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
17.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
36.0, 87.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({ 8.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
8.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 15.0
}),
MatrixRow<float>({
36.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
8.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 110.0
}),
MatrixRow<float>({
8.0, 92.0, 66.0
}),
MatrixRow<float>({
31.0, 92.0, 40.0
}),
MatrixRow<float>({
31.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
2.0, 92.0, 32.0
}),
MatrixRow<float>({
22.0, 92.0, 95.0
}),
MatrixRow<float>({
31.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
42.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 21.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({ 15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
8.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
31.0, 92.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 33.0
}),
MatrixRow<float>({
15.0, 91.0, 21.0
}),
MatrixRow<float>({
15.0, 91.0, 14.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({ 37.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
29.0, 86.0, 19.0
}),
MatrixRow<float>({
29.0, 86.0, 16.0
}),
MatrixRow<float>({
29.0, 86.0, 20.0
}),
MatrixRow<float>({
18.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
32.0, 86.0, 7.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
21.0, 86.0, 0
}),
MatrixRow<float>({
21.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
28.0, 86.0, 30.0
}),
MatrixRow<float>({ 40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
32.0, 86.0, 7.0
}),
MatrixRow<float>({
15.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
37.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
38.0, 90.0, 30.0
}),
MatrixRow<float>({
10.0, 90.0, 41.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 44.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
31.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({ 40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 59.0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
37.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
19.0, 89.0, 37.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
37.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({ 40.0, 88.0, 32.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({
3.0, 88.0, 23.0
}),
MatrixRow<float>({
3.0, 88.0, 44.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({ 40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 19.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
8.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
8.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({ 15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 14.0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
0.0, 86.0, 10.0
}),
MatrixRow<float>({
0.0, 86.0, 19.0
}),
MatrixRow<float>({
37.0, 86.0, 24.0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
2.0, 86.0, 14.0
}),
MatrixRow<float>({ 22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 49.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
22.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
22.0, 92.0, 67.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 68.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 54.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 54.0
}),
MatrixRow<float>({
2.0, 92.0, 30.0
}),
MatrixRow<float>({ 40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 59.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
2.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 51.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 80.0, 28.0
}),
MatrixRow<float>({
40.0, 80.0, 28.0
}),
MatrixRow<float>({
40.0, 80.0, 28.0
}),
MatrixRow<float>({
40.0, 80.0, 23.0
}),
MatrixRow<float>({
40.0, 80.0, 15.0
}),
MatrixRow<float>({
40.0, 80.0, 15.0
}),
MatrixRow<float>({
40.0, 80.0, 20.0
}),
MatrixRow<float>({
17.0, 97.0, 0
}),
MatrixRow<float>({
40.0, 96.0, 85.0
}),
MatrixRow<float>({
40.0, 96.0, 110.0
}),
MatrixRow<float>({
40.0, 96.0, 145.0
}),
MatrixRow<float>({ 40.0, 96.0, 95.0
}),
MatrixRow<float>({
40.0, 95.0, 135.0
}),
MatrixRow<float>({
17.0, 95.0, 333.0
}),
MatrixRow<float>({
40.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 95.0, 82.0
}),
MatrixRow<float>({
40.0, 95.0, 75.0
}),
MatrixRow<float>({
2.0, 95.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
37.0, 94.0, 42.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
22.0, 94.0, 115.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
37.0, 89.0, 18.0
}),
MatrixRow<float>({
31.0, 89.0, 15.0
}),
MatrixRow<float>({
31.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({ 22.0, 89.0, 47.0
}),
MatrixRow<float>({
6.0, 89.0, 26.0
}),
MatrixRow<float>({
8.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
32.0, 89.0, 18.0
}),
MatrixRow<float>({
8.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 72.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
27.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 8.0
}),
MatrixRow<float>({
0.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 17.0
}),
MatrixRow<float>({ 15.0, 84.0, 13.0
}),
MatrixRow<float>({
22.0, 84.0, 17.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 27.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
22.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
0.0, 84.0, 16.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 86.0, 6.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 21.0
}),
MatrixRow<float>({
15.0, 86.0, 69.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({ 31.0, 86.0, 17.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 49.0
}),
MatrixRow<float>({
40.0, 86.0, 67.0
}),
MatrixRow<float>({
18.0, 86.0, 25.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 65.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 56.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
8.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
37.0, 91.0, 105.0
}),
MatrixRow<float>({
31.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
8.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({ 40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
8.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
8.0, 91.0, 76.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 32.0
}),
MatrixRow<float>({
37.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
8.0, 91.0, 120.0
}),
MatrixRow<float>({
22.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
15.0, 91.0, 18.0
}),
MatrixRow<float>({
37.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 6.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({ 40.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 6.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 49.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 28.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 30.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
15.0, 83.0, 14.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({ 40.0, 88.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 49.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 80.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
0.0, 88.0, 22.0
}),
MatrixRow<float>({
8.0, 88.0, 35.0
}),
MatrixRow<float>({
21.0, 88.0, 38.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
0.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
0.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
8.0, 88.0, 19.0
}),
MatrixRow<float>({ 15.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
6.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
17.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
0.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
31.0, 84.0, 14.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
0.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
31.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
31.0, 84.0, 11.0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({ 15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 22.0
}),
MatrixRow<float>({
31.0, 84.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
18.0, 89.0, 36.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
18.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 59.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
8.0, 89.0, 24.0
}),
MatrixRow<float>({
17.0, 89.0, 22.0
}),
MatrixRow<float>({
8.0, 89.0, 17.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({ 22.0, 89.0, 0
}),
MatrixRow<float>({
8.0, 89.0, 24.0
}),
MatrixRow<float>({
15.0, 89.0, 70.0
}),
MatrixRow<float>({
15.0, 89.0, 10.0
}),
MatrixRow<float>({
22.0, 89.0, 88.0
}),
MatrixRow<float>({
22.0, 89.0, 27.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 10.0
}),
MatrixRow<float>({
22.0, 89.0, 49.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
3.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 110.0
}),
MatrixRow<float>({
40.0, 94.0, 98.0
}),
MatrixRow<float>({
40.0, 94.0, 110.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 135.0
}),
MatrixRow<float>({ 15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 64.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 32.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
7.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
22.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({ 15.0, 91.0, 23.0
}),
MatrixRow<float>({
29.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 49.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
0.0, 90.0, 35.0
}),
MatrixRow<float>({
36.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
29.0, 86.0, 14.0
}),
MatrixRow<float>({
29.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 150.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({ 22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 11.0
}),
MatrixRow<float>({
8.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
2.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
29.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 23.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
31.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 113.0
}),
MatrixRow<float>({
22.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 41.0
}),
MatrixRow<float>({
40.0, 92.0, 36.0
}),
MatrixRow<float>({
17.0, 92.0, 54.0
}),
MatrixRow<float>({ 15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 33.0
}),
MatrixRow<float>({
2.0, 92.0, 65.0
}),
MatrixRow<float>({
22.0, 92.0, 120.0
}),
MatrixRow<float>({
0.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
17.0, 92.0, 97.0
}),
MatrixRow<float>({
22.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
17.0, 89.0, 12.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({ 17.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
8.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 41.0
}),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 100.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 31.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 26.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
17.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 39.0
}),
MatrixRow<float>({
2.0, 89.0, 13.0
}),
MatrixRow<float>({
8.0, 89.0, 21.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
40.0, 82.0, 30.0
}),
MatrixRow<float>({
37.0, 82.0, 8.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
42.0, 82.0, 12.0
}),
MatrixRow<float>({ 42.0, 82.0, 20.0
}),
MatrixRow<float>({
37.0, 82.0, 26.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
5.0, 82.0, 10.0
}),
MatrixRow<float>({
30.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 7.0
}),
MatrixRow<float>({
40.0, 82.0, 16.0
}),
MatrixRow<float>({
37.0, 81.0, 14.0
}),
MatrixRow<float>({
30.0, 81.0, 10.0
}),
MatrixRow<float>({
37.0, 81.0, 45.0
}),
MatrixRow<float>({
5.0, 81.0, 36.0
}),
MatrixRow<float>({
40.0, 81.0, 27.0
}),
MatrixRow<float>({
40.0, 81.0, 12.0
}),
MatrixRow<float>({
37.0, 81.0, 75.0
}),
MatrixRow<float>({
40.0, 81.0, 20.0
}),
MatrixRow<float>({
5.0, 80.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
31.0, 90.0, 13.0
}),
MatrixRow<float>({
16.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
31.0, 90.0, 24.0
}),
MatrixRow<float>({ 31.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
16.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
8.0, 90.0, 25.0
}),
MatrixRow<float>({
37.0, 90.0, 27.0
}),
MatrixRow<float>({
39.0, 90.0, 27.0
}),
MatrixRow<float>({
17.0, 90.0, 30.0
}),
MatrixRow<float>({
17.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
31.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
21.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({ 21.0, 91.0, 65.0
}),
MatrixRow<float>({
17.0, 91.0, 20.0
}),
MatrixRow<float>({
17.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 62.0
}),
MatrixRow<float>({
22.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 64.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
21.0, 91.0, 45.0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
31.0, 91.0, 60.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
21.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 80.0
}),
MatrixRow<float>({
21.0, 91.0, 70.0
}),
MatrixRow<float>({
22.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({ 36.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 40.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
0.0, 87.0, 32.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 44.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
0.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 59.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
0.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({ 15.0, 88.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 32.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 34.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
18.0, 88.0, 66.0
}),
MatrixRow<float>({
40.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
18.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({ 8.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
8.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
21.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
37.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
17.0, 89.0, 55.0
}),
MatrixRow<float>({
29.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
37.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 24.0
}),
MatrixRow<float>({
17.0, 89.0, 10.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({ 40.0, 89.0, 25.0
}),
MatrixRow<float>({
17.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
0.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 48.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
29.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
29.0, 87.0, 32.0
}),
MatrixRow<float>({
37.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 17.0
}),
MatrixRow<float>({ 15.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
0.0, 87.0, 11.0
}),
MatrixRow<float>({
29.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 59.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
3.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 9.0
}),
MatrixRow<float>({
2.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 51.0
}),
MatrixRow<float>({
40.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({ 22.0, 87.0, 0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
0.0, 86.0, 24.0
}),
MatrixRow<float>({
36.0, 86.0, 20.0
}),
MatrixRow<float>({
3.0, 86.0, 24.0
}),
MatrixRow<float>({
0.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
2.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
36.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
15.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
29.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 46.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
37.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({ 15.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 27.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 54.0
}),
MatrixRow<float>({
3.0, 91.0, 25.0
}),
MatrixRow<float>({
29.0, 91.0, 46.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
0.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 56.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({ 22.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 49.0
}),
MatrixRow<float>({
15.0, 88.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 80.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 37.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 31.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 27.0
}),
MatrixRow<float>({
15.0, 91.0, 32.0
}),
MatrixRow<float>({ 22.0, 91.0, 13.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
36.0, 91.0, 38.0
}),
MatrixRow<float>({
17.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 85.0
}),
MatrixRow<float>({
40.0, 91.0, 80.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
0.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
17.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
3.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({ 15.0, 91.0, 40.0
}),
MatrixRow<float>({
17.0, 91.0, 29.0
}),
MatrixRow<float>({
17.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 80.0
}),
MatrixRow<float>({
31.0, 88.0, 19.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 21.0
}),
MatrixRow<float>({
31.0, 88.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
17.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
31.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 70.0
}),
MatrixRow<float>({
8.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 18.0
}),
MatrixRow<float>({
7.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({ 40.0, 89.0, 38.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
37.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 95.0
}),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
37.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 9.0
}),
MatrixRow<float>({
37.0, 89.0, 35.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
22.0, 89.0, 38.0
}),
MatrixRow<float>({
37.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 75.0
}),
MatrixRow<float>({
15.0, 89.0, 21.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({ 22.0, 84.0, 9.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 7.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 11.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
26.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 16.0
}),
MatrixRow<float>({
22.0, 84.0, 6.0
}),
MatrixRow<float>({
22.0, 84.0, 19.0
}),
MatrixRow<float>({
22.0, 84.0, 17.0
}),
MatrixRow<float>({
22.0, 84.0, 19.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 21.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({ 40.0, 83.0, 26.0
}),
MatrixRow<float>({
0.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 40.0
}),
MatrixRow<float>({
26.0, 83.0, 22.0
}),
MatrixRow<float>({
22.0, 83.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
29.0, 87.0, 33.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
29.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 37.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({ 40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 67.0
}),
MatrixRow<float>({
29.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 93.0, 125.0
}),
MatrixRow<float>({
40.0, 93.0, 20.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 35.0
}),
MatrixRow<float>({
8.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
22.0, 93.0, 46.0
}),
MatrixRow<float>({
40.0, 93.0, 44.0
}),
MatrixRow<float>({
40.0, 93.0, 500.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
15.0, 93.0, 56.0
}),
MatrixRow<float>({
22.0, 93.0, 55.0
}),
MatrixRow<float>({
17.0, 93.0, 56.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
22.0, 93.0, 26.0
}),
MatrixRow<float>({
17.0, 93.0, 38.0
}),
MatrixRow<float>({
15.0, 93.0, 79.0
}),
MatrixRow<float>({
22.0, 93.0, 39.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 110.0
}),
MatrixRow<float>({
40.0, 93.0, 44.0
}),
MatrixRow<float>({ 40.0, 93.0, 69.0
}),
MatrixRow<float>({
37.0, 93.0, 33.0
}),
MatrixRow<float>({
40.0, 93.0, 54.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
17.0, 93.0, 54.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 150.0
}),
MatrixRow<float>({
40.0, 93.0, 69.0
}),
MatrixRow<float>({
22.0, 93.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 28.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 55.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
21.0, 87.0, 22.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({ 40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 37.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 95.0, 260.0
}),
MatrixRow<float>({
40.0, 95.0, 62.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
31.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 95.0, 120.0
}),
MatrixRow<float>({
40.0, 95.0, 165.0
}),
MatrixRow<float>({
22.0, 95.0, 46.0
}),
MatrixRow<float>({
22.0, 95.0, 144.0
}),
MatrixRow<float>({
40.0, 95.0, 82.0
}),
MatrixRow<float>({
15.0, 95.0, 35.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
40.0, 95.0, 160.0
}),
MatrixRow<float>({
15.0, 95.0, 20.0
}),
MatrixRow<float>({
40.0, 95.0, 48.0
}),
MatrixRow<float>({ 40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 250.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
15.0, 94.0, 40.0
}),
MatrixRow<float>({
22.0, 94.0, 69.0
}),
MatrixRow<float>({
2.0, 94.0, 53.0
}),
MatrixRow<float>({
17.0, 94.0, 33.0
}),
MatrixRow<float>({
17.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 52.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
2.0, 94.0, 109.0
}),
MatrixRow<float>({
15.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 22.0
}),
MatrixRow<float>({
31.0, 94.0, 275.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 42.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
37.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({ 40.0, 87.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
31.0, 93.0, 69.0
}),
MatrixRow<float>({
31.0, 93.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 70.0
}),
MatrixRow<float>({
22.0, 93.0, 48.0
}),
MatrixRow<float>({
15.0, 93.0, 153.0
}),
MatrixRow<float>({ 40.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
15.0, 93.0, 88.0
}),
MatrixRow<float>({
15.0, 93.0, 105.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
15.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
31.0, 93.0, 47.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
17.0, 93.0, 52.0
}),
MatrixRow<float>({
15.0, 93.0, 95.0
}),
MatrixRow<float>({
37.0, 93.0, 234.0
}),
MatrixRow<float>({
2.0, 92.0, 40.0
}),
MatrixRow<float>({
17.0, 92.0, 22.0
}),
MatrixRow<float>({
17.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
15.0, 92.0, 67.0
}),
MatrixRow<float>({ 40.0, 90.0, 38.0
}),
MatrixRow<float>({
29.0, 90.0, 27.0
}),
MatrixRow<float>({
3.0, 90.0, 33.0
}),
MatrixRow<float>({
3.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
3.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
3.0, 90.0, 35.0
}),
MatrixRow<float>({
7.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 27.0
}),
MatrixRow<float>({
29.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
22.0, 90.0, 300.0
}),
MatrixRow<float>({ 22.0, 90.0, 28.0
}),
MatrixRow<float>({
3.0, 90.0, 21.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
3.0, 90.0, 39.0
}),
MatrixRow<float>({
37.0, 89.0, 16.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
22.0, 89.0, 42.0
}),
MatrixRow<float>({
15.0, 89.0, 80.0
}),
MatrixRow<float>({
15.0, 89.0, 80.0
}),
MatrixRow<float>({
22.0, 89.0, 100.0
}),
MatrixRow<float>({
31.0, 89.0, 20.0
}),
MatrixRow<float>({
31.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
0.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
17.0, 89.0, 34.0
}),
MatrixRow<float>({
22.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 54.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
31.0, 89.0, 19.0
}),
MatrixRow<float>({
31.0, 89.0, 31.0
}),
MatrixRow<float>({ 22.0, 89.0, 80.0
}),
MatrixRow<float>({
40.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
15.0, 89.0, 29.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
2.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
36.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
37.0, 84.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 38.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({ 15.0, 84.0, 17.0
}),
MatrixRow<float>({
22.0, 84.0, 20.0
}),
MatrixRow<float>({
36.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 44.0
}),
MatrixRow<float>({
40.0, 84.0, 44.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 49.0
}),
MatrixRow<float>({
0.0, 86.0, 19.0
}),
MatrixRow<float>({
31.0, 86.0, 19.0
}),
MatrixRow<float>({
37.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 60.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
37.0, 86.0, 8.0
}),
MatrixRow<float>({ 40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 36.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
37.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({ 40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 95.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
8.0, 85.0, 26.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 48.0
}),
MatrixRow<float>({
29.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 49.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
22.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
16.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 63.0
}),
MatrixRow<float>({
15.0, 90.0, 75.0
}),
MatrixRow<float>({ 22.0, 90.0, 64.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
15.0, 90.0, 42.0
}),
MatrixRow<float>({
21.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
8.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 16.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
0.0, 90.0, 35.0
}),
MatrixRow<float>({
35.0, 90.0, 12.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 200.0
}),
MatrixRow<float>({
15.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 41.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({ 40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 110.0
}),
MatrixRow<float>({
0.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 105.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
22.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 200.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 24.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 108.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 125.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 74.0
}),
MatrixRow<float>({
22.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({ 40.0, 90.0, 25.0
}),
MatrixRow<float>({
18.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
31.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 75.0
}),
MatrixRow<float>({
31.0, 90.0, 19.0
}),
MatrixRow<float>({
31.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 62.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({ 40.0, 94.0, 90.0
}),
MatrixRow<float>({
15.0, 94.0, 450.0
}),
MatrixRow<float>({
17.0, 94.0, 49.0
}),
MatrixRow<float>({
15.0, 94.0, 75.0
}),
MatrixRow<float>({
15.0, 94.0, 75.0
}),
MatrixRow<float>({
15.0, 94.0, 125.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 103.0
}),
MatrixRow<float>({
15.0, 94.0, 144.0
}),
MatrixRow<float>({
15.0, 94.0, 163.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 140.0
}),
MatrixRow<float>({
15.0, 93.0, 113.0
}),
MatrixRow<float>({
15.0, 93.0, 159.0
}),
MatrixRow<float>({
31.0, 93.0, 20.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
15.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 65.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 72.0
}),
MatrixRow<float>({
17.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({ 22.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 56.0
}),
MatrixRow<float>({
37.0, 93.0, 52.0
}),
MatrixRow<float>({
31.0, 93.0, 89.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 80.0
}),
MatrixRow<float>({
0.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 41.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
31.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 94.0, 50.0
}),
MatrixRow<float>({
3.0, 94.0, 39.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
3.0, 94.0, 70.0
}),
MatrixRow<float>({
3.0, 94.0, 39.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0
}),
MatrixRow<float>({
3.0, 94.0, 28.0
}),
MatrixRow<float>({
3.0, 94.0, 28.0
}),
MatrixRow<float>({
2.0, 94.0, 140.0
}),
MatrixRow<float>({ 15.0, 94.0, 0
}),
MatrixRow<float>({
3.0, 94.0, 25.0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
3.0, 94.0, 48.0
}),
MatrixRow<float>({
22.0, 94.0, 45.0
}),
MatrixRow<float>({
2.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 36.0
}),
MatrixRow<float>({
3.0, 94.0, 55.0
}),
MatrixRow<float>({
3.0, 94.0, 42.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
3.0, 94.0, 49.0
}),
MatrixRow<float>({
3.0, 94.0, 54.0
}),
MatrixRow<float>({
22.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 54.0
}),
MatrixRow<float>({
3.0, 94.0, 46.0
}),
MatrixRow<float>({
22.0, 94.0, 165.0
}),
MatrixRow<float>({
40.0, 94.0, 59.0
}),
MatrixRow<float>({
3.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({ 37.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
3.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
3.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
3.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
3.0, 92.0, 14.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
3.0, 92.0, 25.0
}),
MatrixRow<float>({
22.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
37.0, 81.0, 0
}),
MatrixRow<float>({
37.0, 81.0, 13.0
}),
MatrixRow<float>({ 0.0, 81.0, 10.0
}),
MatrixRow<float>({
40.0, 81.0, 8.0
}),
MatrixRow<float>({
22.0, 81.0, 0
}),
MatrixRow<float>({
37.0, 81.0, 30.0
}),
MatrixRow<float>({
40.0, 81.0, 30.0
}),
MatrixRow<float>({
0.0, 81.0, 12.0
}),
MatrixRow<float>({
37.0, 81.0, 22.0
}),
MatrixRow<float>({
0.0, 80.0, 13.0
}),
MatrixRow<float>({
37.0, 80.0, 17.0
}),
MatrixRow<float>({
0.0, 80.0, 17.0
}),
MatrixRow<float>({
40.0, 80.0, 15.0
}),
MatrixRow<float>({
0.0, 80.0, 10.0
}),
MatrixRow<float>({
15.0, 94.0, 42.0
}),
MatrixRow<float>({
2.0, 91.0, 17.0
}),
MatrixRow<float>({
40.0, 95.0, 70.0
}),
MatrixRow<float>({
22.0, 95.0, 100.0
}),
MatrixRow<float>({
40.0, 95.0, 55.0
}),
MatrixRow<float>({
40.0, 95.0, 65.0
}),
MatrixRow<float>({
15.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
31.0, 92.0, 0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 62.0
}),
MatrixRow<float>({
31.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({ 40.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 99.0
}),
MatrixRow<float>({
40.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
31.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 85.0
}),
MatrixRow<float>({
15.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 45.0
}),
MatrixRow<float>({
31.0, 92.0, 78.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
0.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
22.0, 92.0, 165.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
15.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 125.0
}),
MatrixRow<float>({ 40.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 36.0
}),
MatrixRow<float>({
15.0, 94.0, 30.0
}),
MatrixRow<float>({
15.0, 94.0, 49.0
}),
MatrixRow<float>({
15.0, 94.0, 30.0
}),
MatrixRow<float>({
15.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 58.0
}),
MatrixRow<float>({
15.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 210.0
}),
MatrixRow<float>({
15.0, 94.0, 150.0
}),
MatrixRow<float>({
15.0, 94.0, 37.0
}),
MatrixRow<float>({
15.0, 94.0, 90.0
}),
MatrixRow<float>({
15.0, 94.0, 25.0
}),
MatrixRow<float>({
15.0, 94.0, 42.0
}),
MatrixRow<float>({
40.0, 94.0, 175.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
15.0, 94.0, 36.0
}),
MatrixRow<float>({
15.0, 94.0, 40.0
}),
MatrixRow<float>({
40.0, 94.0, 30.0
}),
MatrixRow<float>({
37.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({ 15.0, 94.0, 44.0
}),
MatrixRow<float>({
15.0, 94.0, 39.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
37.0, 92.0, 100.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 31.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 23.0
}),
MatrixRow<float>({
15.0, 92.0, 145.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
15.0, 92.0, 51.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 44.0
}),
MatrixRow<float>({
15.0, 92.0, 99.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
15.0, 92.0, 140.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 56.0
}),
MatrixRow<float>({
15.0, 92.0, 57.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
8.0, 88.0, 20.0
}),
MatrixRow<float>({ 40.0, 88.0, 12.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 41.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
18.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
18.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
22.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
8.0, 88.0, 21.0
}),
MatrixRow<float>({
8.0, 88.0, 35.0
}),
MatrixRow<float>({ 15.0, 91.0, 23.0
}),
MatrixRow<float>({
15.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 18.0
}),
MatrixRow<float>({
31.0, 91.0, 26.0
}),
MatrixRow<float>({
31.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 125.0
}),
MatrixRow<float>({
21.0, 91.0, 48.0
}),
MatrixRow<float>({
31.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
31.0, 91.0, 22.0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
31.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 15.0
}),
MatrixRow<float>({
8.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 18.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 95.0
}),
MatrixRow<float>({ 40.0, 91.0, 85.0
}),
MatrixRow<float>({
31.0, 91.0, 35.0
}),
MatrixRow<float>({
42.0, 91.0, 60.0
}),
MatrixRow<float>({
42.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
0.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 33.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
2.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 20.0
}),
MatrixRow<float>({
2.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
2.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
31.0, 88.0, 8.0
}),
MatrixRow<float>({
22.0, 88.0, 44.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
2.0, 88.0, 14.0
}),
MatrixRow<float>({ 40.0, 88.0, 45.0
}),
MatrixRow<float>({
17.0, 95.0, 52.0
}),
MatrixRow<float>({
17.0, 95.0, 50.0
}),
MatrixRow<float>({
31.0, 95.0, 150.0
}),
MatrixRow<float>({
40.0, 95.0, 125.0
}),
MatrixRow<float>({
31.0, 95.0, 100.0
}),
MatrixRow<float>({
31.0, 95.0, 100.0
}),
MatrixRow<float>({
31.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
17.0, 94.0, 42.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 49.0
}),
MatrixRow<float>({
40.0, 94.0, 175.0
}),
MatrixRow<float>({
40.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
17.0, 94.0, 37.0
}),
MatrixRow<float>({
37.0, 94.0, 165.0
}),
MatrixRow<float>({
22.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
31.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 150.0
}),
MatrixRow<float>({
37.0, 94.0, 85.0
}),
MatrixRow<float>({
31.0, 94.0, 99.0
}),
MatrixRow<float>({ 31.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 94.0, 72.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
31.0, 94.0, 30.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
2.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 92.0
}),
MatrixRow<float>({ 15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 33.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 42.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 97.0, 125.0
}),
MatrixRow<float>({
40.0, 97.0, 90.0
}),
MatrixRow<float>({
15.0, 97.0, 698.0
}),
MatrixRow<float>({
15.0, 97.0, 450.0
}),
MatrixRow<float>({
15.0, 97.0, 285.0
}),
MatrixRow<float>({
15.0, 97.0, 330.0
}),
MatrixRow<float>({
15.0, 97.0, 150.0
}),
MatrixRow<float>({
15.0, 96.0, 130.0
}),
MatrixRow<float>({
15.0, 96.0, 400.0
}),
MatrixRow<float>({
15.0, 96.0, 2500.0
}),
MatrixRow<float>({
15.0, 96.0, 75.0
}),
MatrixRow<float>({
15.0, 96.0, 64.0
}),
MatrixRow<float>({
40.0, 96.0, 63.0
}),
MatrixRow<float>({
22.0, 96.0, 65.0
}),
MatrixRow<float>({
40.0, 96.0, 54.0
}),
MatrixRow<float>({
37.0, 96.0, 770.0
}),
MatrixRow<float>({
15.0, 96.0, 475.0
}),
MatrixRow<float>({ 15.0, 96.0, 50.0
}),
MatrixRow<float>({
15.0, 96.0, 110.0
}),
MatrixRow<float>({
40.0, 96.0, 90.0
}),
MatrixRow<float>({
40.0, 96.0, 40.0
}),
MatrixRow<float>({
40.0, 96.0, 85.0
}),
MatrixRow<float>({
22.0, 96.0, 199.0
}),
MatrixRow<float>({
15.0, 96.0, 160.0
}),
MatrixRow<float>({
15.0, 96.0, 200.0
}),
MatrixRow<float>({
15.0, 96.0, 80.0
}),
MatrixRow<float>({
15.0, 96.0, 120.0
}),
MatrixRow<float>({
40.0, 96.0, 63.0
}),
MatrixRow<float>({
15.0, 96.0, 175.0
}),
MatrixRow<float>({
40.0, 96.0, 85.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 12.0
}),
MatrixRow<float>({
29.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 89.0, 55.0
}),
MatrixRow<float>({
37.0, 89.0, 23.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({ 40.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 39.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
3.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 85.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({ 40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 85.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
17.0, 88.0, 12.0
}),
MatrixRow<float>({
8.0, 88.0, 16.0
}),
MatrixRow<float>({
2.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({
22.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 72.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({ 40.0, 91.0, 45.0
}),
MatrixRow<float>({
17.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 38.0
}),
MatrixRow<float>({
31.0, 91.0, 32.0
}),
MatrixRow<float>({
17.0, 91.0, 42.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
37.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 141.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 33.0
}),
MatrixRow<float>({
22.0, 91.0, 92.0
}),
MatrixRow<float>({
40.0, 91.0, 100.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 52.0
}),
MatrixRow<float>({
22.0, 90.0, 52.0
}),
MatrixRow<float>({
17.0, 90.0, 34.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 17.0
}),
MatrixRow<float>({
17.0, 90.0, 20.0
}),
MatrixRow<float>({
31.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
17.0, 90.0, 16.0
}),
MatrixRow<float>({ 0.0, 90.0, 15.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
37.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
36.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 7.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
15.0, 85.0, 33.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({ 40.0, 85.0, 19.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
8.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
31.0, 86.0, 0
}),
MatrixRow<float>({
2.0, 86.0, 8.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
31.0, 86.0, 7.0
}),
MatrixRow<float>({
42.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 75.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
5.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
8.0, 86.0, 17.0
}),
MatrixRow<float>({ 22.0, 86.0, 17.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
-1.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
40.0, 86.0, 28.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
8.0, 86.0, 16.0
}),
MatrixRow<float>({
3.0, 89.0, 110.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 36.0
}),
MatrixRow<float>({
15.0, 89.0, 27.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0 }),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
37.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 38.0
}),
MatrixRow<float>({
37.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 37.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 9.0
}),
MatrixRow<float>({
31.0, 87.0, 19.0
}),
MatrixRow<float>({
31.0, 87.0, 0 }),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
31.0, 87.0, 10.0
}),
MatrixRow<float>({
31.0, 87.0, 8.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
31.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 21.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 79.0
}),
MatrixRow<float>({
40.0, 87.0, 21.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 29.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 70.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 21.0
}),
MatrixRow<float>({
31.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0 }),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
37.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
31.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
37.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
40.0, 84.0, 0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
36.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
0.0, 83.0, 8.0 }),
MatrixRow<float>({
31.0, 83.0, 9.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
31.0, 83.0, 8.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 97.0, 0
}),
MatrixRow<float>({
15.0, 97.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 96.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 0
}),
MatrixRow<float>({
15.0, 95.0, 150.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0 }),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
40.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
22.0, 93.0, 46.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
15.0, 93.0, 45.0
}),
MatrixRow<float>({
22.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
15.0, 93.0, 120.0
}),
MatrixRow<float>({
15.0, 93.0, 140.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 85.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
22.0, 93.0, 86.0
}),
MatrixRow<float>({
40.0, 93.0, 27.0
}),
MatrixRow<float>({
15.0, 93.0, 59.0 }),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
22.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
8.0, 92.0, 120.0
}),
MatrixRow<float>({
8.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 23.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
22.0, 92.0, 40.0
}),
MatrixRow<float>({
8.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 20.0
}),
MatrixRow<float>({
22.0, 93.0, 35.0
}),
MatrixRow<float>({
22.0, 93.0, 55.0
}),
MatrixRow<float>({
31.0, 93.0, 36.0
}),
MatrixRow<float>({
31.0, 93.0, 77.0
}),
MatrixRow<float>({
15.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
22.0, 93.0, 24.0 }),
MatrixRow<float>({
15.0, 93.0, 26.0
}),
MatrixRow<float>({
15.0, 93.0, 20.0
}),
MatrixRow<float>({
15.0, 93.0, 41.0
}),
MatrixRow<float>({
15.0, 93.0, 21.0
}),
MatrixRow<float>({
37.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
31.0, 93.0, 50.0
}),
MatrixRow<float>({
40.0, 93.0, 108.0
}),
MatrixRow<float>({
15.0, 93.0, 17.0
}),
MatrixRow<float>({
15.0, 93.0, 65.0
}),
MatrixRow<float>({
15.0, 93.0, 29.0
}),
MatrixRow<float>({
15.0, 93.0, 22.0
}),
MatrixRow<float>({
15.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 39.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
37.0, 92.0, 42.0
}),
MatrixRow<float>({
37.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 12.0
}),
MatrixRow<float>({
10.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 100.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
29.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0 }),
MatrixRow<float>({
10.0, 90.0, 18.0
}),
MatrixRow<float>({
0.0, 90.0, 20.0
}),
MatrixRow<float>({
0.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 21.0
}),
MatrixRow<float>({
29.0, 90.0, 17.0
}),
MatrixRow<float>({
29.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 17.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
0.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 29.0
}),
MatrixRow<float>({
29.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
21.0, 90.0, 14.0
}),
MatrixRow<float>({
37.0, 90.0, 45.0
}),
MatrixRow<float>({
37.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0 }),
MatrixRow<float>({
29.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
36.0, 87.0, 11.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 0
}),
MatrixRow<float>({
36.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0 }),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
37.0, 87.0, 30.0
}),
MatrixRow<float>({
0.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 28.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
40.0, 86.0, 32.0
}),
MatrixRow<float>({
15.0, 86.0, 27.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
17.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 19.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 86.0, 18.0 }),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 36.0
}),
MatrixRow<float>({
21.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
0.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 54.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 42.0
}),
MatrixRow<float>({
40.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
37.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 40.0
}),
MatrixRow<float>({
0.0, 87.0, 19.0
}),
MatrixRow<float>({
37.0, 87.0, 25.0 }),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 58.0
}),
MatrixRow<float>({
15.0, 94.0, 299.0
}),
MatrixRow<float>({
15.0, 94.0, 82.0
}),
MatrixRow<float>({
31.0, 94.0, 56.0
}),
MatrixRow<float>({
15.0, 94.0, 260.0
}),
MatrixRow<float>({
31.0, 94.0, 70.0
}),
MatrixRow<float>({
2.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
37.0, 94.0, 45.0
}),
MatrixRow<float>({
22.0, 94.0, 58.0
}),
MatrixRow<float>({
3.0, 94.0, 30.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
22.0, 94.0, 42.0 }),
MatrixRow<float>({
40.0, 94.0, 28.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 86.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
31.0, 94.0, 94.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 41.0
}),
MatrixRow<float>({
40.0, 94.0, 28.0
}),
MatrixRow<float>({
2.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
17.0, 94.0, 54.0
}),
MatrixRow<float>({
31.0, 94.0, 300.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
15.0, 94.0, 120.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
8.0, 87.0, 13.0
}),
MatrixRow<float>({
29.0, 87.0, 11.0
}),
MatrixRow<float>({
29.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0 }),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
36.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 32.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 84.0
}),
MatrixRow<float>({
15.0, 92.0, 38.0 }),
MatrixRow<float>({
15.0, 92.0, 20.0
}),
MatrixRow<float>({
15.0, 92.0, 38.0
}),
MatrixRow<float>({
37.0, 92.0, 30.0
}),
MatrixRow<float>({
37.0, 92.0, 16.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 49.0
}),
MatrixRow<float>({
37.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 59.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 79.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
15.0, 92.0, 26.0
}),
MatrixRow<float>({
15.0, 92.0, 48.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 125.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0 }),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 17.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
0.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 46.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
21.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 23.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0 }),
MatrixRow<float>({
15.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
3.0, 89.0, 32.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
3.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 44.0
}),
MatrixRow<float>({
3.0, 89.0, 39.0
}),
MatrixRow<float>({
29.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 70.0
}),
MatrixRow<float>({
22.0, 89.0, 23.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
15.0, 89.0, 0 }),
MatrixRow<float>({
15.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 17.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 46.0
}),
MatrixRow<float>({
3.0, 89.0, 54.0
}),
MatrixRow<float>({
15.0, 89.0, 33.0
}),
MatrixRow<float>({
15.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
15.0, 85.0, 19.0
}),
MatrixRow<float>({
10.0, 85.0, 35.0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
4.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0 }),
MatrixRow<float>({
0.0, 85.0, 7.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
17.0, 88.0, 9.0
}),
MatrixRow<float>({
22.0, 88.0, 50.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
22.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 38.0
}),
MatrixRow<float>({
0.0, 93.0, 80.0
}),
MatrixRow<float>({
15.0, 93.0, 142.0
}),
MatrixRow<float>({
40.0, 93.0, 34.0
}),
MatrixRow<float>({
22.0, 93.0, 95.0
}),
MatrixRow<float>({
22.0, 93.0, 94.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0 }),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 95.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
15.0, 93.0, 50.0
}),
MatrixRow<float>({
15.0, 93.0, 100.0
}),
MatrixRow<float>({
22.0, 93.0, 80.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 175.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 0
}),
MatrixRow<float>({
15.0, 93.0, 56.0
}),
MatrixRow<float>({
15.0, 93.0, 54.0
}),
MatrixRow<float>({
22.0, 93.0, 250.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
0.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 44.0
}),
MatrixRow<float>({
22.0, 82.0, 0
}),
MatrixRow<float>({
37.0, 82.0, 16.0
}),
MatrixRow<float>({
37.0, 82.0, 17.0
}),
MatrixRow<float>({
15.0, 82.0, 22.0
}),
MatrixRow<float>({
8.0, 81.0, 7.0
}),
MatrixRow<float>({
37.0, 81.0, 30.0 }),
MatrixRow<float>({
8.0, 81.0, 25.0
}),
MatrixRow<float>({
8.0, 81.0, 90.0
}),
MatrixRow<float>({
17.0, 98.0, 775.0
}),
MatrixRow<float>({
22.0, 97.0, 75.0
}),
MatrixRow<float>({
31.0, 97.0, 40.0
}),
MatrixRow<float>({
17.0, 96.0, 316.0
}),
MatrixRow<float>({
22.0, 96.0, 55.0
}),
MatrixRow<float>({
22.0, 96.0, 70.0
}),
MatrixRow<float>({
40.0, 96.0, 32.0
}),
MatrixRow<float>({
40.0, 96.0, 60.0
}),
MatrixRow<float>({
22.0, 96.0, 98.0
}),
MatrixRow<float>({
40.0, 96.0, 240.0
}),
MatrixRow<float>({
15.0, 95.0, 120.0
}),
MatrixRow<float>({
31.0, 95.0, 55.0
}),
MatrixRow<float>({
22.0, 95.0, 75.0
}),
MatrixRow<float>({
40.0, 95.0, 120.0
}),
MatrixRow<float>({
40.0, 95.0, 60.0
}),
MatrixRow<float>({
31.0, 95.0, 58.0
}),
MatrixRow<float>({
2.0, 95.0, 65.0
}),
MatrixRow<float>({
31.0, 95.0, 50.0
}),
MatrixRow<float>({
31.0, 95.0, 52.0
}),
MatrixRow<float>({
40.0, 95.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
40.0, 84.0, 21.0 }),
MatrixRow<float>({
22.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 24.0
}),
MatrixRow<float>({
37.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 11.0
}),
MatrixRow<float>({
15.0, 83.0, 30.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
8.0, 83.0, 19.0
}),
MatrixRow<float>({
15.0, 83.0, 14.0
}),
MatrixRow<float>({
8.0, 83.0, 12.0
}),
MatrixRow<float>({
40.0, 83.0, 7.0
}),
MatrixRow<float>({
40.0, 83.0, 24.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 28.0
}),
MatrixRow<float>({
40.0, 83.0, 13.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0 }),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
15.0, 92.0, 79.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
37.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
37.0, 92.0, 116.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
3.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 89.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
40.0, 92.0, 17.0
}),
MatrixRow<float>({
40.0, 92.0, 64.0 }),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
37.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 17.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
21.0, 86.0, 15.0
}),
MatrixRow<float>({
3.0, 86.0, 10.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 11.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
29.0, 86.0, 27.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0 }),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 33.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
22.0, 93.0, 39.0
}),
MatrixRow<float>({
0.0, 93.0, 111.0
}),
MatrixRow<float>({
40.0, 93.0, 58.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
17.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 64.0
}),
MatrixRow<float>({
22.0, 93.0, 75.0
}),
MatrixRow<float>({
40.0, 93.0, 28.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
22.0, 93.0, 45.0
}),
MatrixRow<float>({
22.0, 93.0, 140.0
}),
MatrixRow<float>({
22.0, 93.0, 99.0
}),
MatrixRow<float>({
2.0, 93.0, 55.0 }),
MatrixRow<float>({
31.0, 93.0, 22.0
}),
MatrixRow<float>({
22.0, 93.0, 100.0
}),
MatrixRow<float>({
15.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 100.0
}),
MatrixRow<float>({
31.0, 93.0, 24.0
}),
MatrixRow<float>({
40.0, 93.0, 116.0
}),
MatrixRow<float>({
15.0, 93.0, 64.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
17.0, 93.0, 25.0
}),
MatrixRow<float>({
40.0, 93.0, 44.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
17.0, 93.0, 43.0
}),
MatrixRow<float>({
31.0, 93.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
37.0, 85.0, 11.0
}),
MatrixRow<float>({
0.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
37.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 85.0, 24.0 }),
MatrixRow<float>({
15.0, 85.0, 60.0
}),
MatrixRow<float>({
31.0, 85.0, 13.0
}),
MatrixRow<float>({
37.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
37.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 125.0
}),
MatrixRow<float>({
31.0, 85.0, 0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 39.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 24.0
}),
MatrixRow<float>({
3.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0 }),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 55.0
}),
MatrixRow<float>({
3.0, 88.0, 30.0
}),
MatrixRow<float>({
0.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 50.0
}),
MatrixRow<float>({
0.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 9.0
}),
MatrixRow<float>({
8.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
3.0, 88.0, 27.0
}),
MatrixRow<float>({
0.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 35.0
}),
MatrixRow<float>({
8.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0 }),
MatrixRow<float>({
37.0, 90.0, 17.0
}),
MatrixRow<float>({
31.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
31.0, 90.0, 29.0
}),
MatrixRow<float>({
40.0, 90.0, 68.0
}),
MatrixRow<float>({
15.0, 90.0, 23.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 33.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
6.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 85.0
}),
MatrixRow<float>({
31.0, 90.0, 12.0 }),
MatrixRow<float>({
40.0, 90.0, 74.0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
-1.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({ 36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 90.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
36.0, 88.0, 0
}),
MatrixRow<float>({
36.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 98.0, 60.0
}),
MatrixRow<float>({
22.0, 98.0, 70.0
}),
MatrixRow<float>({
22.0, 98.0, 151.0
}),
MatrixRow<float>({
22.0, 98.0, 300.0
}),
MatrixRow<float>({
22.0, 97.0, 82.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
22.0, 89.0, 32.0
}),
MatrixRow<float>({
22.0, 89.0, 10.0
}),
MatrixRow<float>({ 17.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 37.0
}),
MatrixRow<float>({
17.0, 89.0, 31.0
}),
MatrixRow<float>({
17.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
22.0, 89.0, 30.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 44.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 31.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 11.0
}),
MatrixRow<float>({
40.0, 89.0, 27.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 31.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
17.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
17.0, 89.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({ 3.0, 88.0, 13.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 27.0
}),
MatrixRow<float>({
35.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 125.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 39.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
10.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 87.0, 12.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({ 37.0, 87.0, 15.0
}),
MatrixRow<float>({
29.0, 83.0, 30.0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
22.0, 83.0, 11.0
}),
MatrixRow<float>({
40.0, 83.0, 14.0
}),
MatrixRow<float>({
15.0, 83.0, 15.0
}),
MatrixRow<float>({
22.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 82.0, 16.0
}),
MatrixRow<float>({
2.0, 82.0, 15.0
}),
MatrixRow<float>({
8.0, 82.0, 7.0
}),
MatrixRow<float>({
40.0, 82.0, 18.0
}),
MatrixRow<float>({
29.0, 82.0, 0
}),
MatrixRow<float>({
15.0, 82.0, 19.0
}),
MatrixRow<float>({
29.0, 82.0, 14.0
}),
MatrixRow<float>({
22.0, 82.0, 15.0
}),
MatrixRow<float>({
40.0, 81.0, 20.0
}),
MatrixRow<float>({
22.0, 80.0, 11.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 34.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 34.0
}),
MatrixRow<float>({ 40.0, 93.0, 34.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 56.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 24.0
}),
MatrixRow<float>({
37.0, 85.0, 8.0
}),
MatrixRow<float>({
22.0, 85.0, 22.0
}),
MatrixRow<float>({
2.0, 85.0, 13.0
}),
MatrixRow<float>({
37.0, 85.0, 30.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
18.0, 85.0, 35.0
}),
MatrixRow<float>({
2.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
40.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 9.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 29.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({ 37.0, 85.0, 12.0
}),
MatrixRow<float>({
22.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
18.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 38.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 54.0
}),
MatrixRow<float>({
22.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 24.0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 29.0
}),
MatrixRow<float>({
15.0, 85.0, 29.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
29.0, 85.0, 18.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
22.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 85.0, 48.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({ 8.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
8.0, 85.0, 8.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
29.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
15.0, 86.0, 30.0
}),
MatrixRow<float>({
22.0, 86.0, 29.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 70.0
}),
MatrixRow<float>({
17.0, 86.0, 24.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 60.0
}),
MatrixRow<float>({
22.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 49.0
}),
MatrixRow<float>({
8.0, 86.0, 20.0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({ 40.0, 86.0, 125.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
22.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
22.0, 86.0, 21.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 55.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
31.0, 86.0, 8.0
}),
MatrixRow<float>({
22.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
37.0, 90.0, 95.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
29.0, 90.0, 55.0
}),
MatrixRow<float>({
2.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 23.0
}),
MatrixRow<float>({ 40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 49.0
}),
MatrixRow<float>({
15.0, 90.0, 27.0
}),
MatrixRow<float>({
31.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 49.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({
40.0, 90.0, 14.0
}),
MatrixRow<float>({
17.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 84.0, 42.0
}),
MatrixRow<float>({
37.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 40.0
}),
MatrixRow<float>({
22.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({ 0.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
40.0, 84.0, 50.0
}),
MatrixRow<float>({
40.0, 84.0, 36.0
}),
MatrixRow<float>({
0.0, 84.0, 22.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 32.0
}),
MatrixRow<float>({
40.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 34.0
}),
MatrixRow<float>({
40.0, 84.0, 28.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
37.0, 84.0, 17.0
}),
MatrixRow<float>({
31.0, 84.0, 73.0
}),
MatrixRow<float>({
37.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 50.0
}),
MatrixRow<float>({
0.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 29.0
}),
MatrixRow<float>({
31.0, 84.0, 17.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 69.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({ 22.0, 92.0, 87.0
}),
MatrixRow<float>({
22.0, 92.0, 85.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 16.0
}),
MatrixRow<float>({
22.0, 92.0, 79.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
40.0, 92.0, 85.0
}),
MatrixRow<float>({
37.0, 92.0, 99.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 99.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
3.0, 91.0, 20.0
}),
MatrixRow<float>({
3.0, 91.0, 24.0
}),
MatrixRow<float>({
22.0, 90.0, 23.0
}),
MatrixRow<float>({
22.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
3.0, 90.0, 29.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({ 40.0, 90.0, 44.0
}),
MatrixRow<float>({
40.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
37.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
3.0, 90.0, 19.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
3.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
2.0, 85.0, 11.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
2.0, 85.0, 125.0
}),
MatrixRow<float>({ 37.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 19.0
}),
MatrixRow<float>({
22.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 60.0
}),
MatrixRow<float>({
40.0, 85.0, 39.0
}),
MatrixRow<float>({
0.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 21.0
}),
MatrixRow<float>({
0.0, 85.0, 14.0
}),
MatrixRow<float>({
31.0, 85.0, 25.0
}),
MatrixRow<float>({
22.0, 85.0, 19.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
36.0, 85.0, 14.0
}),
MatrixRow<float>({
31.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({ 40.0, 92.0, 42.0
}),
MatrixRow<float>({
0.0, 92.0, 62.0
}),
MatrixRow<float>({
22.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 80.0
}),
MatrixRow<float>({
0.0, 92.0, 135.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
7.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 150.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
2.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
37.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
31.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 112.0
}),
MatrixRow<float>({
37.0, 92.0, 27.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 80.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
31.0, 92.0, 18.0
}),
MatrixRow<float>({ 15.0, 92.0, 24.0
}),
MatrixRow<float>({
37.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
37.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 84.0
}),
MatrixRow<float>({
40.0, 88.0, 95.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 79.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
37.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 59.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 38.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({ 40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
8.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 80.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
37.0, 84.0, 16.0
}),
MatrixRow<float>({
0.0, 84.0, 21.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 65.0
}),
MatrixRow<float>({
15.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
0.0, 84.0, 20.0
}),
MatrixRow<float>({ 0.0, 84.0, 0
}),
MatrixRow<float>({
17.0, 84.0, 17.0
}),
MatrixRow<float>({
37.0, 84.0, 13.0
}),
MatrixRow<float>({
37.0, 84.0, 8.0
}),
MatrixRow<float>({
37.0, 84.0, 11.0
}),
MatrixRow<float>({
22.0, 84.0, 19.0
}),
MatrixRow<float>({
37.0, 83.0, 22.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 16.0
}),
MatrixRow<float>({
15.0, 83.0, 14.0
}),
MatrixRow<float>({
37.0, 83.0, 15.0
}),
MatrixRow<float>({
17.0, 83.0, 14.0
}),
MatrixRow<float>({
0.0, 83.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 49.0
}),
MatrixRow<float>({
22.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 59.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 31.0
}),
MatrixRow<float>({
39.0, 89.0, 17.0
}),
MatrixRow<float>({
21.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({ 15.0, 89.0, 75.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
37.0, 89.0, 95.0
}),
MatrixRow<float>({
37.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
21.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 24.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
15.0, 89.0, 70.0
}),
MatrixRow<float>({
15.0, 89.0, 50.0
}),
MatrixRow<float>({
15.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 28.0
}),
MatrixRow<float>({
18.0, 89.0, 30.0
}),
MatrixRow<float>({
22.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 11.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 50.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({ 15.0, 86.0, 0
}),
MatrixRow<float>({
37.0, 86.0, 19.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 28.0
}),
MatrixRow<float>({
0.0, 86.0, 11.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 10.0
}),
MatrixRow<float>({
37.0, 86.0, 11.0
}),
MatrixRow<float>({
15.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
3.0, 86.0, 15.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 37.0
}),
MatrixRow<float>({
40.0, 86.0, 21.0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 12.0
}),
MatrixRow<float>({
31.0, 86.0, 22.0
}),
MatrixRow<float>({
37.0, 86.0, 49.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({ 15.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
15.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 48.0
}),
MatrixRow<float>({
37.0, 86.0, 11.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0
}),
MatrixRow<float>({
8.0, 86.0, 10.0
}),
MatrixRow<float>({
15.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 13.0
}),
MatrixRow<float>({
18.0, 85.0, 16.0
}),
MatrixRow<float>({
37.0, 85.0, 24.0
}),
MatrixRow<float>({
40.0, 85.0, 13.0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
31.0, 85.0, 14.0
}),
MatrixRow<float>({
37.0, 85.0, 18.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
17.0, 89.0, 26.0
}),
MatrixRow<float>({
17.0, 89.0, 15.0
}),
MatrixRow<float>({ 36.0, 89.0, 53.0
}),
MatrixRow<float>({
40.0, 89.0, 21.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 30.0
}),
MatrixRow<float>({
18.0, 89.0, 37.0
}),
MatrixRow<float>({
40.0, 89.0, 130.0
}),
MatrixRow<float>({
0.0, 89.0, 25.0
}),
MatrixRow<float>({
18.0, 89.0, 19.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
18.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
3.0, 89.0, 27.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 39.0
}),
MatrixRow<float>({
17.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 54.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
22.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 40.0
}),
MatrixRow<float>({
22.0, 94.0, 220.0
}),
MatrixRow<float>({ 40.0, 94.0, 45.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 60.0
}),
MatrixRow<float>({
15.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 35.0
}),
MatrixRow<float>({
22.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
15.0, 94.0, 215.0
}),
MatrixRow<float>({
40.0, 94.0, 85.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 20.0
}),
MatrixRow<float>({
22.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
15.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 128.0
}),
MatrixRow<float>({
15.0, 94.0, 141.0
}),
MatrixRow<float>({
15.0, 94.0, 195.0
}),
MatrixRow<float>({
15.0, 93.0, 70.0
}),
MatrixRow<float>({
22.0, 93.0, 88.0
}),
MatrixRow<float>({
8.0, 89.0, 20.0
}),
MatrixRow<float>({ 3.0, 89.0, 24.0
}),
MatrixRow<float>({
3.0, 89.0, 29.0
}),
MatrixRow<float>({
3.0, 89.0, 12.0
}),
MatrixRow<float>({
17.0, 89.0, 25.0
}),
MatrixRow<float>({
31.0, 89.0, 18.0
}),
MatrixRow<float>({
8.0, 89.0, 30.0
}),
MatrixRow<float>({
8.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 16.0
}),
MatrixRow<float>({
3.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 18.0
}),
MatrixRow<float>({
31.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
15.0, 89.0, 40.0
}),
MatrixRow<float>({
17.0, 89.0, 16.0
}),
MatrixRow<float>({
17.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
3.0, 89.0, 0
}),
MatrixRow<float>({
3.0, 89.0, 19.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
17.0, 89.0, 21.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({ 40.0, 90.0, 75.0
}),
MatrixRow<float>({
17.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 16.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
31.0, 90.0, 10.0
}),
MatrixRow<float>({
40.0, 90.0, 52.0
}),
MatrixRow<float>({
17.0, 90.0, 48.0
}),
MatrixRow<float>({
17.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
22.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
8.0, 90.0, 22.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
31.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
31.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
17.0, 90.0, 22.0
}),
MatrixRow<float>({
17.0, 90.0, 34.0
}),
MatrixRow<float>({ 40.0, 90.0, 21.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
17.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
17.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 65.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
15.0, 88.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({ 15.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
15.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
36.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
8.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 55.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
15.0, 84.0, 29.0
}),
MatrixRow<float>({
2.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
15.0, 84.0, 11.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
2.0, 84.0, 22.0
}),
MatrixRow<float>({
8.0, 84.0, 12.0
}),
MatrixRow<float>({ 15.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 70.0
}),
MatrixRow<float>({
40.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
40.0, 84.0, 18.0
}),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 13.0
}),
MatrixRow<float>({
8.0, 84.0, 22.0
}),
MatrixRow<float>({
40.0, 84.0, 34.0
}),
MatrixRow<float>({
8.0, 84.0, 13.0
}),
MatrixRow<float>({
2.0, 84.0, 43.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 31.0
}),
MatrixRow<float>({
3.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
3.0, 87.0, 18.0
}),
MatrixRow<float>({
3.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 33.0
}),
MatrixRow<float>({ 40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 75.0
}),
MatrixRow<float>({
40.0, 87.0, 56.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 50.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 61.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 85.0
}),
MatrixRow<float>({
17.0, 88.0, 55.0
}),
MatrixRow<float>({
8.0, 88.0, 12.0
}),
MatrixRow<float>({
17.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 56.0
}),
MatrixRow<float>({
15.0, 88.0, 60.0
}),
MatrixRow<float>({ 40.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
8.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 100.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
8.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 55.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 25.0
}),
MatrixRow<float>({
17.0, 88.0, 92.0
}),
MatrixRow<float>({
40.0, 89.0, 55.0
}),
MatrixRow<float>({
37.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({ 40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 48.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
18.0, 89.0, 14.0
}),
MatrixRow<float>({
31.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
8.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
18.0, 89.0, 19.0
}),
MatrixRow<float>({
8.0, 89.0, 21.0
}),
MatrixRow<float>({
15.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 40.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 23.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({ 22.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
17.0, 89.0, 23.0
}),
MatrixRow<float>({
15.0, 89.0, 38.0
}),
MatrixRow<float>({
15.0, 89.0, 17.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
8.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
8.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0
}),
MatrixRow<float>({
22.0, 89.0, 39.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
29.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
17.0, 89.0, 39.0
}),
MatrixRow<float>({
17.0, 89.0, 24.0
}),
MatrixRow<float>({
3.0, 89.0, 47.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({ 40.0, 89.0, 45.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
15.0, 86.0, 67.0
}),
MatrixRow<float>({
31.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
15.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 29.0
}),
MatrixRow<float>({
37.0, 86.0, 18.0
}),
MatrixRow<float>({
2.0, 86.0, 17.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
22.0, 86.0, 30.0
}),
MatrixRow<float>({
37.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
0.0, 86.0, 14.0
}),
MatrixRow<float>({
40.0, 86.0, 17.0
}),
MatrixRow<float>({
40.0, 86.0, 29.0
}),
MatrixRow<float>({ 40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 24.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 15.0
}),
MatrixRow<float>({
31.0, 87.0, 12.0
}),
MatrixRow<float>({
31.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
37.0, 87.0, 8.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 28.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
8.0, 87.0, 22.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({
15.0, 87.0, 23.0
}),
MatrixRow<float>({ 18.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 18.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 36.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 39.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({ 15.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 50.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
37.0, 87.0, 50.0
}),
MatrixRow<float>({
15.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
15.0, 87.0, 120.0
}),
MatrixRow<float>({
15.0, 87.0, 50.0
}),
MatrixRow<float>({
15.0, 87.0, 11.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
17.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({ 40.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 14.0
}),
MatrixRow<float>({
0.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
0.0, 87.0, 12.0
}),
MatrixRow<float>({
36.0, 87.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 63.0
}),
MatrixRow<float>({
31.0, 92.0, 16.0
}),
MatrixRow<float>({
31.0, 92.0, 65.0
}),
MatrixRow<float>({
31.0, 92.0, 41.0
}),
MatrixRow<float>({
31.0, 92.0, 70.0
}),
MatrixRow<float>({ 37.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
22.0, 92.0, 25.0
}),
MatrixRow<float>({
17.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 95.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
17.0, 92.0, 26.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 37.0
}),
MatrixRow<float>({
17.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
2.0, 92.0, 42.0
}),
MatrixRow<float>({
31.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 17.0
}),
MatrixRow<float>({
31.0, 92.0, 20.0
}),
MatrixRow<float>({
15.0, 92.0, 19.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 54.0
}),
MatrixRow<float>({
22.0, 92.0, 29.0
}),
MatrixRow<float>({
31.0, 92.0, 26.0
}),
MatrixRow<float>({
0.0, 85.0, 35.0
}),
MatrixRow<float>({
37.0, 85.0, 18.0
}),
MatrixRow<float>({ 40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
37.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 85.0, 20.0
}),
MatrixRow<float>({
0.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 32.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 23.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
22.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
0.0, 84.0, 15.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 19.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
0.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
31.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 42.0
}),
MatrixRow<float>({ 40.0, 88.0, 60.0
}),
MatrixRow<float>({
40.0, 88.0, 34.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
40.0, 88.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 9.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0
}),
MatrixRow<float>({
22.0, 87.0, 48.0
}),
MatrixRow<float>({
22.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
31.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 65.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
8.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 94.0, 90.0
}),
MatrixRow<float>({
0.0, 94.0, 50.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 66.0
}),
MatrixRow<float>({ 15.0, 94.0, 132.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
22.0, 94.0, 103.0
}),
MatrixRow<float>({
40.0, 94.0, 75.0
}),
MatrixRow<float>({
40.0, 94.0, 175.0
}),
MatrixRow<float>({
15.0, 94.0, 305.0
}),
MatrixRow<float>({
40.0, 94.0, 150.0
}),
MatrixRow<float>({
31.0, 94.0, 25.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
31.0, 94.0, 68.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 125.0
}),
MatrixRow<float>({
31.0, 94.0, 55.0
}),
MatrixRow<float>({
15.0, 94.0, 129.0
}),
MatrixRow<float>({
15.0, 94.0, 95.0
}),
MatrixRow<float>({
15.0, 94.0, 0
}),
MatrixRow<float>({
15.0, 94.0, 62.0
}),
MatrixRow<float>({
22.0, 94.0, 106.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 94.0, 47.0
}),
MatrixRow<float>({
22.0, 94.0, 96.0
}),
MatrixRow<float>({
40.0, 94.0, 145.0
}),
MatrixRow<float>({
0.0, 94.0, 90.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({ 2.0, 92.0, 17.0
}),
MatrixRow<float>({
40.0, 92.0, 74.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 100.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
31.0, 92.0, 54.0
}),
MatrixRow<float>({
31.0, 92.0, 0
}),
MatrixRow<float>({
31.0, 92.0, 45.0
}),
MatrixRow<float>({
31.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
2.0, 92.0, 45.0
}),
MatrixRow<float>({
2.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
2.0, 92.0, 27.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({ 40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 22.0
}),
MatrixRow<float>({
40.0, 92.0, 52.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
17.0, 85.0, 9.0
}),
MatrixRow<float>({
8.0, 85.0, 14.0
}),
MatrixRow<float>({
2.0, 85.0, 20.0
}),
MatrixRow<float>({
8.0, 85.0, 11.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
31.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
22.0, 85.0, 60.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
31.0, 85.0, 14.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
31.0, 85.0, 17.0
}),
MatrixRow<float>({
31.0, 85.0, 9.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
31.0, 84.0, 12.0
}),
MatrixRow<float>({ 40.0, 84.0, 85.0
}),
MatrixRow<float>({
15.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 7.0
}),
MatrixRow<float>({
40.0, 84.0, 75.0
}),
MatrixRow<float>({
37.0, 84.0, 14.0
}),
MatrixRow<float>({
2.0, 84.0, 14.0
}),
MatrixRow<float>({
37.0, 84.0, 15.0
}),
MatrixRow<float>({
31.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 12.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 40.0
}),
MatrixRow<float>({
22.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({ 40.0, 86.0, 18.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 23.0
}),
MatrixRow<float>({
36.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
22.0, 86.0, 14.0
}),
MatrixRow<float>({
15.0, 86.0, 10.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 26.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
40.0, 86.0, 19.0
}),
MatrixRow<float>({
7.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 18.0
}),
MatrixRow<float>({
15.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
36.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({ 40.0, 85.0, 18.0
}),
MatrixRow<float>({
36.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 22.0
}),
MatrixRow<float>({
40.0, 88.0, 29.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
0.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 50.0
}),
MatrixRow<float>({
22.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
37.0, 88.0, 16.0
}),
MatrixRow<float>({
0.0, 88.0, 11.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
17.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 88.0, 36.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 36.0
}),
MatrixRow<float>({ 22.0, 88.0, 39.0
}),
MatrixRow<float>({
37.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 49.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
17.0, 88.0, 14.0
}),
MatrixRow<float>({
37.0, 88.0, 9.0
}),
MatrixRow<float>({
40.0, 88.0, 13.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
17.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 35.0
}),
MatrixRow<float>({
15.0, 85.0, 60.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 23.0
}),
MatrixRow<float>({
8.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 26.0
}),
MatrixRow<float>({
3.0, 85.0, 0
}),
MatrixRow<float>({
22.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
3.0, 85.0, 0
}),
MatrixRow<float>({
40.0, 85.0, 30.0
}),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
22.0, 85.0, 23.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
8.0, 85.0, 15.0
}),
MatrixRow<float>({ 40.0, 85.0, 26.0
}),
MatrixRow<float>({
40.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 0
}),
MatrixRow<float>({
37.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
8.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
17.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
40.0, 87.0, 52.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
2.0, 87.0, 20.0
}),
MatrixRow<float>({
2.0, 87.0, 20.0
}),
MatrixRow<float>({
36.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({ 22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 26.0
}),
MatrixRow<float>({
29.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 60.0
}),
MatrixRow<float>({
29.0, 87.0, 14.0
}),
MatrixRow<float>({
29.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
8.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 110.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 46.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({ 40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
22.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 31.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
2.0, 88.0, 16.0
}),
MatrixRow<float>({
2.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
8.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
22.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
0.0, 94.0, 50.0
}),
MatrixRow<float>({
15.0, 94.0, 44.0
}),
MatrixRow<float>({
40.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 55.0
}),
MatrixRow<float>({
40.0, 94.0, 65.0
}),
MatrixRow<float>({
40.0, 94.0, 49.0
}),
MatrixRow<float>({
15.0, 94.0, 25.0
}),
MatrixRow<float>({
15.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 62.0
}),
MatrixRow<float>({ 15.0, 94.0, 40.0
}),
MatrixRow<float>({
15.0, 94.0, 100.0
}),
MatrixRow<float>({
15.0, 94.0, 70.0
}),
MatrixRow<float>({
15.0, 94.0, 20.0
}),
MatrixRow<float>({
37.0, 94.0, 0
}),
MatrixRow<float>({
40.0, 94.0, 155.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 60.0
}),
MatrixRow<float>({
40.0, 94.0, 62.0
}),
MatrixRow<float>({
40.0, 94.0, 49.0
}),
MatrixRow<float>({
22.0, 94.0, 35.0
}),
MatrixRow<float>({
40.0, 94.0, 90.0
}),
MatrixRow<float>({
40.0, 94.0, 70.0
}),
MatrixRow<float>({
40.0, 94.0, 48.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 63.0
}),
MatrixRow<float>({
40.0, 94.0, 95.0
}),
MatrixRow<float>({
40.0, 94.0, 80.0
}),
MatrixRow<float>({
40.0, 94.0, 50.0
}),
MatrixRow<float>({
37.0, 94.0, 68.0
}),
MatrixRow<float>({
40.0, 94.0, 105.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 57.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({ 22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 31.0
}),
MatrixRow<float>({
22.0, 92.0, 44.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 125.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 85.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 18.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
22.0, 92.0, 60.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
15.0, 92.0, 96.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
3.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({ 40.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 49.0
}),
MatrixRow<float>({
8.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
3.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 50.0
}),
MatrixRow<float>({
15.0, 92.0, 55.0
}),
MatrixRow<float>({
15.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 92.0, 26.0
}),
MatrixRow<float>({
3.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
40.0, 92.0, 75.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
3.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 24.0
}),
MatrixRow<float>({ 40.0, 90.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 80.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
17.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 45.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
37.0, 90.0, 17.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 27.0
}),
MatrixRow<float>({
15.0, 90.0, 59.0
}),
MatrixRow<float>({
35.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 90.0, 17.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
-1.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0 }),
MatrixRow<float>({
8.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 90.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
22.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 60.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 47.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
31.0, 91.0, 22.0
}),
MatrixRow<float>({
31.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
36.0, 91.0, 51.0
}),
MatrixRow<float>({
40.0, 91.0, 27.0
}),
MatrixRow<float>({
22.0, 91.0, 85.0
}),
MatrixRow<float>({
22.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 26.0 }),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
31.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 90.0
}),
MatrixRow<float>({
40.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 90.0, 37.0
}),
MatrixRow<float>({
0.0, 89.0, 22.0
}),
MatrixRow<float>({
15.0, 89.0, 26.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
0.0, 89.0, 14.0
}),
MatrixRow<float>({
40.0, 89.0, 22.0
}),
MatrixRow<float>({
37.0, 89.0, 0
}),
MatrixRow<float>({
37.0, 89.0, 0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
15.0, 89.0, 103.0
}),
MatrixRow<float>({
15.0, 89.0, 12.0
}),
MatrixRow<float>({
40.0, 89.0, 50.0
}),
MatrixRow<float>({
22.0, 89.0, 16.0
}),
MatrixRow<float>({
15.0, 89.0, 29.0
}),
MatrixRow<float>({
40.0, 89.0, 11.0
}),
MatrixRow<float>({
40.0, 89.0, 29.0 }),
MatrixRow<float>({
22.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 45.0
}),
MatrixRow<float>({
37.0, 89.0, 13.0
}),
MatrixRow<float>({
37.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
15.0, 89.0, 20.0
}),
MatrixRow<float>({
37.0, 89.0, 28.0
}),
MatrixRow<float>({
15.0, 89.0, 14.0
}),
MatrixRow<float>({
37.0, 89.0, 25.0
}),
MatrixRow<float>({
37.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 88.0, 16.0
}),
MatrixRow<float>({
17.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 125.0
}),
MatrixRow<float>({
17.0, 88.0, 10.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
8.0, 88.0, 44.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 12.0
}),
MatrixRow<float>({
18.0, 88.0, 12.0 }),
MatrixRow<float>({
37.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 59.0
}),
MatrixRow<float>({
40.0, 88.0, 66.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
31.0, 88.0, 8.0
}),
MatrixRow<float>({
40.0, 88.0, 30.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
37.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
15.0, 88.0, 16.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 45.0
}),
MatrixRow<float>({
22.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
29.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 82.0
}),
MatrixRow<float>({
15.0, 90.0, 105.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 28.0 }),
MatrixRow<float>({
15.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
29.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
36.0, 90.0, 30.0
}),
MatrixRow<float>({
3.0, 90.0, 24.0
}),
MatrixRow<float>({
15.0, 90.0, 72.0
}),
MatrixRow<float>({
22.0, 90.0, 27.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
3.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 52.0
}),
MatrixRow<float>({
3.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 27.0
}),
MatrixRow<float>({
40.0, 85.0, 8.0 }),
MatrixRow<float>({
40.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 8.0
}),
MatrixRow<float>({
37.0, 85.0, 14.0
}),
MatrixRow<float>({
0.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 75.0
}),
MatrixRow<float>({
0.0, 85.0, 17.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 85.0, 46.0
}),
MatrixRow<float>({
15.0, 85.0, 9.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
10.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
6.0, 87.0, 8.0
}),
MatrixRow<float>({
8.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 35.0
}),
MatrixRow<float>({
15.0, 87.0, 60.0
}),
MatrixRow<float>({
29.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
10.0, 87.0, 13.0 }),
MatrixRow<float>({
10.0, 87.0, 13.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
40.0, 87.0, 49.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 9.0
}),
MatrixRow<float>({
40.0, 86.0, 30.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 70.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 32.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 21.0
}),
MatrixRow<float>({
15.0, 92.0, 15.0
}),
MatrixRow<float>({
15.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
31.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
8.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0 }),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
22.0, 92.0, 108.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
31.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 44.0
}),
MatrixRow<float>({
31.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 0
}),
MatrixRow<float>({
31.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
21.0, 89.0, 65.0
}),
MatrixRow<float>({
21.0, 89.0, 50.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 13.0
}),
MatrixRow<float>({
10.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 60.0
}),
MatrixRow<float>({
40.0, 89.0, 52.0
}),
MatrixRow<float>({
15.0, 89.0, 26.0
}),
MatrixRow<float>({
22.0, 89.0, 23.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
29.0, 88.0, 35.0 }),
MatrixRow<float>({
22.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 10.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
21.0, 88.0, 23.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
31.0, 88.0, 40.0
}),
MatrixRow<float>({
21.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
40.0, 88.0, 28.0
}),
MatrixRow<float>({
22.0, 88.0, 0
}),
MatrixRow<float>({
22.0, 88.0, 12.0
}),
MatrixRow<float>({
40.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 8.0
}),
MatrixRow<float>({
40.0, 88.0, 38.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
40.0, 88.0, 33.0
}),
MatrixRow<float>({
40.0, 88.0, 48.0
}),
MatrixRow<float>({
22.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
37.0, 91.0, 27.0
}),
MatrixRow<float>({
31.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 33.0 }),
MatrixRow<float>({
22.0, 91.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 75.0
}),
MatrixRow<float>({
15.0, 91.0, 21.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
31.0, 91.0, 10.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
31.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 12.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 23.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0
}),
MatrixRow<float>({
15.0, 91.0, 85.0
}),
MatrixRow<float>({
37.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
31.0, 83.0, 7.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0 }),
MatrixRow<float>({
40.0, 83.0, 38.0
}),
MatrixRow<float>({
37.0, 83.0, 17.0
}),
MatrixRow<float>({
22.0, 83.0, 10.0
}),
MatrixRow<float>({
8.0, 83.0, 22.0
}),
MatrixRow<float>({
40.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 17.0
}),
MatrixRow<float>({
15.0, 83.0, 10.0
}),
MatrixRow<float>({
22.0, 83.0, 16.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
40.0, 82.0, 32.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 17.0
}),
MatrixRow<float>({
8.0, 82.0, 16.0
}),
MatrixRow<float>({
40.0, 82.0, 19.0
}),
MatrixRow<float>({
21.0, 82.0, 28.0
}),
MatrixRow<float>({
40.0, 82.0, 10.0
}),
MatrixRow<float>({
31.0, 82.0, 0
}),
MatrixRow<float>({
22.0, 82.0, 16.0
}),
MatrixRow<float>({
40.0, 82.0, 34.0
}),
MatrixRow<float>({
8.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 60.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
40.0, 82.0, 27.0
}),
MatrixRow<float>({
40.0, 82.0, 20.0 }),
MatrixRow<float>({
40.0, 82.0, 19.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
8.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 24.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
22.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
15.0, 87.0, 0
}),
MatrixRow<float>({
15.0, 87.0, 30.0
}),
MatrixRow<float>({
15.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 10.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 35.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 34.0
}),
MatrixRow<float>({
22.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0 }),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
37.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
15.0, 87.0, 13.0
}),
MatrixRow<float>({
31.0, 88.0, 28.0
}),
MatrixRow<float>({
3.0, 88.0, 22.0
}),
MatrixRow<float>({
37.0, 88.0, 20.0
}),
MatrixRow<float>({
22.0, 88.0, 40.0
}),
MatrixRow<float>({
15.0, 88.0, 18.0
}),
MatrixRow<float>({
22.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 16.0
}),
MatrixRow<float>({
31.0, 88.0, 9.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
17.0, 88.0, 24.0
}),
MatrixRow<float>({
3.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
3.0, 88.0, 19.0
}),
MatrixRow<float>({
40.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 21.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0
}),
MatrixRow<float>({
22.0, 88.0, 22.0 }),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 30.0
}),
MatrixRow<float>({
22.0, 88.0, 28.0
}),
MatrixRow<float>({
8.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 39.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 60.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
3.0, 91.0, 34.0
}),
MatrixRow<float>({
15.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 95.0
}),
MatrixRow<float>({
37.0, 91.0, 25.0
}),
MatrixRow<float>({
15.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
40.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 34.0
}),
MatrixRow<float>({
22.0, 91.0, 50.0 }),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
36.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 60.0
}),
MatrixRow<float>({
2.0, 91.0, 30.0
}),
MatrixRow<float>({
37.0, 91.0, 40.0
}),
MatrixRow<float>({
3.0, 91.0, 41.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
37.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 55.0
}),
MatrixRow<float>({
31.0, 92.0, 110.0
}),
MatrixRow<float>({
8.0, 92.0, 65.0
}),
MatrixRow<float>({
15.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 46.0
}),
MatrixRow<float>({
22.0, 92.0, 15.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0 }),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 58.0
}),
MatrixRow<float>({
22.0, 92.0, 38.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
22.0, 92.0, 95.0
}),
MatrixRow<float>({
31.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
42.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 21.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
15.0, 92.0, 60.0
}),
MatrixRow<float>({
15.0, 92.0, 35.0
}),
MatrixRow<float>({
15.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 29.0
}),
MatrixRow<float>({
15.0, 92.0, 30.0
}),
MatrixRow<float>({
2.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 55.0
}),
MatrixRow<float>({
22.0, 92.0, 119.0
}),
MatrixRow<float>({
40.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 17.0
}),
MatrixRow<float>({
37.0, 86.0, 16.0
}),
MatrixRow<float>({
15.0, 86.0, 13.0
}),
MatrixRow<float>({
37.0, 86.0, 15.0 }),
MatrixRow<float>({
0.0, 86.0, 18.0
}),
MatrixRow<float>({
37.0, 86.0, 10.0
}),
MatrixRow<float>({
0.0, 86.0, 15.0
}),
MatrixRow<float>({
0.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 42.0
}),
MatrixRow<float>({
15.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
15.0, 86.0, 39.0
}),
MatrixRow<float>({
37.0, 86.0, 25.0
}),
MatrixRow<float>({
15.0, 86.0, 20.0
}),
MatrixRow<float>({
37.0, 86.0, 23.0
}),
MatrixRow<float>({
37.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 27.0
}),
MatrixRow<float>({
3.0, 86.0, 0
}),
MatrixRow<float>({
0.0, 86.0, 16.0
}),
MatrixRow<float>({
40.0, 86.0, 25.0
}),
MatrixRow<float>({
40.0, 86.0, 15.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 26.0
}),
MatrixRow<float>({
37.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0 }),
MatrixRow<float>({
3.0, 90.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 135.0
}),
MatrixRow<float>({
36.0, 90.0, 22.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
37.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
37.0, 90.0, 15.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
22.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
3.0, 87.0, 24.0 }),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 16.0
}),
MatrixRow<float>({
29.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
15.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
3.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 11.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 22.0
}),
MatrixRow<float>({
22.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 15.0
}),
MatrixRow<float>({
8.0, 87.0, 11.0
}),
MatrixRow<float>({
3.0, 87.0, 33.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
22.0, 87.0, 26.0
}),
MatrixRow<float>({
22.0, 87.0, 0 }),
MatrixRow<float>({
31.0, 87.0, 16.0
}),
MatrixRow<float>({
22.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
31.0, 86.0, 12.0
}),
MatrixRow<float>({
31.0, 86.0, 13.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
8.0, 86.0, 13.0
}),
MatrixRow<float>({
22.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 14.0
}),
MatrixRow<float>({
2.0, 86.0, 8.0
}),
MatrixRow<float>({
22.0, 86.0, 35.0
}),
MatrixRow<float>({
8.0, 86.0, 18.0
}),
MatrixRow<float>({
8.0, 86.0, 11.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 34.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
8.0, 86.0, 12.0
}),
MatrixRow<float>({
40.0, 86.0, 22.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
40.0, 86.0, 35.0
}),
MatrixRow<float>({
15.0, 86.0, 25.0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 86.0, 35.0 }),
MatrixRow<float>({
22.0, 86.0, 32.0
}),
MatrixRow<float>({
29.0, 85.0, 14.0
}),
MatrixRow<float>({
40.0, 85.0, 16.0
}),
MatrixRow<float>({
32.0, 85.0, 7.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 25.0
}),
MatrixRow<float>({
40.0, 85.0, 35.0
}),
MatrixRow<float>({
40.0, 85.0, 22.0
}),
MatrixRow<float>({
29.0, 85.0, 20.0
}),
MatrixRow<float>({
31.0, 85.0, 10.0
}),
MatrixRow<float>({
21.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
18.0, 85.0, 18.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
6.0, 85.0, 9.0
}),
MatrixRow<float>({
29.0, 85.0, 28.0
}),
MatrixRow<float>({
22.0, 84.0, 18.0
}),
MatrixRow<float>({
29.0, 84.0, 18.0
}),
MatrixRow<float>({
18.0, 84.0, 17.0
}),
MatrixRow<float>({
40.0, 84.0, 30.0
}),
MatrixRow<float>({
31.0, 84.0, 11.0
}),
MatrixRow<float>({
29.0, 84.0, 32.0
}),
MatrixRow<float>({
18.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0 }),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 14.0
}),
MatrixRow<float>({
0.0, 83.0, 25.0
}),
MatrixRow<float>({
37.0, 83.0, 27.0
}),
MatrixRow<float>({
31.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 83.0, 13.0
}),
MatrixRow<float>({
0.0, 83.0, 13.0
}),
MatrixRow<float>({
15.0, 83.0, 63.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 31.0
}),
MatrixRow<float>({
0.0, 83.0, 15.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
40.0, 83.0, 15.0
}),
MatrixRow<float>({
40.0, 83.0, 30.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
0.0, 83.0, 15.0
}),
MatrixRow<float>({
15.0, 83.0, 12.0
}),
MatrixRow<float>({
37.0, 83.0, 10.0
}),
MatrixRow<float>({
37.0, 83.0, 13.0
}),
MatrixRow<float>({
40.0, 83.0, 17.0
}),
MatrixRow<float>({
40.0, 83.0, 13.0
}),
MatrixRow<float>({
37.0, 83.0, 9.0
}),
MatrixRow<float>({
37.0, 83.0, 12.0 }),
MatrixRow<float>({
0.0, 83.0, 22.0
}),
MatrixRow<float>({
31.0, 83.0, 8.0
}),
MatrixRow<float>({
37.0, 82.0, 27.0
}),
MatrixRow<float>({
40.0, 82.0, 14.0
}),
MatrixRow<float>({
15.0, 82.0, 18.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
37.0, 82.0, 10.0
}),
MatrixRow<float>({
3.0, 90.0, 19.0
}),
MatrixRow<float>({
17.0, 90.0, 80.0
}),
MatrixRow<float>({
17.0, 90.0, 0
}),
MatrixRow<float>({
2.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 26.0
}),
MatrixRow<float>({
40.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 54.0
}),
MatrixRow<float>({
40.0, 90.0, 15.0
}),
MatrixRow<float>({
40.0, 90.0, 39.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
17.0, 90.0, 23.0
}),
MatrixRow<float>({
37.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
0.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
15.0, 90.0, 22.0
}),
MatrixRow<float>({
15.0, 90.0, 15.0
}),
MatrixRow<float>({
15.0, 90.0, 0 }),
MatrixRow<float>({
22.0, 90.0, 33.0
}),
MatrixRow<float>({
22.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 92.0, 65.0
}),
MatrixRow<float>({
40.0, 92.0, 70.0
}),
MatrixRow<float>({
22.0, 92.0, 0
}),
MatrixRow<float>({
40.0, 92.0, 33.0
}),
MatrixRow<float>({
22.0, 92.0, 69.0
}),
MatrixRow<float>({
2.0, 92.0, 29.0
}),
MatrixRow<float>({
40.0, 92.0, 29.0
}),
MatrixRow<float>({
22.0, 92.0, 70.0
}),
MatrixRow<float>({
31.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
2.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 32.0
}),
MatrixRow<float>({
3.0, 92.0, 58.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
40.0, 92.0, 30.0
}),
MatrixRow<float>({
40.0, 92.0, 42.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
22.0, 92.0, 75.0
}),
MatrixRow<float>({
31.0, 92.0, 25.0 }),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 60.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 20.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 20.0
}),
MatrixRow<float>({
40.0, 85.0, 18.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 20.0
}),
MatrixRow<float>({
22.0, 85.0, 17.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
15.0, 85.0, 14.0
}),
MatrixRow<float>({
22.0, 85.0, 10.0
}),
MatrixRow<float>({
40.0, 85.0, 19.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0
}),
MatrixRow<float>({
40.0, 85.0, 40.0
}),
MatrixRow<float>({
0.0, 85.0, 12.0
}),
MatrixRow<float>({
37.0, 85.0, 9.0
}),
MatrixRow<float>({
15.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 28.0
}),
MatrixRow<float>({
40.0, 85.0, 50.0
}),
MatrixRow<float>({
40.0, 85.0, 15.0
}),
MatrixRow<float>({
15.0, 85.0, 13.0 }),
MatrixRow<float>({
22.0, 85.0, 15.0
}),
MatrixRow<float>({
40.0, 85.0, 38.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 22.0
}),
MatrixRow<float>({
40.0, 85.0, 45.0
}),
MatrixRow<float>({
15.0, 85.0, 8.0
}),
MatrixRow<float>({
40.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 10.0
}),
MatrixRow<float>({
15.0, 85.0, 11.0
}),
MatrixRow<float>({
22.0, 91.0, 35.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
3.0, 91.0, 21.0
}),
MatrixRow<float>({
22.0, 91.0, 52.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0
}),
MatrixRow<float>({
22.0, 91.0, 30.0
}),
MatrixRow<float>({
2.0, 91.0, 22.0
}),
MatrixRow<float>({
36.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 63.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 42.0
}),
MatrixRow<float>({
8.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
8.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 160.0
}),
MatrixRow<float>({
15.0, 91.0, 35.0 }),
MatrixRow<float>({
40.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
3.0, 91.0, 34.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
15.0, 91.0, 59.0
}),
MatrixRow<float>({
36.0, 91.0, 50.0
}),
MatrixRow<float>({
22.0, 91.0, 100.0
}),
MatrixRow<float>({
8.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
8.0, 90.0, 24.0
}),
MatrixRow<float>({
8.0, 90.0, 19.0
}),
MatrixRow<float>({
22.0, 90.0, 30.0
}),
MatrixRow<float>({
8.0, 88.0, 25.0
}),
MatrixRow<float>({
37.0, 88.0, 17.0
}),
MatrixRow<float>({
22.0, 88.0, 23.0
}),
MatrixRow<float>({
40.0, 88.0, 37.0
}),
MatrixRow<float>({
3.0, 88.0, 15.0
}),
MatrixRow<float>({
22.0, 88.0, 20.0
}),
MatrixRow<float>({
8.0, 88.0, 80.0
}),
MatrixRow<float>({
8.0, 88.0, 17.0
}),
MatrixRow<float>({
31.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
31.0, 88.0, 12.0
}),
MatrixRow<float>({
17.0, 88.0, 30.0
}),
MatrixRow<float>({
17.0, 88.0, 20.0 }),
MatrixRow<float>({
40.0, 88.0, 17.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
3.0, 88.0, 0
}),
MatrixRow<float>({
8.0, 88.0, 41.0
}),
MatrixRow<float>({
37.0, 88.0, 10.0
}),
MatrixRow<float>({
31.0, 88.0, 11.0
}),
MatrixRow<float>({
8.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 22.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 45.0
}),
MatrixRow<float>({
40.0, 88.0, 40.0
}),
MatrixRow<float>({
7.0, 88.0, 25.0
}),
MatrixRow<float>({
15.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 18.0
}),
MatrixRow<float>({
40.0, 88.0, 24.0
}),
MatrixRow<float>({
22.0, 88.0, 45.0
}),
MatrixRow<float>({
15.0, 88.0, 27.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 13.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 20.0 }),
MatrixRow<float>({
15.0, 88.0, 19.0
}),
MatrixRow<float>({
15.0, 88.0, 30.0
}),
MatrixRow<float>({
37.0, 88.0, 32.0
}),
MatrixRow<float>({
40.0, 88.0, 35.0
}),
MatrixRow<float>({
15.0, 88.0, 14.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
40.0, 88.0, 20.0
}),
MatrixRow<float>({
15.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 88.0, 0
}),
MatrixRow<float>({
15.0, 88.0, 26.0
}),
MatrixRow<float>({
40.0, 88.0, 25.0
}),
MatrixRow<float>({
31.0, 88.0, 40.0
}),
MatrixRow<float>({
40.0, 88.0, 15.0
}),
MatrixRow<float>({
31.0, 88.0, 15.0
}),
MatrixRow<float>({
15.0, 93.0, 140.0
}),
MatrixRow<float>({
15.0, 93.0, 200.0
}),
MatrixRow<float>({
40.0, 93.0, 60.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
40.0, 92.0, 25.0
}),
MatrixRow<float>({
15.0, 92.0, 43.0
}),
MatrixRow<float>({
40.0, 92.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 48.0
}),
MatrixRow<float>({
40.0, 92.0, 34.0
}),
MatrixRow<float>({
15.0, 92.0, 56.0 }),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 39.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
40.0, 92.0, 40.0
}),
MatrixRow<float>({
15.0, 92.0, 179.0
}),
MatrixRow<float>({
2.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
15.0, 91.0, 68.0
}),
MatrixRow<float>({
15.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 36.0
}),
MatrixRow<float>({
40.0, 91.0, 30.0
}),
MatrixRow<float>({
15.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
17.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
0.0, 87.0, 13.0
}),
MatrixRow<float>({
0.0, 87.0, 18.0 }),
MatrixRow<float>({
0.0, 87.0, 0
}),
MatrixRow<float>({
0.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
15.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 18.0
}),
MatrixRow<float>({
15.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 34.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
0.0, 87.0, 34.0
}),
MatrixRow<float>({
40.0, 87.0, 30.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
0.0, 87.0, 10.0
}),
MatrixRow<float>({
3.0, 87.0, 0
}),
MatrixRow<float>({
22.0, 86.0, 0
}),
MatrixRow<float>({
40.0, 90.0, 16.0
}),
MatrixRow<float>({
40.0, 90.0, 32.0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 38.0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
8.0, 90.0, 17.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
40.0, 90.0, 125.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0 }),
MatrixRow<float>({
2.0, 89.0, 125.0
}),
MatrixRow<float>({
40.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 34.0
}),
MatrixRow<float>({
40.0, 89.0, 35.0
}),
MatrixRow<float>({
40.0, 89.0, 19.0
}),
MatrixRow<float>({
3.0, 89.0, 65.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
15.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 100.0
}),
MatrixRow<float>({
3.0, 89.0, 29.0
}),
MatrixRow<float>({
7.0, 89.0, 35.0
}),
MatrixRow<float>({
22.0, 89.0, 36.0
}),
MatrixRow<float>({
22.0, 89.0, 22.0
}),
MatrixRow<float>({
40.0, 89.0, 85.0
}),
MatrixRow<float>({
2.0, 89.0, 22.0
}),
MatrixRow<float>({
36.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 70.0
}),
MatrixRow<float>({
15.0, 90.0, 41.0
}),
MatrixRow<float>({
15.0, 90.0, 14.0
}),
MatrixRow<float>({
15.0, 90.0, 17.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
22.0, 90.0, 50.0 }),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 70.0
}),
MatrixRow<float>({
37.0, 90.0, 28.0
}),
MatrixRow<float>({
22.0, 90.0, 48.0
}),
MatrixRow<float>({
36.0, 90.0, 20.0
}),
MatrixRow<float>({
36.0, 90.0, 35.0
}),
MatrixRow<float>({
37.0, 90.0, 144.0
}),
MatrixRow<float>({
22.0, 90.0, 75.0
}),
MatrixRow<float>({
40.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 42.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
22.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
37.0, 90.0, 32.0
}),
MatrixRow<float>({
40.0, 90.0, 19.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
15.0, 90.0, 70.0
}),
MatrixRow<float>({
40.0, 90.0, 125.0
}),
MatrixRow<float>({
40.0, 90.0, 50.0
}),
MatrixRow<float>({
22.0, 90.0, 90.0
}),
MatrixRow<float>({
22.0, 90.0, 46.0
}),
MatrixRow<float>({
40.0, 90.0, 28.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 25.0 }),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 29.0
}),
MatrixRow<float>({
15.0, 90.0, 25.0
}),
MatrixRow<float>({
15.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 34.0
}),
MatrixRow<float>({
40.0, 90.0, 18.0
}),
MatrixRow<float>({
22.0, 90.0, 60.0
}),
MatrixRow<float>({
3.0, 90.0, 17.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
3.0, 90.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 79.0
}),
MatrixRow<float>({
37.0, 89.0, 44.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
15.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 15.0
}),
MatrixRow<float>({
40.0, 89.0, 25.0
}),
MatrixRow<float>({
22.0, 89.0, 28.0
}),
MatrixRow<float>({
40.0, 89.0, 18.0
}),
MatrixRow<float>({
3.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 28.0
}),
MatrixRow<float>({
3.0, 89.0, 20.0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
40.0, 89.0, 65.0
}),
MatrixRow<float>({
40.0, 89.0, 66.0
}),
MatrixRow<float>({
22.0, 89.0, 37.0 }),
MatrixRow<float>({
37.0, 89.0, 14.0
}),
MatrixRow<float>({
3.0, 89.0, 15.0
}),
MatrixRow<float>({
22.0, 92.0, 50.0
}),
MatrixRow<float>({
22.0, 92.0, 30.0
}),
MatrixRow<float>({
22.0, 92.0, 90.0
}),
MatrixRow<float>({
40.0, 91.0, 125.0
}),
MatrixRow<float>({
22.0, 91.0, 17.0
}),
MatrixRow<float>({
15.0, 91.0, 22.0
}),
MatrixRow<float>({
15.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
15.0, 91.0, 20.0
}),
MatrixRow<float>({
15.0, 91.0, 70.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
15.0, 91.0, 19.0
}),
MatrixRow<float>({
22.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 38.0
}),
MatrixRow<float>({
22.0, 91.0, 18.0
}),
MatrixRow<float>({
15.0, 91.0, 16.0
}),
MatrixRow<float>({
15.0, 91.0, 17.0
}),
MatrixRow<float>({
8.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 65.0
}),
MatrixRow<float>({
15.0, 91.0, 24.0
}),
MatrixRow<float>({
40.0, 91.0, 29.0 }),
MatrixRow<float>({
40.0, 91.0, 58.0
}),
MatrixRow<float>({
15.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 16.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
2.0, 89.0, 19.0
}),
MatrixRow<float>({
37.0, 89.0, 36.0
}),
MatrixRow<float>({
22.0, 89.0, 19.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 30.0
}),
MatrixRow<float>({
2.0, 89.0, 15.0
}),
MatrixRow<float>({
31.0, 89.0, 12.0
}),
MatrixRow<float>({
22.0, 89.0, 0
}),
MatrixRow<float>({
40.0, 89.0, 45.0
}),
MatrixRow<float>({
17.0, 89.0, 33.0
}),
MatrixRow<float>({
22.0, 89.0, 25.0
}),
MatrixRow<float>({
40.0, 89.0, 42.0
}),
MatrixRow<float>({
2.0, 89.0, 35.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
15.0, 89.0, 16.0
}),
MatrixRow<float>({
40.0, 89.0, 32.0
}),
MatrixRow<float>({
31.0, 89.0, 0
}),
MatrixRow<float>({
36.0, 89.0, 27.0
}),
MatrixRow<float>({
15.0, 89.0, 13.0
}),
MatrixRow<float>({
40.0, 89.0, 80.0
}),
MatrixRow<float>({
40.0, 89.0, 24.0 }),
MatrixRow<float>({
31.0, 89.0, 20.0
}),
MatrixRow<float>({
22.0, 89.0, 38.0
}),
MatrixRow<float>({
31.0, 84.0, 8.0
}),
MatrixRow<float>({
31.0, 84.0, 0
}),
MatrixRow<float>({
32.0, 84.0, 9.0
}),
MatrixRow<float>({
15.0, 84.0, 45.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 17.0
}),
MatrixRow<float>({
15.0, 84.0, 12.0
}),
MatrixRow<float>({
15.0, 84.0, 35.0
}),
MatrixRow<float>({
31.0, 84.0, 18.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
42.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 12.0
}),
MatrixRow<float>({
40.0, 84.0, 16.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
15.0, 84.0, 16.0
}),
MatrixRow<float>({
22.0, 84.0, 20.0
}),
MatrixRow<float>({
31.0, 84.0, 9.0
}),
MatrixRow<float>({
31.0, 84.0, 18.0
}),
MatrixRow<float>({
31.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 25.0
}),
MatrixRow<float>({
40.0, 84.0, 19.0
}),
MatrixRow<float>({
40.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 14.0 }),
MatrixRow<float>({
15.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 20.0
}),
MatrixRow<float>({
15.0, 84.0, 19.0
}),
MatrixRow<float>({
15.0, 84.0, 35.0
}),
MatrixRow<float>({
40.0, 84.0, 10.0
}),
MatrixRow<float>({
40.0, 84.0, 95.0
}),
MatrixRow<float>({
40.0, 84.0, 15.0
}),
MatrixRow<float>({
40.0, 93.0, 29.0
}),
MatrixRow<float>({
40.0, 93.0, 30.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 103.0
}),
MatrixRow<float>({
40.0, 93.0, 40.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
15.0, 93.0, 80.0
}),
MatrixRow<float>({
15.0, 93.0, 50.0
}),
MatrixRow<float>({
37.0, 93.0, 40.0
}),
MatrixRow<float>({
40.0, 93.0, 42.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 70.0
}),
MatrixRow<float>({
40.0, 93.0, 55.0
}),
MatrixRow<float>({
15.0, 93.0, 54.0
}),
MatrixRow<float>({
15.0, 93.0, 84.0
}),
MatrixRow<float>({
40.0, 93.0, 32.0
}),
MatrixRow<float>({
40.0, 93.0, 24.0 }),
MatrixRow<float>({
40.0, 93.0, 18.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 93.0, 36.0
}),
MatrixRow<float>({
40.0, 93.0, 75.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
15.0, 93.0, 90.0
}),
MatrixRow<float>({
40.0, 93.0, 45.0
}),
MatrixRow<float>({
15.0, 93.0, 45.0
}),
MatrixRow<float>({
40.0, 93.0, 52.0
}),
MatrixRow<float>({
15.0, 93.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 46.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
22.0, 84.0, 15.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 14.0
}),
MatrixRow<float>({
0.0, 84.0, 12.0
}),
MatrixRow<float>({
2.0, 84.0, 20.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
22.0, 84.0, 14.0
}),
MatrixRow<float>({
22.0, 84.0, 0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
40.0, 84.0, 26.0
}),
MatrixRow<float>({
8.0, 84.0, 9.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 9.0 }),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
15.0, 84.0, 0
}),
MatrixRow<float>({
0.0, 84.0, 13.0
}),
MatrixRow<float>({
0.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 11.0
}),
MatrixRow<float>({
8.0, 84.0, 10.0
}),
MatrixRow<float>({
8.0, 84.0, 20.0
}),
MatrixRow<float>({
2.0, 84.0, 10.0
}),
MatrixRow<float>({
0.0, 83.0, 45.0
}),
MatrixRow<float>({
40.0, 83.0, 20.0
}),
MatrixRow<float>({
0.0, 83.0, 0
}),
MatrixRow<float>({
22.0, 83.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
2.0, 91.0, 49.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 26.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
2.0, 91.0, 40.0
}),
MatrixRow<float>({
22.0, 91.0, 60.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 38.0 }),
MatrixRow<float>({
2.0, 91.0, 22.0
}),
MatrixRow<float>({
2.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
17.0, 91.0, 50.0
}),
MatrixRow<float>({
40.0, 91.0, 44.0
}),
MatrixRow<float>({
17.0, 91.0, 22.0
}),
MatrixRow<float>({
40.0, 91.0, 45.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
3.0, 91.0, 30.0
}),
MatrixRow<float>({
22.0, 91.0, 55.0
}),
MatrixRow<float>({
3.0, 91.0, 0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
22.0, 91.0, 75.0
}),
MatrixRow<float>({
40.0, 87.0, 12.0
}),
MatrixRow<float>({
40.0, 87.0, 28.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
22.0, 87.0, 27.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
37.0, 87.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 45.0
}),
MatrixRow<float>({
40.0, 87.0, 25.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 23.0 }),
MatrixRow<float>({
40.0, 87.0, 55.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
40.0, 87.0, 40.0
}),
MatrixRow<float>({
40.0, 87.0, 20.0
}),
MatrixRow<float>({
22.0, 87.0, 17.0
}),
MatrixRow<float>({
40.0, 87.0, 18.0
}),
MatrixRow<float>({
22.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 38.0
}),
MatrixRow<float>({
40.0, 87.0, 32.0
}),
MatrixRow<float>({
40.0, 87.0, 14.0
}),
MatrixRow<float>({
22.0, 87.0, 10.0
}),
MatrixRow<float>({
22.0, 87.0, 14.0
}),
MatrixRow<float>({
37.0, 87.0, 12.0
}),
MatrixRow<float>({
22.0, 87.0, 22.0
}),
MatrixRow<float>({
37.0, 87.0, 15.0
}),
MatrixRow<float>({
40.0, 87.0, 19.0
}),
MatrixRow<float>({
40.0, 87.0, 48.0
}),
MatrixRow<float>({
40.0, 87.0, 24.0
}),
MatrixRow<float>({
15.0, 81.0, 7.0
}),
MatrixRow<float>({
15.0, 80.0, 8.0
}),
MatrixRow<float>({
15.0, 80.0, 10.0
}),
MatrixRow<float>({
29.0, 80.0, 16.0
}),
MatrixRow<float>({
40.0, 85.0, 100.0
}),
MatrixRow<float>({
40.0, 84.0, 45.0 }),
MatrixRow<float>({
40.0, 85.0, 7.0
}),
MatrixRow<float>({
40.0, 90.0, 13.0
}),
MatrixRow<float>({
40.0, 87.0, 6.0
}),
MatrixRow<float>({
40.0, 87.0, 16.0
}),
MatrixRow<float>({
40.0, 87.0, 7.0
}),
MatrixRow<float>({
40.0, 86.0, 20.0
}),
MatrixRow<float>({
40.0, 86.0, 45.0
}),
MatrixRow<float>({
40.0, 82.0, 7.0
}),
MatrixRow<float>({
37.0, 94.0, 25.0
}),
MatrixRow<float>({
37.0, 94.0, 14.0
}),
MatrixRow<float>({
40.0, 94.0, 100.0
}),
MatrixRow<float>({
40.0, 93.0, 48.0
}),
MatrixRow<float>({
3.0, 93.0, 24.0
}),
MatrixRow<float>({
37.0, 93.0, 17.0
}),
MatrixRow<float>({
40.0, 93.0, 50.0
}),
MatrixRow<float>({
3.0, 93.0, 48.0
}),
MatrixRow<float>({
40.0, 93.0, 35.0
}),
MatrixRow<float>({
40.0, 92.0, 28.0
}),
MatrixRow<float>({
40.0, 92.0, 45.0
}),
MatrixRow<float>({
29.0, 92.0, 23.0
}),
MatrixRow<float>({
40.0, 90.0, 55.0
}),
MatrixRow<float>({
40.0, 90.0, 38.0
}),
MatrixRow<float>({
40.0, 90.0, 36.0
}),
MatrixRow<float>({
40.0, 90.0, 43.0
}),
MatrixRow<float>({
22.0, 90.0, 28.0 }),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 50.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 26.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 30.0
}),
MatrixRow<float>({
15.0, 90.0, 35.0
}),
MatrixRow<float>({
15.0, 90.0, 28.0
}),
MatrixRow<float>({
15.0, 90.0, 20.0
}),
MatrixRow<float>({
15.0, 90.0, 0
}),
MatrixRow<float>({
15.0, 90.0, 40.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 60.0
}),
MatrixRow<float>({
40.0, 90.0, 48.0
}),
MatrixRow<float>({
40.0, 90.0, 20.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
22.0, 90.0, 0
}),
MatrixRow<float>({
37.0, 90.0, 34.0
}),
MatrixRow<float>({
40.0, 90.0, 150.0
}),
MatrixRow<float>({
22.0, 90.0, 25.0
}),
MatrixRow<float>({
40.0, 90.0, 30.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0
}),
MatrixRow<float>({
22.0, 90.0, 20.0
}),
MatrixRow<float>({
22.0, 90.0, 65.0
}),
MatrixRow<float>({
40.0, 90.0, 40.0 }),
MatrixRow<float>({
40.0, 91.0, 28.0
}),
MatrixRow<float>({
40.0, 91.0, 48.0
}),
MatrixRow<float>({
40.0, 91.0, 35.0
}),
MatrixRow<float>({
29.0, 91.0, 19.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
29.0, 91.0, 15.0
}),
MatrixRow<float>({
15.0, 91.0, 93.0
}),
MatrixRow<float>({
40.0, 91.0, 55.0
}),
MatrixRow<float>({
40.0, 91.0, 25.0
}),
MatrixRow<float>({
8.0, 91.0, 48.0
}),
MatrixRow<float>({
15.0, 91.0, 48.0
}),
MatrixRow<float>({
22.0, 91.0, 37.0
}),
MatrixRow<float>({
40.0, 91.0, 20.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
8.0, 91.0, 70.0
}),
MatrixRow<float>({
40.0, 91.0, 50.0
}),
MatrixRow<float>({
2.0, 91.0, 20.0
}),
MatrixRow<float>({
8.0, 91.0, 45.0
}),
MatrixRow<float>({
22.0, 91.0, 65.0
}),
MatrixRow<float>({
40.0, 91.0, 32.0
}),
MatrixRow<float>({
40.0, 91.0, 40.0
}),
MatrixRow<float>({
40.0, 91.0, 39.0
}),
MatrixRow<float>({
2.0, 91.0, 65.0
}),
MatrixRow<float>({
22.0, 86.0, 19.0
}),
MatrixRow<float>({
36.0, 86.0, 13.0 }),
}
});
