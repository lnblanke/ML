#include <bits/stdc++.h>
using namespace std;

int main() {
    double t = 0;
    for (int i = 0; i <= 20; i += 2) {
        t += 3.0 * cos(7 * 3.1415926 * i) * ((i == 0 || i == 20)? 1 : 2);
    }
    printf("%.6f\n", t);
}