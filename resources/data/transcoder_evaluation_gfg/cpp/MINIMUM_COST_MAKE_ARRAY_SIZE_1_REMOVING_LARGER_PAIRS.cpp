// Copyright (c) 2019-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <bits/stdc++.h>
using namespace std;
int f_gold ( int a [ ], int n ) {
  return ( n - 1 ) * ( * min_element ( a, a + n ) );
}


//TOFILL

int main() {
    int n_success = 0;
    vector<vector<int>> param0 {{1,2,3,4,7,8,10,10,16,20,22,22,23,23,23,27,29,32,35,39,41,46,51,53,54,59,59,60,61,69,70,70,79,79,81,84,90,91,98},{-6,10},{0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1},{20,61,92,45,75,26,83,5,85,27,39,88,36,39,83,41,56,77,39,69,72,98,39,15,29,69,64,92,96,49,59,62,53,82,40,37,29,41},{-88,-60,-60,-58,-56,-56,-46,-44,-40,-38,-32,-28,-22,-18,-12,-4,-2,10,24,24,28,38,42,46,54,64,72,74,78,96,96},{0,1,1,1,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,1,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1},{1,4,6,9,10,12,17,17,18,21,22,26,26,31,32,33,34,39,42,43,45,46,48,50,53,53,54,55,60,61,62,63,63,64,70,70,70,71,71,78,86,88,91,92,95,95,96,97,99},{-42,44,-80,-60,48,66,54,-76,26,-74,-10,46,-50,42,-26,-24,-14,36,-2,-26,16,-10,20,-88,-78},{0,0,0,0,1,1,1,1,1,1},{65,32,66,82,86,98,15,33,57,3,73,45,90,82,33,99,44,76,50,89,5,7,64}};
    vector<int> param1 {25,1,15,23,26,39,28,19,5,22};
    for(int i = 0; i < param0.size(); ++i)
    {
        if(f_filled(&param0[i].front(),param1[i]) == f_gold(&param0[i].front(),param1[i]))
        {
            n_success+=1;
        }
    }
    cout << "#Results:" << " " << n_success << ", " << param0.size();
    return 0;
}