# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    i = 1
    the_sum = 0
    while the_sum < n :
        the_sum += i
        if the_sum == n :
            return True
        i += 2
    return False


#TOFILL

if __name__ == '__main__':
    param = [
    (1,),
    (4,),
    (9,),
    (25,),
    (36,),
    (3,),
    (121,),
    (14,),
    (17,),
    (80,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if f_filled(*parameters_set) == f_gold(*parameters_set):
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))