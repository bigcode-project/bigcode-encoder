# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    a = [ 0 for i in range ( n ) ]
    b = [ 0 for i in range ( n ) ]
    a [ 0 ] = b [ 0 ] = 1
    for i in range ( 1 , n ) :
        a [ i ] = a [ i - 1 ] + b [ i - 1 ]
        b [ i ] = a [ i - 1 ]
    return a [ n - 1 ] + b [ n - 1 ]


#TOFILL

if __name__ == '__main__':
    param = [
    (86,),
    (75,),
    (14,),
    (5,),
    (41,),
    (35,),
    (30,),
    (89,),
    (84,),
    (53,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if f_filled(*parameters_set) == f_gold(*parameters_set):
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))