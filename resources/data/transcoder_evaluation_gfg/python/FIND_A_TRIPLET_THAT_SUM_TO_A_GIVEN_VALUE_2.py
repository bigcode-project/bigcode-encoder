# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( A , arr_size , sum ) :
    for i in range ( 0 , arr_size - 1 ) :
        s = set ( )
        curr_sum = sum - A [ i ]
        for j in range ( i + 1 , arr_size ) :
            if ( curr_sum - A [ j ] ) in s :
                print ( "Triplet is" , A [ i ] , ", " , A [ j ] , ", " , curr_sum - A [ j ] )
                return True
            s.add ( A [ j ] )
    return False


#TOFILL

if __name__ == '__main__':
    param = [
    ([1, 6, 8, 8, 9, 11, 13, 13, 15, 17, 21, 24, 38, 38, 42, 43, 46, 46, 47, 54, 55, 56, 57, 58, 60, 60, 60, 62, 63, 63, 65, 66, 67, 67, 69, 81, 84, 84, 85, 86, 95, 99],27,24,),
    ([20, -86, -24, 38, -32, -64, -72, 72, 68, 94, 18, -60, -4, -18, -18, -70, 6, -86, 46, -16, 46, -28],21,20,),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],17,13,),
    ([13, 96, 31, 39, 23, 39, 50, 10, 21, 64, 41, 54, 44, 97, 24, 91, 79, 86, 38, 49, 77, 71, 8, 98, 85, 36, 37, 65, 42, 48],17,18,),
    ([-86, -68, -58, -56, -54, -54, -48, -40, -36, -32, -26, -16, -14, -12, -12, -4, -4, -4, 0, 10, 22, 22, 30, 54, 62, 68, 88, 88],21,25,),
    ([0, 1, 1, 1, 1, 0, 0],5,3,),
    ([8, 8, 9, 13, 20, 24, 29, 52, 53, 96],9,8,),
    ([18, -92, -10, 26, 58, -48, 38, 66, -98, -72, 4, 76, -52, 20, 60, -56, 96, 60, -10, -26, -64, -66, -22, -86, 74, 82, 2, -14, 76, 82, 40, 70, -40, -2, -46, -38, 22, 98, 58],30,30,),
    ([1, 1, 1, 1],2,2,),
    ([72],0,0,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if f_filled(*parameters_set) == f_gold(*parameters_set):
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))