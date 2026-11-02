import "lib/github.com/diku-dk/sorts/radix_sort"
-- Radix sort
-- ==
-- "10" input @ data/dataset10.in
-- "100" input @ data/dataset100.in
-- "10000" input @ data/dataset10000.in
-- "1000000" input @ data/dataset1000000.in
-- "10000000" input @ data/dataset10000000.in
-- "276447232" input @ data/dataset276447232.in
-- nobench input { [9,8,7,6,5,4,3,2,1] }
-- output { [1,2,3,4,5,6,7,8,9] }

let main (n: []i32) : []i32 = 
    radix_sort i32.num_bits i32.get_bit n
