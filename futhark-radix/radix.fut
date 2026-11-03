import "lib/github.com/diku-dk/sorts/radix_sort"
-- Radix sort
-- ==
-- "100" input @ data/dataset100.in
-- "1000" input @ data/dataset1000.in
-- "4000" input @ data/dataset4000.in
-- "16000" input @ data/dataset16000.in
-- "64000" input @ data/dataset64000.in
-- "1000000" input @ data/dataset1000000.in
-- "4000000" input @ data/dataset4000000.in
-- nobench input { [9,8,7,6,5,4,3,2,1] }
-- output { [1,2,3,4,5,6,7,8,9] }

let main (n: []i32) : []i32 = 
    radix_sort i32.num_bits i32.get_bit n

-- "64000000" input @ data/dataset64000000.in // Add this above to also bench for a input with size 64000000