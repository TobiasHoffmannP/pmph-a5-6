import "lib/github.com/diku-dk/sorts/radix_sort"
-- Primes: Flat-Parallel Version
-- ==
-- "100" input @ dataset1.in
-- "10000" input @ dataset2.in
-- "1000000" input @ dataset3.in
-- "100000000" input @ dataset4.in
-- nobench input { [9,8,7,6,5,4,3,2,1] }
-- output { [1,2,3,4,5,6,7,8,9] }

let main (n: []i32) : []i32 = 
    radix_sort i32.num_bits i32.get_bit n
