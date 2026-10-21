-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 }
-- output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out




let main (n: []i32) : []i32 = radixSort n 