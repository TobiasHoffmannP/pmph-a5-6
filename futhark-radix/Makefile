.PHONY: test bench clean

test:
	futhark test --backend=cuda radix.fut

bench:
	futhark dataset --i32-bounds=0:9 -g '[100]i32' > dataset1.in
	futhark dataset --i32-bounds=0:9 -g '[10000]i32' > dataset2.in
	futhark dataset --i32-bounds=0:9 -g '[1000000]i32' > dataset3.in
	futhark dataset --i32-bounds=0:9 -g '[100000000]i32' > dataset4.in
	futhark bench --backend=cuda radix.fut
clean:
	rm -rf *.c *.h *.json radix *.actual *.expected dataset*.in
