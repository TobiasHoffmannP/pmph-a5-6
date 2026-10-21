default: test

primes-seq:
	futhark c primes-seq.fut

test: radix.fut ref10000000.out
	futhark test --backend=cuda $<

ref10000000.out: primes-seq
	echo "10000000i64" | ./$< -b > $@

clean:
	# rm -rf ref10000000.out
	rm -rf *.c primes-flat primes-seq primes-naive primes-adhoc
	rm -rf *.actual *.expected
